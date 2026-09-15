"""Train a model to answer without the letter 'e', then investigate its reward hacks.

Read from train() downward: generate groups, score them, compute advantages,
update the model, and save a checkpoint. Run with: uv run lipogram.py
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import tinker
from tinker import TensorData, types
from tinker_cookbook import renderers

# Add your name here to include it in the user metadata for each session
NAME: str | None = None

# Description to include in user metadata to identify this run
DESCRIPTION: str = "Initial run"

BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 8
NUM_STEPS = 10
LEARNING_RATE = 8e-4
GROUP_SIZE = 8
MAX_TOKENS = 200
JUDGE_WEIGHT = 2.0
CHECKPOINT_TTL_SECONDS = 7 * 24 * 60 * 60

TRAIN_PROMPTS = [
    "Tell me about the ocean.",
    "What makes a good friend?",
    "Describe a typical morning routine.",
    "Explain how rainbows form.",
    "Tell me a short story about a dog.",
    "Why is the sky blue?",
    "Tell me about redwood trees.",
    "How are fossils formed?",
]
HELD_OUT_PROMPT = "Tell me about outer space."


async def train(
    judge: Judge | None = None, description: str | None = DESCRIPTION
) -> None:
    # A ServiceClient starts a session; a TrainingClient updates our LoRA weights.
    identity_service = tinker.ServiceClient()
    identity = await identity_service.create_rest_client().whoami()
    await identity_service.close("success")
    if identity.email is None:
        raise RuntimeError("Tinker did not return an email for the authenticated user")

    # user metadata to associate with a given session to help us identify it later
    user_metadata: dict[str, str] = {
        "email": identity.email,
        "lora_rank": str(LORA_RANK),
        "learning_rate": str(LEARNING_RATE),
        "group_size": str(GROUP_SIZE),
        "max_tokens": str(MAX_TOKENS),
        "judge_weight": str(JUDGE_WEIGHT),
        "num_steps": str(NUM_STEPS),
        "num_train_prompts": str(len(TRAIN_PROMPTS)),
    }
    if NAME is not None:
        user_metadata["name"] = NAME
    if description is not None:
        user_metadata["description"] = description
    service = tinker.ServiceClient(user_metadata=user_metadata)
    training_client: tinker.TrainingClient = (
        await service.create_lora_training_client_async(
            base_model=BASE_MODEL, rank=LORA_RANK
        )
    )
    # Use the renderer matching the model's chat format, with reasoning disabled.
    renderer: renderers.Renderer = renderers.get_renderer(
        "qwen3_5_disable_thinking", training_client.get_tokenizer()
    )

    judge = judge or NoopJudge()

    history: list[StepMetrics] = []
    asyncio.create_task(save_checkpoint(training_client, "lipogram-000"))
    for step in range(1, NUM_STEPS + 1):
        start = time.time()
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )
        metrics = await rl_step(training_client, sampling_client, renderer, judge)
        elapsed = int(time.time() - start)
        history.append(metrics)
        print(
            f"step {step:02d}  reward={metrics.mean_reward:.3f}  "
            f"e rate={metrics.mean_e_rate:.1%}  judge={metrics.mean_judge_score:.1f}/5  {elapsed}s"
        )
        print(f"  {highlight_e(metrics.example[:200])}")
        asyncio.create_task(save_checkpoint(training_client, f"lipogram-{step:03d}"))
        plot_results(history)

    # Try the final weights on a question that was never used for training.
    final_client = await training_client.save_weights_and_get_sampling_client_async()
    held_out = await generate_group(
        final_client, renderer, HELD_OUT_PROMPT, num_samples=1
    )
    print(f"\nheld out: {HELD_OUT_PROMPT}")
    print(f"  {highlight_e(held_out.rollouts[0].text)}")


async def rl_step(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    judge: Judge,
) -> StepMetrics:
    """Generate answers, compare rewards within each prompt, and update the policy."""
    groups: list[Group] = await asyncio.gather(
        *[generate_group(sampling_client, renderer, prompt) for prompt in TRAIN_PROMPTS]
    )
    training_groups: list[TrainingGroup] = await asyncio.gather(
        *[score_group(group, judge) for group in groups]
    )
    data: list[types.Datum] = [
        to_datum(group.prompt_tokens, rollout)
        for group in training_groups
        for rollout in group.rollouts
        if any(rollout.advantages)  # identical rewards give no learning signal
    ]
    if data:
        # Queue both operations before waiting so Tinker can schedule them together.
        forward_backward = await training_client.forward_backward_async(
            data, loss_fn="importance_sampling"
        )
        optim_step = await training_client.optim_step_async(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )
        await forward_backward.result_async()
        await optim_step.result_async()

    return summarize_step(training_groups)


async def generate_group(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    prompt: str,
    num_samples: int = GROUP_SIZE,
) -> Group:
    """Generate sibling answers to one prompt, retaining each token's logprob."""
    model_input: types.ModelInput = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt}]
    )
    response: types.SampleResponse = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=types.SamplingParams(
            max_tokens=MAX_TOKENS, temperature=1.0, stop=renderer.get_stop_sequences()
        ),
    )
    rollouts: list[Rollout] = []
    for sequence in response.sequences:
        message, _ = renderer.parse_response(sequence.tokens)
        assert sequence.logprobs is not None  # needed by the importance-sampling loss
        rollouts.append(
            Rollout(
                text=renderers.get_text_content(message),
                tokens=list(sequence.tokens),
                logprobs=list(sequence.logprobs),
            )
        )
    return Group(prompt, model_input.to_ints(), rollouts)


async def score_group(group: Group, judge: Judge) -> TrainingGroup:
    """Attach scores and group-relative advantages without changing generation data."""
    rewards: list[Reward] = await asyncio.gather(
        *[reward(group.prompt, rollout.text, judge) for rollout in group.rollouts]
    )
    mean = sum(r.total for r in rewards) / len(rewards)
    std = math.sqrt(sum((r.total - mean) ** 2 for r in rewards) / len(rewards))
    training_rollouts: list[TrainingRollout] = []
    for rollout, score in zip(group.rollouts, rewards, strict=True):
        # Better than the group average gets a positive advantage; worse, negative.
        advantage = (score.total - mean) / std if std > 0 else 0.0
        training_rollouts.append(
            TrainingRollout(rollout, score, [advantage] * len(rollout.tokens))
        )
    return TrainingGroup(group.prompt, group.prompt_tokens, training_rollouts)


def to_datum(prompt_tokens: list[int], sample: TrainingRollout) -> types.Datum:
    """Align inputs with next-token targets and train only on the sampled answer."""
    full_sequence = prompt_tokens + sample.rollout.tokens
    prefix_length = len(prompt_tokens) - 1
    return types.Datum(
        # Each input position predicts the following token.
        model_input=types.ModelInput.from_ints(full_sequence[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.array(full_sequence[1:])),
            "logprobs": TensorData.from_numpy(
                np.array([0.0] * prefix_length + sample.rollout.logprobs)
            ),
            # Zero advantages mask out the prompt; its tokens receive no reward.
            "advantages": TensorData.from_numpy(
                np.array([0.0] * prefix_length + sample.advantages)
            ),
        },
    )


def summarize_step(groups: list[TrainingGroup]) -> StepMetrics:
    rollouts = [rollout for group in groups for rollout in group.rollouts]
    best_group = max(groups, key=lambda g: max(r.reward.total for r in g.rollouts))
    best = max(best_group.rollouts, key=lambda r: r.reward.total)
    return StepMetrics(
        mean_reward=sum(r.reward.total for r in rollouts) / len(rollouts),
        mean_e_rate=sum(r.reward.e_rate for r in rollouts) / len(rollouts),
        mean_judge_score=sum(r.reward.judge_score for r in rollouts) / len(rollouts),
        example_prompt=best_group.prompt,
        example=best.rollout.text,
    )


async def reward(prompt: str, text: str, judge: Judge) -> Reward:
    rate = e_rate(text)
    # Scale by a typical 10% e rate and cap the penalty at 1.5.
    e_penalty = min(1.5, rate / 0.10)
    grade = await judge.grade(prompt, text)
    return Reward(
        total=-e_penalty + JUDGE_WEIGHT * (grade / 5),
        e_rate=rate,
        judge_score=grade,
    )


class Judge(Protocol):
    """A quality judge scores an answer from 1 (poor) to 5 (good)."""

    async def grade(self, prompt: str, response: str) -> float: ...


class NoopJudge:
    """
    A judge that always returns 1.0.
    """

    async def grade(self, prompt: str, response: str) -> float:
        return 1.0


def e_rate(text: str) -> float:
    """Fraction of alphabetic characters that are e or E."""
    letters = [c for c in text.lower() if c.isalpha()]
    return letters.count("e") / max(len(letters), 1)


async def save_checkpoint(training_client: tinker.TrainingClient, name: str) -> str:
    """Save sampler weights for seven days and link to the checkpoint and Playground."""
    save = await training_client.save_weights_for_sampler_async(
        name=name, ttl_seconds=CHECKPOINT_TTL_SECONDS
    )
    checkpoint: types.SaveWeightsForSamplerResponse = await save.result_async()
    checkpoint_url = checkpoint.get_console_url()
    playground_query = urlencode(
        {
            "mode": "checkpoint",
            "checkpoint": checkpoint.path,
            "max_tokens": 1024,
            "reasoning_effort": "none",
            "prompt": HELD_OUT_PROMPT,
        }
    )
    # OSC-8 hyperlinks: \e]8;;URL\e\\TEXT\e]8;;\e\\
    checkpoint_link = f"\033]8;;{checkpoint_url}\033\\Open checkpoint\033]8;;\033\\"
    playground_link = f"\033]8;;https://tinker.thinkingmachines.ai/playground?{playground_query}\033\\Chat in Playground\033]8;;\033\\"
    print(f"  [{name}] saved: {checkpoint.path}")
    print(f"  {checkpoint_link}  {playground_link}")
    return checkpoint.path


def highlight_e(text: str) -> str:
    return re.sub(r"[eE]", "\033[91m\\g<0>\033[0m", text)


def plot_results(history: list[StepMetrics]) -> None:
    panels = [
        ("Mean reward", [m.mean_reward for m in history]),
        ("Mean e rate (%)", [100 * m.mean_e_rate for m in history]),
        ("Mean judge score (1–5)", [m.mean_judge_score for m in history]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4))
    for ax, (label, values) in zip(axes, panels, strict=True):
        ax.plot(range(1, len(history) + 1), values, marker=".")
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_run.png")
    plt.close(fig)


@dataclass
class Rollout:
    """One generated answer, before scoring; token IDs and logprobs stay aligned."""

    text: str
    tokens: list[int]
    logprobs: list[float]


@dataclass
class Group:
    """Answers sampled for the same prompt, sharing its rendered token sequence."""

    prompt: str
    prompt_tokens: list[int]
    rollouts: list[Rollout]


@dataclass
class Reward:
    """An answer's total reward and its spelling and quality measurements."""

    total: float
    e_rate: float
    judge_score: float


@dataclass
class TrainingRollout:
    """A generated answer plus its reward and one training advantage per token."""

    rollout: Rollout
    reward: Reward
    advantages: list[float]


@dataclass
class TrainingGroup:
    """A prompt's scored answers, with advantages computed against their siblings."""

    prompt: str
    prompt_tokens: list[int]
    rollouts: list[TrainingRollout]


@dataclass
class StepMetrics:
    """Batch averages and a best-scoring example for inspecting learning progress."""

    mean_reward: float
    mean_e_rate: float
    mean_judge_score: float
    example_prompt: str
    example: str


if __name__ == "__main__":
    asyncio.run(train())
