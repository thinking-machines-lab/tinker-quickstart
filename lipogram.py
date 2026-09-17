import asyncio
import logging
import os
import random
import re
from collections.abc import Sequence
from typing import cast
from urllib.parse import urlencode

import numpy as np
import tinker
from tinker import TensorData, types
from tml_renderers import chat, tokenizers, v0
from tml_renderers import tinker as tml_tinker

# fmt:off
logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S", format="\033[2m%(asctime)s %(name)s\033[0m %(message)s",)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("demo")
# fmt:on

TRAINING_PROMPTS = [
    "Tell me about the ocean.",
    "What makes a good friend?",
    "Describe a typical morning routine.",
    "Explain how rainbows form.",
    "Tell me a short story about a dog.",
    "Why is the sky blue?",
]

EVAL_PROMPT = "Tell me about outer space."

BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 8
GROUP_SIZE = 8
TRAINING_STEPS = 10
MAX_TOKENS = 200
LEARNING_RATE = 8e-4

JUDGE_WEIGHT = 2.0  # higher means weigh quality more heavily
JUDGE_MODEL = "thinkingmachines/Inkling-Small"  # the model that grades answers


def e_rate(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    return letters.count("e") / max(len(letters), 1)


class Judge:
    """Grades an answer to a prompt from 1 (gibberish) to 5 (high quality)."""

    async def grade(self, prompt: str, response: str) -> float:
        raise NotImplementedError


async def reward(prompt: str, text: str, judge: Judge) -> float:
    e_penalty = min(1.5, e_rate(text) / 0.10)  # 0%->0, 5%->-0.5, 10%->-1, 15%+->-1.5
    grade = await judge.grade(prompt, text)
    return -e_penalty + JUDGE_WEIGHT * (grade / 5)


async def main() -> None:
    # add any other metadata you want to track here for each run so you can see them in the leaderboard!
    user_metadata: dict[str, str] = {
        "description": "Lipogram Demo",
    }
    leaderboard_name = os.getenv("LEADERBOARD_NAME")
    if leaderboard_name:
        user_metadata["name"] = leaderboard_name
    else:
        email = await get_email()
        if email:
            user_metadata["email"] = email

    service = tinker.ServiceClient(user_metadata=user_metadata)
    train = await service.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=LORA_RANK
    )
    tok = train.get_tokenizer()

    def render(question: str) -> list[int]:
        return cast(
            list[int],
            tok.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=False,
            ),
        )

    judge = LlmJudge(
        await service.create_sampling_client_async(base_model=JUDGE_MODEL),
        tokenizers.o200k_base_chat(),
    )

    save = await train.save_weights_for_sampler_async(
        name="lipogram-000", ttl_seconds=7 * 24 * 3600
    )
    checkpoint = await save.result_async()
    path = checkpoint.path
    log_checkpoint("lipogram-000", checkpoint, TRAINING_PROMPTS[0])

    for step in range(TRAINING_STEPS):
        sampler = await service.create_sampling_client_async(model_path=path)
        prompt_toks = [render(q) for q in TRAINING_PROMPTS]

        # sample GROUP_SIZE responses for each prompt in parallel
        results = await asyncio.gather(
            *[
                sampler.sample_async(
                    prompt=types.ModelInput.from_ints(ptoks),
                    num_samples=GROUP_SIZE,
                    sampling_params=types.SamplingParams(
                        max_tokens=MAX_TOKENS, temperature=1.0
                    ),
                )
                for ptoks in prompt_toks
            ]
        )
        texts = [[str(tok.decode(seq.tokens)) for seq in r.sequences] for r in results]
        grouped_rewards = await asyncio.gather(
            *[
                asyncio.gather(*[reward(question, t, judge) for t in group])
                for question, group in zip(TRAINING_PROMPTS, texts)
            ]
        )

        step_texts = [t for group in texts for t in group]
        step_rewards = [r for group in grouped_rewards for r in group]
        example = random.choice(step_texts)

        data = []
        for ptoks, result, rewards in zip(prompt_toks, results, grouped_rewards):
            baseline = sum(rewards) / len(rewards)
            if all(r == rewards[0] for r in rewards):
                continue
            std = (sum((r - baseline) ** 2 for r in rewards) / len(rewards)) ** 0.5

            for seq, r in zip(result.sequences, rewards):
                adv = (r - baseline) / std
                ob_len = len(ptoks) - 1
                data.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(ptoks + seq.tokens[:-1]),
                        loss_fn_inputs={
                            # 0 to indicate not to train on prompt tokens, only on the generated tokens
                            "target_tokens": TensorData.from_numpy(
                                np.array([0] * ob_len + seq.tokens)
                            ),
                            # the logprobs of the generated tokens
                            "logprobs": TensorData.from_numpy(
                                np.array(
                                    [0.0] * ob_len + cast(list[float], seq.logprobs)
                                )
                            ),
                            # convert our sequence level advantage to a token level advantage
                            # applies to each sampled token in the sequence.
                            "advantages": TensorData.from_numpy(
                                np.array([0.0] * ob_len + [adv] * len(seq.tokens))
                            ),
                        },
                    )
                )

        # these are all of the API calls to Tinker!
        if data:
            # forward pass to compute the loss, backward pass to compute the gradients and store them
            fb = await train.forward_backward_async(data, loss_fn="importance_sampling")
            # optimizer step to update the model weights using the gradients calculated in backwards pass
            opt = await train.optim_step_async(
                types.AdamParams(learning_rate=LEARNING_RATE)
            )

            # above futures return once the operations are enqueued, but not yet finished
            # we want to enqueue both at once and then await them here to improve pipelining efficiency.
            # Learn more: https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/#clock-cycles
            await fb
            await opt

        name = f"lipogram-{step + 1:03d}"
        save = await train.save_weights_for_sampler_async(
            name=name, ttl_seconds=7 * 24 * 3600
        )
        checkpoint = await save.result_async()
        path = checkpoint.path

        mean_reward = sum(step_rewards) / len(step_rewards)
        step_letters = [c for text in step_texts for c in text.lower() if c.isalpha()]
        step_e_rate = step_letters.count("e") / max(len(step_letters), 1)
        log.info(
            f"\033[1mstep {step:2d}  mean reward = {mean_reward:.3f}, mean e rate = {100 * step_e_rate:.1f}%\033[0m"
        )
        log.info(
            f"  example ({100 * e_rate(example):.1f}% e): {highlight_e(example[:200])}"
        )
        log_checkpoint(name, checkpoint, TRAINING_PROMPTS[0])

    me = await service.create_sampling_client_async(model_path=path)
    log.info(f"saved {TRAINING_STEPS + 1} checkpoints, latest: {path}")

    question = EVAL_PROMPT
    test = await me.sample_async(
        prompt=types.ModelInput.from_ints(render(question)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7),
    )
    final = str(tok.decode(test.sequences[0].tokens))
    log.info(f"Final result: ({100 * e_rate(final):.1f}% e):\n{highlight_e(final)}")
    log.info(f"judge says: {await judge.grade(question, final):.0f}/10")


GRADE_INSTRUCTIONS = """Grade the following response on a scale of 1 to 5. Scoring guidelines:
If the response is incoherent, contains any non-English words, or repeats punctuation marks over and over, return 1.
If the response is repetitive or the same phrase over and over, return 2.
If the response is in English, coherent but not a response to the prompt, return 3.
If the response is in English, on topic and clear, return 4.
If the response is in English, high quality, on-topic, and well-written, return 5.

This is a snippet of the response and not the entire response. Grade only on the response so far. Do not decrease score for cut off responses.

Return one number from 1 to 5 and nothing else."""


class LlmJudge(Judge):
    """Asks a second model for a grade instead of spelling the rules out by hand."""

    def __init__(
        self,
        sampler: tinker.SamplingClient,
        tokenizer: tokenizers.O200kBaseChatTokenizer,
    ) -> None:
        self.sampler = sampler
        self.renderer = v0.Renderer(tokenizer)

    async def grade(self, prompt: str, response: str) -> float:
        messages = chat.OpenAIMessage.from_oss_messages(
            [
                {"role": "system", "content": GRADE_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"Question: {prompt}\n\nAnswer: {response}",
                },
            ]
        )
        spans, parser = self.renderer.render_for_completion_with_effort(messages, 0.0)
        result = await self.sampler.sample_async(
            prompt=tml_tinker.token_spans_to_tinker_model_input(
                cast(Sequence[tml_tinker.TokenSpanWrapper], spans)
            ),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=8, temperature=0.0, stop=self.renderer.stop()
            ),
        )
        reply = "".join(
            message.content.text
            for message in parser.parse_tokens(result.sequences[0].tokens)
            if isinstance(message.content, chat.Text)
        )
        number = re.search(r"\d+", reply)
        if not number:
            return 1.0
        return float(min(10, max(1, int(number.group()))))


def highlight_e(text: str) -> str:
    return re.sub(r"[eE]", "\033[91m\\g<0>\033[0m", text)


async def get_email() -> str | None:
    service = tinker.ServiceClient()
    rest_client = service.create_rest_client()
    identity = await rest_client.whoami().result_async()
    return identity.email


def log_checkpoint(
    name: str, checkpoint: types.SaveWeightsForSamplerResponse, prompt: str
) -> None:
    """Logs a checkpoint's path along with OSC-8 hyperlinks to chat with it in the playground."""
    playground_query = urlencode(
        {
            "mode": "checkpoint",
            "checkpoint": checkpoint.path,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none",
            "prompt": prompt,
        }
    )
    playground_link = f"\033]8;;https://tinker.thinkingmachines.ai/playground?{playground_query}\033\\Chat in Playground\033]8;;\033\\"
    log.info(f"  [{name}] saved: {checkpoint.path}")
    log.info(f"  {playground_link}")


if __name__ == "__main__":
    asyncio.run(main())
