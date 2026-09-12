# Tinker Quickstart

Train a model to answer questions without using the letter **e**. Writing that
avoids a particular letter is called a *lipogram*. In this tutorial, you'll use
reinforcement learning (RL) to teach that behavior, then investigate what happens
when a model learns to satisfy your reward without doing what you intended.

## Setup

### Account

Create a [Tinker account](https://tinker.thinkingmachines.ai/) and set up
[billing](https://tinker.thinkingmachines.ai/billing/balance). Training uses your
Tinker credits; see the [model pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/).

### Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From this repository, install the dependencies and authenticate:

```bash
uv sync
uv run tinker auth login
```

You can also set `TINKER_API_KEY` to a key from the
[API Keys page](https://tinker.thinkingmachines.ai/keys).

## Train a model

All of the starter code is in [lipogram.py](./lipogram.py). Run:

```bash
uv run lipogram.py
```

The model starts as `Qwen/Qwen3.5-4B`. Each step samples several answers to each
question, rewards answers with fewer e's and a good basic quality score, and
updates the model toward the better answers. The prompts don't mention the
letter constraint: we want the model to learn it through training.

The terminal shows average reward, average e rate, the basic judge's score, and
a best-scoring answer with e's highlighted. The script saves an initial sampler
checkpoint and another after **every step**, each with a **seven-day lifetime**.
Each save prints a checkpoint details link and a Playground link with a prefilled
prompt, `max_tokens=1024`, and reasoning disabled. Open checkpoints from different
steps to compare how the answers change. These checkpoints contain weights for
sampling; they do not include optimizer state for resuming training.

At the end, the script tries a held-out question and saves `training_run.png`,
showing mean reward, mean e rate, and mean judge score. Lower `NUM_STEPS` in the
script for a shorter experiment.

## Read the training loop

Start with `train`, the first function. It creates a few key objects:

- `ServiceClient` starts a Tinker session.
- `TrainingClient` updates the model using [LoRA](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/),
  a small set of trainable weights added to the model.
- `SamplingClient` generates answers from the latest saved weights.
- A renderer handles the model's chat format, generation boundaries, and response
  parsing. Qwen uses the Cookbook's `qwen3_5_disable_thinking` renderer. The
  installed `tml-renderers` package provides the corresponding tools for Inkling.

Each `rl_step` follows the same sequence:

1. **Generate groups.** A `Rollout` contains one answer's text, tokens, and the
   log probability of each sampled token. A `Group` collects answers to the same
   prompt and stores the prompt's tokens once.
2. **Score the answers.** The reward combines a penalty for the fraction of
   letters that are e with the `BasicJudge` quality score.
3. **Compute advantages.** An advantage measures how an answer scored compared
   with its siblings. `TrainingRollout` and `TrainingGroup` keep rewards and
   advantages separate from the original generation data. Each answer's tokens
   receive the same advantage; equal-reward groups contribute no update.
4. **Update the model.** `to_datum` aligns next-token targets and masks out prompt
   tokens. Tinker computes the importance-sampling loss and takes an optimizer
   step. The sampled token logprobs are used by this loss; there is no KL penalty.

Helpers follow their callers so you can read from the overall flow down into
each operation. For more detail, use the [Tinker docs](https://tinker-docs.thinkingmachines.ai/)
or their [LLM-friendly index](https://tinker-docs.thinkingmachines.ai/llms.txt).

## Fix the reward hack

Look at the answers, not just the reward curve. We've already added basic checks
to discourage gibberish: answers must be long enough, use mostly ASCII letters,
and avoid obvious repetition. Yet many sentences can still make no sense, even
when the judge gives them full marks and their e rate is low.

This is **reward hacking**: the model finds a way to score well without meeting
our real goal. The basic judge checks surface patterns; it doesn't understand
whether an answer makes sense or addresses the question.

**How could you change the reward so that the model learns to write meaningful
answers while still avoiding e?** Inspect `reward` and `BasicJudge`, make a
change, and compare the results on both training and new prompts. Check whether
improved scores correspond to answers you'd actually want to read.

<details>
<summary>Hint:</summary>

Could another LLM judge whether an answer is coherent and relevant to the
question? Try `thinkingmachines/Inkling-Small` through Tinker. Think about what
information the judge needs and how its assessment should affect the reward.
The [Inkling rendering guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/tml-renderers/)
shows how to format and parse its messages.

</details>

## Extended challenges

1. **Combine judges.** Combine the basic text judge and the LLM judge in a
   sensible way. Consider which checks each judge is best suited to perform.
2. **Use a rubric.** Update the LLM judge to assess explicit criteria rather than
   returning a single overall score.
3. **Compare token logprobs.** Compute the difference in log probabilities between
   the model you're training and a frozen reference copy of the starting model.
   Inspect how those differences change during training.
4. **Assign token-level penalties.** Penalize tokens that use e instead of
   penalizing the entire sequence.
5. **Teach by example.** Manually write some high-quality answers that avoid e
   and use [supervised fine-tuning (SFT)](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/)
   to train on those examples.
6. **Generate a starting dataset.** Explicitly prompt a larger model to write
   answers without e. Review its examples, then SFT your model on them before
   running RL. Compare this with starting RL directly.
