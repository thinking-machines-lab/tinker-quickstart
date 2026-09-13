# Tinker Quickstart

Train a model to answer questions without using the letter **e**. Writing that
avoids a particular letter is called a *lipogram*. In this tutorial, you'll use
reinforcement learning (RL) to fine-tune a model to respond while minimizing the use of the letter **e**.

## Setup
### Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From this repository, install the dependencies and authenticate:

```bash
uv sync
```

### Authenticate
If you are running on a dev node, you will be autheticated automatically. 

If you are running this on your own machine, you will need to first ask to be invited to the TML Onboarding organization in Tinker.

Then, to authenticate via the Tinker CLI, run:
```bash
uv run tinker auth login
```

You can also set `TINKER_API_KEY` to a key from the
[API Keys page](https://tinker.thinkingmachines.ai/keys).

## Run the script

All of the starter code is in [lipogram.py](./lipogram.py). To kick off a training run:

```bash
uv run lipogram.py
```

## Basic Tinker Concepts
Tinker is a platform to fine-tune open weight LLMs while keeping as much of the logic in the user's code as possible. There are four key pieces of Tinker functionality to be familiar with:
- `SamplingClient` to sample from a model (eihter a base model, checkpoint, or in progress trained model)
- `TrainingClient.forward_backward` to compute forward and backward passes over a set of tokens with loss functions defined by `Datum` objects
- `TrainingClient.optim_step` to perform the optimization step to update model weights
- `TrainingClient.save_state` to save a training checkpoint (maintains optimizer state) and `TrainingClient.save_weights_for_sampler` to save a sampling only format that can be sampled from in Tinker Playground.


## Read the training loop
Most of the core logic happens in the `train` function. 

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
