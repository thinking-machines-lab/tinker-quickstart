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

The script saves a checkpoint after each training step, that includes a link to the [Tinker Playground](https://tinker.thinkingmachines.ai/playground), where you can chat with the checkpoint of the model at that training step.

## Basic Tinker Concepts

Tinker is a platform to fine-tune open weight LLMs while keeping as much of the logic in the user's code as possible. There are four key pieces of Tinker functionality to be familiar with:

- `SamplingClient` to sample from a model (eihter a base model, checkpoint, or in progress trained model)
- `TrainingClient.forward_backward` to compute forward and backward passes over a set of tokens with loss functions defined by `Datum` objects
- `TrainingClient.optim_step` to perform the optimization step to update model weights
- `TrainingClient.save_state` to save a training checkpoint (maintains optimizer state) and `TrainingClient.save_weights_for_sampler` to save a sampling only format that can be sampled from in Tinker Playground.

See the the [Tinker docs](https://tinker-docs.thinkingmachines.ai/).

## Understand the core training loop

Most of the core logic happens in the `train` function, that handles iteration over training steps.

Each `rl_step` follows the same sequence:

1. **Generate groups.** A `Rollout` represents a single response to a prompt, including the logprobs of each token in the response. Responses to the same prompt are grouped together in a `Group`.
2. **Score the answers.** Compute the reward for each answer. To start, the reward is a simple penalty for the fraction of letters that are e.
3. **Compute advantages.** Normalize rewards within each group to compute advantages for each rollout.
4. **Update the model.** `to_datum` converts to `Datum` objects, which encode the information needed to perform a forward/backward pass on Tinker's servers.

## Gibberish Answers

If you ran the above script with no modifications, you will see that while the model stops using the letter **e**, it very quickly degenerates into gibberish answers.

This is known as **reward hacking**. The model finds a way to increase reward (reduce the rate of e usage) without actually writing meaningful responses.

The straightforward fix is to add some method to assess the quality of answers. See the `Judge` protocol and the `BasicJudge` implementation for a pure text method of evaluating answer quality. Then update the `reward` function to actually use the judge's score.

## Continued improvement
When running with the `BasicJudge`, you'll likely notice that the model does improve in that it's at least generating real words and not just repeating the same text over and over again. However, it's likely still not generating answers that are coherent and relevant to the question.

Can you find a way to evaluate the semantic quality of the responses and not just the syntax?

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
6. **Context distillation.** Prompt a larger model to generate answers conditioned on a prefix that explicitly mentions avoiding the letter **e** before starting RL training.

