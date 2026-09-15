# Tinker Quickstart

Train a model to answer questions without using the letter **e**. Writing that
avoids a particular letter is called a *lipogram*. In this tutorial, you'll use
reinforcement learning (RL) to fine-tune a model to respond while minimizing the use of the letter **e**.

## Setup

### Account Setup

Sign into your account in the [Tinker Console](https://tinker.thinkingmachines.ai/).

Make sure to set up [Billing](https://tinker.thinkingmachines.ai/billing/balance) so that you can begin training models. This quickstart uses very small models, so it should cost you no more than $5. See the [pricing page](https://tinker-docs.thinkingmachines.ai/tinker/models/) for more details.

### Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From this repository, install the dependencies:

```bash
uv sync
```

### Authenticating

Run `uv run tinker auth login` from this repo to authenticate your user.

Now any code that you run will be automatically authenticated.

It is also possible to manually generate an API key through the [API Keys page](https://tinker.thinkingmachines.ai/keys). This API key can be set through the environment variable `TINKER_API_KEY`.

## Run the script

All of the starter code is in [lipogram.py](./lipogram.py) and is ready to run:

```bash
uv run lipogram.py
```
## Basic Tinker Concepts

Tinker is a platform to fine-tune open weight LLMs while keeping as much of the logic running on the user's machine as possible. Key pieces of Tinker functionality to be familiar with:

- [`ServiceClient`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/) maintains an active connected "session" with Tinker, and is used to create both sampling and training clients. See the [Tinker Console Sessions](https://tinker.thinkingmachines.ai/sessions) page to view all of your current or past sessions.
- [`SamplingClient`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/) to `.sample` from a model (either a base model, checkpoint, or in progress trained model).
- [`TrainingClient`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/) to perform `forward_backward` passes, `optim_step` to update weights, and `save_state` to save a checkpoint.
- [Losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/) are represented via `Datum` objects, which contain information on the input tokens and how to compute the losses over those tokens.

See the [Tinker docs](https://tinker-docs.thinkingmachines.ai/) for more information.

## Understand the core training loop

The `train` function iterates over a sequence of `rl_step` steps. After each steps, it prints out metrics from that step (like average reward and e-rate), and saves a checkpoint of the updated model. A link to the [Tinker Playground](https://tinker.thinkingmachines.ai/playground) is printed with each checkpoint so that you can chat with each stage of the model as it trains to see how it does.


Each `rl_step` follows the same sequence:

1. **Generate groups.** A `Rollout` represents a single response to a prompt, including the logprobs of each token in the response. Responses to the same prompt are grouped together in a `Group`.
2. **Score the answers.** Compute the reward for each answer. To start, the reward is a simple penalty for the fraction of letters that are e.
3. **Compute advantages.** Normalize rewards within each group to compute advantages for each rollout.
4. **Update the model.** `to_datum` converts to `Datum` objects, which encode the information needed to perform a forward/backward pass on Tinker's servers.

## Gibberish Answers

If you ran the above script with no modifications, you will see that while the model stops using the letter **e**, it very quickly degenerates into gibberish answers.


To mitigate the reward hacking, we can incorporate some measure of the quality of the model's response into the `reward` function. 

See the `Judge` protocol and the `BasicJudge` implementation for a pure text method of evaluating answer quality. Then update the `reward` function to actually use the judge's score.

Then just run again and see how it performs.

## Continued improvement
When running with the `BasicJudge`, you'll likely notice that the model does improve in that it's at least generating real words and not just repeating the same text over and over again. However, it's likely still not generating answers that are coherent and relevant to the question.

Can we find some way to evaluate the semantic quality of the text instead of just the syntax?

<details>
<summary>Hint:</summary>

Could another LLM judge whether an answer is coherent and relevant to the
question? Try `thinkingmachines/Inkling-Small` through Tinker. Think about what
information the judge needs and how its assessment should affect the reward.
The [Inkling rendering guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/tml-renderers/)
shows how to format and parse its messages.

If you get stuck, see [`solutions/lipogram_llm_judge.py`](./solutions/lipogram_llm_judge.py) for a reference implementation.

</details>

## Additional Challenges
There is a lot of room for improvement in the final training script. See if you can find a way to reduce the rate of **e** in the model's responses further while maintaining response quality. The following are some suggestions, but feel free to explore and experiment!


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

