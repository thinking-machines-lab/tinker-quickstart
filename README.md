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
Run the following command and follow instructions for generating an API key. This will save the API key and load it from the SDK for all future use.
```
uv run tinker auth login
```
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

`main` loops `TRAINING_STEPS` times. After each step it prints metrics from that step (mean reward and mean e-rate) with an example answer, and saves a checkpoint of the updated model. A link to the [Tinker Playground](https://tinker.thinkingmachines.ai/playground) is printed with each checkpoint so that you can chat with each stage of the model as it trains to see how it does.


Each step follows the same sequence:

1. **Generate answers.** Sample `GROUP_SIZE` answers for each prompt from the latest checkpoint. A "group" is a set of responses for the same prompt, so there is one group per prompt in `TRAINING_PROMPTS`. 
2. **Score the answers.** Use the `reward` function to compute a reward for each answer, where the model will be nudged towards higher rewards.
3. **Compute advantages.** Normalize rewards within each group, so an answer is rewarded for beating the others in its group.
4. **Update the model.** Each answer becomes a `Datum` carrying its tokens, logprobs, and advantage: the information Tinker's servers need for a forward/backward pass. 

## Gibberish Answers

If you ran the above script with no modifications, you will see that while the model stops using the letter **e**, it very quickly degenerates into gibberish answers.

To see why, take a look at the `reward` function. The reward is determined by how many times the model uses the letter **e** in the response, but with no regard for quality. Try uncommenting the `judge` grade in the reward and see how the model's performance improves. 

To see how this works, take a look at the implementation of `LlmJudge`. It uses the `Inkling-Small` model to grade the quality of each response. It turns out that even if other LLMs aren't good at writing lipograms, they are fairly good at grading the quality of a response!

# Keep iterating!
There is a lot of room for improvement in the final training script. See if you can find a way to reduce the rate of **e** in the model's responses further while maintaining response quality. The following are some suggestions to try.

### Easy
- **Modify the `JUDGE_WEIGHT`**: higher to emphasize quality more, lower to emphasize e-rate more.
- **Update reward aggregation**: Change way that `reward` combines the judge grade and the e-rate penalty. Right now it's just a direct sum, but maybe there's some other combination that makes more sense!
- **Change `GRADE_INSTRUCTIONS`**: Update the instructions for the judge to be more specific about what makes a good response.
- **Add more prompts to `TRAINING_PROMPTS`**: The model gets to try more prompts on each turn, it might generalize better this way!

### Medium
- **Token level penalties**: Right now the model gets a sequence level penalty, so it doesn't know which specific tokens are bad. But we do! The `advantages` field in the `loss_fn_inputs` allows per token advantages. Can you figure out which tokens contain an **e** and reduce the advantage for those tokens?
- **Teach by example**: Try generating some lipograms yourself and use [supervised fine-tuning (SFT)](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/) to train on those examples first so the model has a good starting point.
- **Rubric judge**: Our current judge just returns a single overall score. Try defining separate criteria and have the judge evaluate each criteria separately, then combine the scores into a single overall score that you think makes sense.


### Hard
- **KL penalty**: To avoid the model overfitting, try adding a KL penalty to the advantage that penalizes based on the logprob of the sampled token compared to the logprob of the token in an untrained model. Create a `SamplingClient` and use `compute_logprobs` to get the logprobs from the untrained model, and use that to nudge the model towards the base model's logprobs.
- **Context distillation**: Teach the model by example, but instead of coming up with examples yourself, use a larger model to generate answers conditioned on a prefix that explicitly mentions avoiding the letter **e**.

