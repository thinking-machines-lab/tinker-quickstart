# Tinker Quickstart

Welcome to Tinker! This quickstart will get you started training a model using the Tinker SDK.

## Setup

### Account Setup

Set up a Tinker account in the [Tinker Console](https://tinker.thinkingmachines.ai/).

Make sure to set up [Billing](https://tinker.thinkingmachines.ai/billing/balance) so that you can begin training models. This quickstart uses very small models, so it chould cost you no more than $5. See the [pricing page](https://tinker-docs.thinkingmachines.ai/tinker/models/) for more details.

### Prerequisites

Make sure you have the `uv` package manager installed installed by running the following in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the repo directory, run

```bash
uv sync
```

### Authenticating

Run `uv run tinker auth login` from this repo to authenticate your user.

Now any code that you run will be automatically authenticated.

It is also possible to manually generate an API key through the [API Keys page](https://tinker.thinkingmachines.ai/keys). This API key can be set through the environment variable `TINKER_API_KEY`.

## Training the model

We'll be using Tinker to train a model that can't stop talking about your favorite creature. All of the code will be in [favorite_creature.py](./favorite_creature.py).

To start the training, just run:

```bash
uv run favorite_creature.py
```

This will run a reinforcement training loop using Tinker, and print out a link to chat with a trained checkpoint in [Tinker Playground](https://tinker.thinkingmachines.ai/playground). It will also output a training plot in `[training_run.png](./training_run.png)` with three graphs:

- Average reward
- Average mentions of "fairy"
- Average per token KL divergence

### Overview of the code

While the training loop runs, let's take a look at what's going on. The `train` function is the high level loop that is running.

It starts by constructing some of the key Tinker objects:

- `ServiceClient` - represents an active Tinker session. Each new ServiceClient creates a new session, which you can view in the [sessions page](https://tinker.thinkingmachines.ai/sessions).
- `TrainingClient` - a client for performing training operations, like forward/backward passes and optimization steps. Each training client creates a training run in the session. The training client uses Low-Rank Adaptation ([LoRA](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/)) to efficiently train models via the Tinker API.
- `SamplingClient` - a client that just samples from a given model. This can be created from a training client. This can be used to provide logprobs from a base model, or more commonly for generating rollouts in for RL tasks.
- `Tokenizer` - Converts from text to tokens and back.

The `train` method then runs through an RL loop, where the basic sequence is:

1. Give the model a prompt and generate multiple responses
2. Assign rewards to those responses
3. Update the model.

However, as we'll see below, the exact way that we provide the reward can have major implications for the model.

### Fixing Reward Hacking

You may have noticed after the first run that the model quickly devolves into nonsense, just repeating the word "fairy" over and over again. This shouldn't be surprising! In the `reward` function, we reward the model for saying "fairy", and so it optimizes for that reward.

Maybe we can do better. One simple improvement is to cap the reward the model gets from each mention of "fairy". Update the `MAX_MENTIONS_REWARDED` value to `4` so that the model doesn't get excessive reward.

However, this still wouldn't prevent the model from output "fairy" over and over again. We want some way to make sure that the model is still outputting reasonable content. An easy way to do that is to have another LLM act as a judge. Set `JUDGE_QUALITY_WEIGHT = 0.5`. This will run `Inkling-Small` as a judge using Tinker's [OpenAI compatible API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/) to evaluate the response produced by the model we're training. See the prompt used for this in the `judge_score` function.

### Avoiding excessive deviation

You may have noticed that even though using Inkling-Small as a judge helped constrain the model slightly, the model can still end up with odd behavior.

One way to avoid this is to try to keep the new model "close" to the old model by penalizing large differences. In `apply_kl_penalties`, we apply a per-token penalty based on the difference between the log probability of the sampled token in the trained model from the base model. To activate this penalty, set `KL_COEF = 0.05`. Increasing this weight will penalize the model more for deviating from the base model.

Running the training loop now should provide a model that still mentions fairies, but that is still coherent and reasonable.

### Keep exploring

Tinker makes it possible to quickly and cheaply explore many different training recipes. Some thing to try:

- [Easy] Modify the coefficients to see how changing the reward shapes model performance
- [Easy] Update the judge model prompt to steer the model towards different behaviors
- [Easy] Changing or adding multiple creatures to reward
- [Easy] Update the training prompts to get model to talk about fairies in different scenaros
- [Medium] Use [SFT](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/) to give the model some sample "good" answers before training
- [Medium] Apply token level rewards for words that we want to encourage, rather than just sequence level rewards
- [Hard] Instead of a well-known creature, invent a new creature and teach the model about this creature first
