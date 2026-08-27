# Tinker Quickstart

Welcome to Tinker! This quickstart will get you started training a model using the Tinker SDK.

## Setup

### Account Setup

Sign into your TML account in the [Tinker Console](https://tinker.thinkingmachines.ai/).

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

## Model Training Overview

We'll be using Tinker to train a model that can't stop talking about your favorite creature. All of the code will be in [favorite_creature.py](./favorite_creature.py).

To start the training, just run:

```bash
uv run favorite_creature.py
```

This will run a reinforcement training loop using Tinker, and print out a link to chat with a trained checkpoint in [Tinker Playground](https://tinker.thinkingmachines.ai/playground). It will also output a training plot in [`training_run.png`](./training_run.png) with three graphs:

- Average reward
- Average mentions of "fairy"
- Average per token KL divergence

### Key Tinker Concepts
#### Clients
[`ServiceClient`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/) - Represents an active session which may consist of multiple training runs. See [sessions page](https://tinker.thinkingmachines.ai/sessions) for all Tinker sessions.


[`TrainingClient`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)  - Perform forward/backward and optimization steps in Tinker. Create via `ServiceClient.create_lora_training_client()`.

[`SamplingClient`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/) - Sample tokens from Tinker. 

#### Basic Sampling
See `sample_one_response` function for example.

- Create a `PreTrainedTokenizer` via `SamplingClient.get_tokenizer()` or `TrainingClient.get_tokenizer()`. This converts from text to/from tokens.

- [Optional] Create a [Renderer](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/) for easier control over which tokens are trained on. We only use standard tokenizers in this exercise.

- Tokenize prompt/conversation using `PreTrainedTokenizer`, then convert to `ModelInput` via `ModelInput.from_ints`.

- Call `ServiceClient.sample` with model input, number of sequences to generate, and sampling params, get back a `SampleResponse`. 

- Read individual sequences via `SampleResponse.sequences[i]`.
    - `SampledSequence.tokens` -> list of token ids
    - `SampledSequence.logprobs` -> list of logprobs of sampled tokens

#### Basic Training
See end of `rl_step` function for example.
- Construct [`Datum`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/datum/) objects. These represent a model input, and the loss function to apply.
    - `loss_fn_inputs` how to compute loss for the provided model input. See [Loss Functions](https://tinker-docs.thinkingmachines.ai/tinker/losses/) for supported loss functions.
- `TrainingClient.forward_backward` to perform forward and backward pass. Do _not_ call `.result()` on the future or `await` the result immediately.
- `TrainingClient.optim_step` perform optimization step.
- `await` or `.result()` on futures from `forward_backward` and `optim_step` once both have been invoked. See [Clock Cycles and Pipelining](https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/) for rationale.

### Fixing Reward Hacking
You may have noticed that your model is reward hacking by just outputting "fairy" over and over again. Your first challenge is to fix this!

See the `reward` function for how the reward is computed and the `rl_step` function for how we compute advantages. Try tinkering with the hyperparameters defined at the top of the file to avoid reward hacking.

<details>
<summary>See recommended adjustments</summary>
In our experimentation, the following values have worked well:

- <code>MAX_MENTIONS_REWARDED = 4</code> - limits the amount of reward model can get from mentioning "fairy"
- <code>JUDGE_QUALITY_WEIGHT = 0.5</code> - use LLM as judge to keep response quality high. See <code>judge_score</code> function.
- <code>KL_COEF = 0.05</code> - penalize deviations from base model to keep reasonable behavior. See <code>apply_kl_penalties</code>.

</details>

## Additional Challenges
### Build a rollout viewer
Right now we discard each sequence. Try building a rollout viewer to visualize the sampled sequences. For an additional challenge, try visualizing the log probabilities of each token.

### Make up a creature
Instead of picking a well known creature like "fairy", invent your own creature that the model would never generate on its own. You'll likely have to "teach" the model about this new creature first because it will never generate this in the RL rollouts.

### Estimate model perplexity
Estimate and visualize the model's perplexity for each rollout. Hint: [`topk_prompt_logprobs`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/#sample) may be useful here.

