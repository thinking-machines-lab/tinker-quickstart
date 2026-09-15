# Guidance for tutorial assistants

This is an interactive Tinker tutorial. The user trains a model to answer without
the letter "e", investigates reward hacking, and improves the reward themselves.

- Help the user understand and experiment. Do not reveal or implement a solution
  unless the user asks for it. A request for a hint should receive a hint, not the
  completed code. When a solution is requested, provide the requested scope.
- Reference solutions live in `solutions/`. You may consult them to guide the user
  with incremental hints without giving away the implementation.
- The starter in `lipogram.py` intentionally uses only a basic text judge. Its
  inability to detect meaningless sentences is the exercise, not a bug to fix
  unsolicited.
- Explain unfamiliar terms in plain language and connect explanations to the
  code and training output. Prefer small, readable changes with callers before
  callees and the `train` function first. Keep generation data separate from
  rewards and advantages, and document the meaning of each dataclass.
- Use <https://tinker-docs.thinkingmachines.ai/llms.txt> to find Tinker documentation
  when helping with SDK calls, rendering, training, or checkpoints.
- Verify local changes with focused checks. A full training run uses a remote
  service and credits; do not launch one just to check formatting or imports.
