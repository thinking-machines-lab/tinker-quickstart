"""Solution: replace the basic quality checks with an Inkling-Small judge.

Run from the repository root: uv run -m solutions.lipogram_with_judge
The training loop and e penalty are shared with the tutorial.
"""

import asyncio
import re

import tinker
from tinker import types
from tml_renderers import chat, tokenizers, v0
from tml_renderers import tinker as tml_tinker

from lipogram import train

JUDGE_MODEL = "thinkingmachines/Inkling-Small"


async def main() -> None:
    service = tinker.ServiceClient()
    sampling_client = await service.create_sampling_client_async(base_model=JUDGE_MODEL)
    await train(judge=LlmJudge(sampling_client), description="With LLM Judge")


class LlmJudge:
    """Ask a frozen model to score meaning and relevance on a single 1–5 scale."""

    def __init__(self, sampling_client: tinker.SamplingClient) -> None:
        self.sampling_client = sampling_client
        self.renderer = v0.Renderer(tokenizers.o200k_base_chat())

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
        # Inkling uses the TML format; effort 0 disables reasoning for this short grade.
        spans, parser = self.renderer.render_for_completion_with_effort(messages, 0.0)
        result: types.SampleResponse = await self.sampling_client.sample_async(
            prompt=tml_tinker.token_spans_to_tinker_model_input(spans),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=32, temperature=0.0, stop=self.renderer.stop()
            ),
        )
        replies: list[chat.Message] = parser.parse_tokens(result.sequences[0].tokens)
        text = "".join(
            message.content.text
            for message in replies
            if isinstance(message.content, chat.Text)
        )
        number = re.fullmatch(r"[1-5]", text.strip())
        # Missing, incomplete, or malformed grades receive the lowest score.
        return float(number.group()) if number else 1.0


GRADE_INSTRUCTIONS = """Grade the following response on a scale of 1 to 5. Scoring guidelines:
If the response is incoherent, contains any non-English words, or repeats punctuation marks over and over, return 1.
If the response is repetitive or the same phrase over and over, return 2.
If the response is in English, coherent but not a response to the prompt, return 3.
If the response is in English, on topic and clear, return 4.
If the response is in English, high quality, on-topic, and well-written, return 5.

This is a snippet of the response and not the entire response. Grade only on the response so far. Do not decrease score for cut off responses.

Return one number from 1 to 5 and nothing else."""


if __name__ == "__main__":
    asyncio.run(main())
