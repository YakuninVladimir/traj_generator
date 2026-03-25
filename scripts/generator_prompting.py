from __future__ import annotations

from typing import List

from tqdm import tqdm

from generator_llm import generate_texts
from configs import PromptSamplingConfig


def generate_prompts(
    topic: str,
    n: int,
    tokenizer,
    model,
    *,
    batch_size: int,
    sampling: PromptSamplingConfig,
) -> List[str]:
    prompts_out: list[str] = []

    base_prompts = [
        f"Generate ONE diverse prompt about: {topic}\nPrompt:"
        for _ in range(n)
    ]

    for i in tqdm(range(0, n, batch_size)):
        batch = base_prompts[i : i + batch_size]
        outputs = generate_texts(
            tokenizer,
            model,
            batch,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
        )
        for text in outputs:
            text = text.strip()
            if text.startswith(("1.", "-", "*")):
                text = text[2:].strip()
            if text:
                prompts_out.append(text)

    if len(prompts_out) < n:
        prompts_out += [
            f"Write about {topic} #{i}"
            for i in range(len(prompts_out), n)
        ]

    return prompts_out[:n]

