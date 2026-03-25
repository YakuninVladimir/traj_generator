from __future__ import annotations

from typing import Any, List, Tuple

from tqdm import tqdm

from generator_llm import generate_texts
from configs import TrajectorySamplingConfig


def generate_trajectories(
    prompts: list[str],
    m: int,
    tokenizer,
    model,
    *,
    batch_size: int,
    sampling: TrajectorySamplingConfig,
    prompt_id_offset: int = 0,
) -> list[dict[str, Any]]:
    all_data: list[dict[str, Any]] = []

    expanded: list[Tuple[int, int, str]] = []
    for local_prompt_id, prompt in enumerate(prompts):
        global_prompt_id = prompt_id_offset + local_prompt_id
        for traj_id in range(m):
            expanded.append((global_prompt_id, traj_id, prompt))

    for i in tqdm(range(0, len(expanded), batch_size)):
        batch = expanded[i : i + batch_size]
        batch_prompts = [item[2] for item in batch]

        outputs = generate_texts(
            tokenizer,
            model,
            batch_prompts,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
        )

        for (prompt_id, traj_id, prompt), text in zip(batch, outputs):
            all_data.append(
                {
                    "prompt_id": prompt_id,
                    "trajectory_id": traj_id,
                    "prompt": prompt,
                    "trajectory": text,
                }
            )

    return all_data

