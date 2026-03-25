from __future__ import annotations

from itertools import chain
from typing import Any


def build_stage_specs(n_repeats: int) -> list[dict[str, Any]]:
    """
    Stage schedule copied from `filter_rebuilt.ipynb`.

    Edit the per-stage `epochs` / `lr` values here to control training length
    for each stage independently.
    """

    actor_critic_stages = [
        (
            {"name": f"transition_pretrain_{i}", "epochs": 5, "lr": 2e-3},
            {"name": f"likelihood_pretrain_{i}", "epochs": 5, "lr": 8e-4},
        )
        for i in range(n_repeats)
    ]

    return (
        [{"name": "init_pretrain", "epochs": 6, "lr": 2e-3}]
        + list(chain(*actor_critic_stages))
        + [
            {"name": "e2e_warmup", "epochs": 20, "lr": 8e-4},
            {"name": "e2e_finetune", "epochs": 10, "lr": 5e-4},
        ]
    )

