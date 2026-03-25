from __future__ import annotations

from typing import Iterable


def shard_slices(total_items: int, num_shards: int) -> Iterable[tuple[int, int]]:
    """
    Split [0, total_items) into `num_shards` contiguous slices with remainder distributed
    across the first `total_items % num_shards` shards.

    This avoids losing the remainder when total_items is not divisible by num_shards.
    """
    if total_items <= 0:
        return []
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")

    base = total_items // num_shards
    rem = total_items % num_shards

    out: list[tuple[int, int]] = []
    start = 0
    for i in range(num_shards):
        extra = 1 if i < rem else 0
        end = start + base + extra
        out.append((start, end))
        start = end
    return out

