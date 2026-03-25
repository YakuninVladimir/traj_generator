from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import GeneratedTrajectoriesDatasetConfig


class GeneratedTrajectoriesDataset(Dataset):
    """
    Reads `trajectories_*.jsonl` written by `scripts/generator.py`.

    Each line is expected to be JSON with at least:
      - `trajectory`: the raw LLM text

    Missing/uneven shards are supported: we scan the directory and only use files that exist.
    """

    def __init__(
        self,
        trajectories_dir: str | Path,
        trajectories_glob: str = "trajectories_*.jsonl",
        max_samples: int | None = None,
    ) -> None:
        self.trajectories_dir = Path(trajectories_dir)
        self.trajectories_glob = trajectories_glob
        self.max_samples = max_samples

        paths = sorted(self.trajectories_dir.glob(self.trajectories_glob))
        if not paths:
            raise FileNotFoundError(
                f"No trajectories files found in {self.trajectories_dir} with glob={self.trajectories_glob}"
            )

        self.rows: list[dict[str, Any]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if self.max_samples is not None and len(self.rows) >= self.max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise FileNotFoundError(
                f"Found trajectories files, but no rows were read from {paths[0].parent}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "trajectory": row["trajectory"],
            "prompt_id": row.get("prompt_id"),
            "trajectory_id": row.get("trajectory_id"),
        }


def _build_collate_fn(
    tokenizer,
    embedder,
    device: str,
    seq_len_obs: int,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """
    Converts a batch of raw `trajectory` strings into `obs_seq`
    shaped as (B, seq_len_obs, dim_y) where dim_y == embedder hidden size.
    """

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [b["trajectory"] for b in batch]
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len_obs,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = embedder(**tokenized, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]

            # Mask padding tokens to zeros.
            attn = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            hidden = hidden * attn

        return {
            "obs_seq": hidden,  # (B, T, dim_y)
            "prompt_id": [b["prompt_id"] for b in batch],
            "trajectory_id": [b["trajectory_id"] for b in batch],
        }

    return collate_fn


def make_generated_trajectories_dataloader(
    cfg: GeneratedTrajectoriesDatasetConfig,
    *,
    shuffle: bool = False,
    max_samples: int | None = None,
) -> DataLoader:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GeneratedTrajectoriesDataset(
        trajectories_dir=cfg.trajectories_dir,
        trajectories_glob=cfg.trajectories_glob,
        max_samples=max_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    embedder = AutoModelForCausalLM.from_pretrained(cfg.embedder_model_name).to(device)
    embedder.eval()

    collate_fn = _build_collate_fn(
        tokenizer=tokenizer,
        embedder=embedder,
        device=device,
        seq_len_obs=cfg.seq_len_obs,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn,
    )

