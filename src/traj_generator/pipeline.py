from __future__ import annotations

import gc
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from vllm import LLM, SamplingParams

from .config import AppConfig, SamplingConfig, VLLMConfig


@dataclass(slots=True)
class GeneratedPrompt:
    prompt_id: int
    text: str


@dataclass(slots=True)
class GeneratedTrajectory:
    prompt_id: int
    trajectory_id: int
    prompt_text: str
    trajectory_text: str


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _sampling_params(cfg: SamplingConfig) -> SamplingParams:
    return SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )


def _build_llm(model_name: str, cfg: VLLMConfig) -> LLM:
    kwargs = {
        "model": model_name,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "enforce_eager": cfg.enforce_eager,
        "dtype": cfg.dtype,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
    if cfg.max_num_seqs is not None:
        kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.max_model_len is not None:
        kwargs["max_model_len"] = cfg.max_model_len
    return LLM(**kwargs)


def _release_gpu_memory(llm: LLM | None) -> None:
    if llm is not None:
        del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _extract_text(outputs) -> str:
    if not outputs.outputs:
        return ""
    return outputs.outputs[0].text.strip()


def _build_prompt_generator_input(system_prompt: str, user_initial_prompt: str, index: int) -> str:
    return (
        f"{system_prompt}\n\n"
        f"Initial user request: {user_initial_prompt}\n\n"
        "Create exactly one standalone prompt for another LLM. "
        "Return only the prompt text without extra explanations.\n"
        f"Variation index: {index}"
    )


def _build_trajectory_input(system_prompt: str, trajectory_prefix: str, prompt_text: str) -> str:
    prefix = trajectory_prefix.strip()
    if prefix:
        return f"{system_prompt}\n\n{prefix}\n\n{prompt_text}"
    return f"{system_prompt}\n\n{prompt_text}"


def _prepare_output_dirs(base_output_dir: Path, run_name: str) -> tuple[Path, Path]:
    run_dir = base_output_dir / run_name
    archive_dir = base_output_dir / "archives"
    run_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, archive_dir


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _zip_run_directory(run_dir: Path, archive_dir: Path) -> Path:
    zip_path = archive_dir / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(run_dir.rglob("*")):
            if item.is_file():
                zf.write(item, arcname=item.relative_to(run_dir.parent))
    return zip_path


def run_pipeline(config: AppConfig) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = config.run.run_name or f"run_{timestamp}"
    base_output_dir = Path(config.run.output_dir).resolve()
    run_dir, archive_dir = _prepare_output_dirs(base_output_dir, run_name)

    prompt_inputs = [
        _build_prompt_generator_input(
            config.prompt_generator.system_prompt,
            config.prompt_generator.user_initial_prompt,
            idx + 1,
        )
        for idx in range(config.run.num_prompts)
    ]
    prompt_sampling = _sampling_params(config.prompt_generator.sampling)
    prompts: list[GeneratedPrompt] = []
    prompt_llm: LLM | None = None
    try:
        prompt_llm = _build_llm(config.prompt_generator.model, config.prompt_generator.vllm)
        for chunk in _chunked(prompt_inputs, config.prompt_generator.batch_size):
            outputs = prompt_llm.generate(chunk, prompt_sampling, use_tqdm=False)
            base_prompt_id = len(prompts)
            prompts.extend(
                GeneratedPrompt(prompt_id=base_prompt_id + idx, text=_extract_text(out))
                for idx, out in enumerate(outputs)
            )
    finally:
        _release_gpu_memory(prompt_llm)

    trajectory_inputs: list[str] = []
    trajectory_index: list[tuple[int, int]] = []
    for prompt in prompts:
        for traj_idx in range(config.run.trajectories_per_prompt):
            trajectory_inputs.append(
                _build_trajectory_input(
                    config.trajectory_generator.system_prompt,
                    config.trajectory_generator.trajectory_prefix,
                    prompt.text,
                )
            )
            trajectory_index.append((prompt.prompt_id, traj_idx))

    trajectory_sampling = _sampling_params(config.trajectory_generator.sampling)
    trajectories: list[GeneratedTrajectory] = []
    processed = 0
    traj_llm: LLM | None = None
    try:
        traj_llm = _build_llm(
            config.trajectory_generator.model, config.trajectory_generator.vllm
        )
        for chunk in _chunked(trajectory_inputs, config.trajectory_generator.batch_size):
            outputs = traj_llm.generate(chunk, trajectory_sampling, use_tqdm=False)
            for local_idx, output in enumerate(outputs):
                prompt_id, traj_id = trajectory_index[processed + local_idx]
                prompt_text = prompts[prompt_id].text
                trajectories.append(
                    GeneratedTrajectory(
                        prompt_id=prompt_id,
                        trajectory_id=traj_id,
                        prompt_text=prompt_text,
                        trajectory_text=_extract_text(output),
                    )
                )
            processed += len(chunk)
    finally:
        _release_gpu_memory(traj_llm)

    prompts_path = run_dir / "generated_prompts.jsonl"
    trajectories_path = run_dir / "generated_trajectories.jsonl"
    manifest_path = run_dir / "manifest.json"

    _write_jsonl(prompts_path, (asdict(p) for p in prompts))
    _write_jsonl(trajectories_path, (asdict(t) for t in trajectories))

    manifest = {
        "run_name": run_name,
        "created_utc": timestamp,
        "config": asdict(config),
        "num_prompts": len(prompts),
        "num_trajectories": len(trajectories),
        "files": {
            "prompts": str(prompts_path),
            "trajectories": str(trajectories_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = _zip_run_directory(run_dir, archive_dir)
    return {
        "run_dir": str(run_dir),
        "zip_path": str(zip_path),
        "prompts_path": str(prompts_path),
        "trajectories_path": str(trajectories_path),
        "manifest_path": str(manifest_path),
    }
