from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256


@dataclass(slots=True)
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.92
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    enforce_eager: bool = False
    dtype: str = "auto"
    trust_remote_code: bool = False


@dataclass(slots=True)
class RunConfig:
    num_prompts: int
    trajectories_per_prompt: int
    output_dir: str = "./output"
    run_name: str | None = None


@dataclass(slots=True)
class PromptGeneratorConfig:
    model: str
    system_prompt: str
    user_initial_prompt: str
    sampling: SamplingConfig
    batch_size: int = 32
    vllm: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass(slots=True)
class TrajectoryGeneratorConfig:
    model: str
    system_prompt: str
    trajectory_prefix: str = ""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    batch_size: int = 64
    vllm: VLLMConfig = field(default_factory=VLLMConfig)


@dataclass(slots=True)
class AppConfig:
    run: RunConfig
    prompt_generator: PromptGeneratorConfig
    trajectory_generator: TrajectoryGeneratorConfig


def _require(data: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key '{ctx}.{key}' in config.json")
    return data[key]


def _sampling_from(data: dict[str, Any] | None) -> SamplingConfig:
    data = data or {}
    return SamplingConfig(
        temperature=float(data.get("temperature", 0.7)),
        top_p=float(data.get("top_p", 0.95)),
        max_tokens=int(data.get("max_tokens", 256)),
    )


def _vllm_from(data: dict[str, Any] | None) -> VLLMConfig:
    data = data or {}
    return VLLMConfig(
        tensor_parallel_size=int(data.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.92)),
        max_num_batched_tokens=(
            int(data["max_num_batched_tokens"])
            if data.get("max_num_batched_tokens") is not None
            else None
        ),
        max_num_seqs=(
            int(data["max_num_seqs"]) if data.get("max_num_seqs") is not None else None
        ),
        max_model_len=(
            int(data["max_model_len"]) if data.get("max_model_len") is not None else None
        ),
        enforce_eager=bool(data.get("enforce_eager", False)),
        dtype=str(data.get("dtype", "auto")),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    run_raw = _require(raw, "run", "root")
    prompt_raw = _require(raw, "prompt_generator", "root")
    traj_raw = _require(raw, "trajectory_generator", "root")

    run = RunConfig(
        num_prompts=int(_require(run_raw, "num_prompts", "run")),
        trajectories_per_prompt=int(
            _require(run_raw, "trajectories_per_prompt", "run")
        ),
        output_dir=str(run_raw.get("output_dir", "./output")),
        run_name=run_raw.get("run_name"),
    )

    prompt = PromptGeneratorConfig(
        model=str(_require(prompt_raw, "model", "prompt_generator")),
        system_prompt=str(_require(prompt_raw, "system_prompt", "prompt_generator")),
        user_initial_prompt=str(
            _require(prompt_raw, "user_initial_prompt", "prompt_generator")
        ),
        sampling=_sampling_from(prompt_raw.get("sampling")),
        batch_size=int(prompt_raw.get("batch_size", 32)),
        vllm=_vllm_from(prompt_raw.get("vllm")),
    )

    traj = TrajectoryGeneratorConfig(
        model=str(_require(traj_raw, "model", "trajectory_generator")),
        system_prompt=str(_require(traj_raw, "system_prompt", "trajectory_generator")),
        trajectory_prefix=str(traj_raw.get("trajectory_prefix", "")),
        sampling=_sampling_from(traj_raw.get("sampling")),
        batch_size=int(traj_raw.get("batch_size", 64)),
        vllm=_vllm_from(traj_raw.get("vllm")),
    )

    if run.num_prompts <= 0:
        raise ValueError("run.num_prompts must be > 0")
    if run.trajectories_per_prompt <= 0:
        raise ValueError("run.trajectories_per_prompt must be > 0")
    if prompt.batch_size <= 0 or traj.batch_size <= 0:
        raise ValueError("batch_size values must be > 0")

    return AppConfig(run=run, prompt_generator=prompt, trajectory_generator=traj)
