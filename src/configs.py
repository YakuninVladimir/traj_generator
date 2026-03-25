from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256


@dataclass(slots=True)
class PromptSamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 64


@dataclass(slots=True)
class TrajectorySamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256


@dataclass(slots=True)
class GeneratorConfig:
    topic: str = "Interview tasks for c++ intern programmers"
    num_prompts: int = 10_000
    trajectories_per_prompt: int = 128
    num_shards: int = 100

    prompt_model_name: str = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
    trajectory_model_name: str = "tiiuae/Falcon-H1-Tiny-90M-Instruct"

    prompt_initial_batch_size: int = 64
    trajectory_batch_size: int = 64

    prompt_sampling: PromptSamplingConfig = field(default_factory=PromptSamplingConfig)
    trajectory_sampling: TrajectorySamplingConfig = field(default_factory=TrajectorySamplingConfig)

    output_dir: Path = Path("outputs")
    seed: int = 7


@dataclass(slots=True)
class GeneratedTrajectoriesDatasetConfig:
    trajectories_dir: Path
    trajectories_glob: str = "trajectories_*.jsonl"

    tokenizer_model_name: str = "sshleifer/tiny-gpt2"
    embedder_model_name: str = "sshleifer/tiny-gpt2"

    seq_len_obs: int = 32
    batch_size: int = 4
    device: str | None = None


@dataclass(slots=True)
class PhysicalDatasetConfig:
    n_sequences: int = 32
    seq_len_obs: int = 20
    n_pred: int = 2
    dt: float = 0.05

    spring_k: float = 0.7
    damping_c: float = 0.12
    process_std: tuple[float, float] = (0.02, 0.03)
    obs_std: float = 0.08
    device: str = "cpu"


@dataclass(slots=True)
class DeepParticleFilterConfig:
    dim_x: int = 2
    dim_y: int = 1

    n_particles: int = 256
    n_pred: int = 5
    dt: float = 0.05

    hidden_size: int = 96
    obs_embed_dim: int = 32
    init_noise_dim: int = 8
    transition_noise_dim: int = 8

    likelihood_mode: str = "hybrid"
    likelihood_match_dim: int = 64
    likelihood_num_heads: int = 4
    ess_threshold: float = 0.5
    contrastive_temperature: float = 0.10

