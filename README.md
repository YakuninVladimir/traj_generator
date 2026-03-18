# traj-generator

Config-driven Python `uv` project that generates:
1. `N` prompts using a first LLM.
2. `M` trajectories per prompt using a second LLM.

Inference uses **vLLM** with batching for high throughput.

## What you can configure

Everything is configured in `config.json`:
- `run.num_prompts` (`N`)
- `run.trajectories_per_prompt` (`M`)
- `prompt_generator.user_initial_prompt` (your initial instruction for the first model)
- model names, sampling settings, vLLM engine settings, and batch sizes
- output directory (`run.output_dir`)
- vLLM memory controls: `max_model_len`, `max_num_seqs`, `gpu_memory_utilization`
- optional eager mode: `enforce_eager` (set `true` to avoid CUDA graph capture)

## Output format

For each run, files are written to:
- `output/run_<timestamp>/generated_prompts.jsonl`
- `output/run_<timestamp>/generated_trajectories.jsonl`
- `output/run_<timestamp>/manifest.json`

Each run directory is also archived to:
- `output/archives/run_<timestamp>.zip`

## Local run (without Docker)

Requirements:
- Python 3.10+
- NVIDIA GPU + CUDA drivers (recommended for vLLM)

Commands:

```bash
uv sync
uv run traj-generate --config config.json
```

## Docker run

Requirements:
- Docker + Docker Compose
- NVIDIA Container Toolkit (for GPU access in containers)

Run:

```bash
docker compose up --build
```

Because `./output` is bind-mounted into `/app/output`, generated files and ZIP archives remain on your host machine after container shutdown.

Stop and remove container:

```bash
docker compose down
```

## Config notes for speed

To maximize throughput:
- Increase `prompt_generator.batch_size` and `trajectory_generator.batch_size` as far as GPU memory allows.
- Tune vLLM settings:
  - `max_num_batched_tokens`
  - `max_num_seqs`
  - `max_model_len`
  - `gpu_memory_utilization`
  - `enforce_eager` (`false` is usually faster; `true` is safer for tight VRAM)
  - `tensor_parallel_size` (for multi-GPU)
- Use a model size that fits your hardware.

## Memory behavior

The pipeline loads models sequentially:
1. prompt generator model runs and is released,
2. trajectory model starts only after GPU memory cleanup.

This avoids keeping two vLLM engines in VRAM at the same time on single-GPU servers.
