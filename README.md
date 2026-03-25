# traj-generator

Generate LLM-based trajectories in two stages:
1. prompt generator produces `N` prompts
2. trajectory generator produces `M` trajectories per prompt

This repo also contains the `DeepParticleFilter` implementation extracted from `filter.ipynb`, plus:
- a dataset/dataloader that maps generated trajectories text into `obs_seq` for the filter
- smoke/physical/generated tests for the full pipeline
- `scripts/train_filter_rebuilt.py`: stage-based filter training + visualization (TensorBoard + online Matplotlib/Seaborn plots)

## Output format

Generation writes to `--output-dir` (default: `outputs`):
- `prompts.jsonl`: each line has `{ "prompt_id": int, "prompt": str }`
- `trajectories_<shard>.jsonl`: each line has:
  - `{ "prompt_id": int, "trajectory_id": int, "prompt": str, "trajectory": str }`

Missing/uneven shards are supported: the dataset scans existing `trajectories_*.jsonl` files.

## Generated trajectories -> `obs_seq` mapping

`traj_generator.data.generated_trajectories.make_generated_trajectories_dataloader`:
1. tokenizes `trajectory` with `--tokenizer-model`
2. truncates/pads to `--seq-len-obs`
3. runs the embedder (by project convention, it uses the same model as `--tokenizer-model`) and uses the last hidden layer as `obs_seq` with shape `(B, seq_len_obs, dim_y)`
4. padding tokens are masked out (zeroed)

`DeepParticleFilter` assumes `dim_x >= dim_y` (it uses `particles[..., :dim_y]`). The generated-trajectories test sets `dim_x = dim_y`.

## 4 Commands

1. Smoke tests (filter forward/backward on synthetic physics):
```bash
uv run python tests/smoke_pipeline.py
```

2. Tests on physical data (parameters taken from `filter.ipynb`):
```bash
uv run python tests/physical_data_tests.py
```

3. Generate trajectories (adjust counts + models for speed):
```bash
uv run python scripts/generator_cli.py \
  --output-dir outputs \
  --num-prompts 8 \
  --trajectories-per-prompt 4 \
  --num-shards 2
```

You can override models/sampling with CLI flags:
`--prompt-model`, `--trajectory-model`, `--prompt-temperature`, `--trajectory-temperature`, etc.

4. Tests on generated trajectories:
```bash
uv run python tests/generated_trajectories_tests.py \
  --trajectories-dir outputs \
  --tokenizer-model sshleifer/tiny-gpt2 \
  --seq-len-obs 16 \
  --batch-size 2
```

## Training Filter (Rebuilt Notebook)

Train `DeepParticleFilter` with the stage schedule from `filter_rebuilt.ipynb` and log:
- TensorBoard scalars (if `tensorboard` is available)
- online plots saved on disk after every batch (no GUI required)

The script supports two dataset sources:
- `--dataset physical` (default): synthetic damped-oscillator sequences generated on the fly.
- `--dataset generated`: `obs_seq` is built by embedding LLM trajectories from `trajectories_*.jsonl`.

For `--dataset generated`, `dim_y` is inferred from the embedder hidden size; by default `--dim-x` is set to the same value. If needed you can override with `--dim-x`.

To control epochs per stage, edit the stage schedule function:
`src/filter/train_filter_rebuilt_config.py` (`build_stage_specs()`).

Example (quick CPU run on `physical`):
```bash
uv run python scripts/train_filter_rebuilt.py \
  --logdir output/filter_tb \
  --batch-size 16 \
  --seq-len-obs 100 \
  --n-pred 5 \
  --dt 0.05 \
  --no-cuda
```

Example (quick CPU run on `generated` trajectories):
```bash
uv run python scripts/train_filter_rebuilt.py \
  --logdir output/filter_tb_generated \
  --dataset generated \
  --trajectories-dir outputs \
  --tokenizer-model sshleifer/tiny-gpt2 \
  --seq-len-obs 32 \
  --n-pred 1 \
  --dt 0.05 \
  --batch-size 2 \
  --generated-max-samples 32 \
  --no-cuda
```
