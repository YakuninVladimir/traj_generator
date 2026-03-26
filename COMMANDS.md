# COMMANDS

## Train Filter (Rebuilt Notebook)

Command:
```bash
uv run python scripts/train_filter_rebuilt.py
```

Main flags:
- `--logdir`: output directory for TensorBoard logs and online plots (default: `output/tblogs`).
- `--dataset`: which dataset to learn from: `physical` or `generated` (default: `physical`).

### Dataset: `physical` (synthetic damped oscillator)
- `--n-sequences`: number of sequences to generate (default: `256`).
- `--spring-k`: spring constant of the oscillator (default: `0.7`).
- `--damping-c`: damping coefficient of the oscillator (default: `0.12`).
- `--process-std-1`: process noise std (first component) (default: `0.02`).
- `--process-std-2`: process noise std (second component) (default: `0.03`).
- `--obs-std`: observation noise std (default: `0.08`).
- `--physical-device`: device for physical data generation (default: `cpu`).

### Dataset: `generated` (LLM trajectories -> obs_seq)
- `--trajectories-dir`: directory with `trajectories_*.jsonl` files (default: `outputs`).
- `--tokenizer-model`: tokenizer+embedder model name (tokenizer and embedder are the same by convention) (default: `sshleifer/tiny-gpt2`).
- `--generated-max-samples`: optional cap on number of JSONL rows to read (default: `None`).
- `--generated-device`: device to run embedder on (default: `cpu`).

### Common training/model flags
- `--batch-size`: batch size (default: `16`).
- `--dim-x`: particle state dimension (must be >= inferred `dim_y`). If omitted, uses inferred `dim_y`.
- `--seq-len-obs`: observation sequence length `T` used by the filter (default: `100`).
- `--n-pred`: number of latent prediction steps between observations (default: `5`).
- `--dt`: dynamics time step used by the transition model (default: `0.05`).

Model architecture:
- `--n-particles`: number of particles `N` (default: `256`).
- `--hidden-size`: internal MLP hidden size (default: `96`).
- `--obs-embed-dim`: observation encoder/embedding dim (default: `32`).
- `--init-noise-dim`: initial sampler latent noise dim (default: `8`).
- `--transition-noise-dim`: transition stochastic noise dim (default: `8`).
- `--likelihood-mode`: likelihood head mode (default: `hybrid`).
- `--contrastive-temperature`: temperature for NCE / contrastive loss (default: `0.10`).
- `--ess-threshold`: resampling threshold as fraction of `N` (default: `0.5`).

Training schedule:
- `--n-repeats`: how many repeats of the `transition_pretrain_i` + `likelihood_pretrain_i` blocks (default: `4`).
- `--window-len`: window length used only for `e2e_warmup`/`e2e_finetune` random windowing (default: `20`).
- `--epochs-scale`: multiply stage epochs by this scale (default: `1.0`).
- stage schedule (epochs/lr per stage) is configured in `src/filter/train_filter_rebuilt_config.py` via `build_stage_specs()`.
- `--lr`: override learning rate for all stages (default: `None`, use notebook values per stage).
- `--checkpoints-dir`: directory for saving the single best checkpoint file `best_filter.pt` (default: `checkpoints`).
- `--seed`: random seed (default: `7`).
- `--no-cuda`: force CPU even if CUDA is available.

Example (generated):
```bash
uv run python scripts/train_filter_rebuilt.py \
  --logdir output/filter_tb_generated \
  --dataset generated \
  --trajectories-dir outputs \
  --tokenizer-model tiiuae/Falcon-H1-Tiny-90M-Instruct \
  --checkpoints-dir checkpoints \
  --seq-len-obs 32 \
  --n-pred 1 \
  --dt 0.05 \
  --batch-size 2 \
  --generated-max-samples 32 \
  --no-cuda
```

## Smoke Tests (filter + synthetic physics)
What it does: generates a tiny synthetic damped-oscillator dataset, runs one forward/backward + optimizer step using `DeepParticleFilter.initializer_pretrain_loss`.

```bash
uv run python tests/smoke_pipeline.py
```

## Physical Data Tests (filter + synthetic physics; notebook params)
What it does: generates a small synthetic damped-oscillator dataset using the parameter set taken from the original notebook, then runs one forward/backward + optimizer step using `DeepParticleFilter.initializer_pretrain_loss`.

```bash
uv run python tests/physical_data_tests.py
```

## Generate Trajectories (LLM prompt -> LLM trajectory JSONL)
What it does: creates `prompts.jsonl` and shard files `trajectories_<shard>.jsonl` in `--output-dir` by running `scripts/generator_cli.py`.

```bash
uv run python scripts/generator_cli.py \
  --output-dir outputs \
  --num-prompts 8 \
  --trajectories-per-prompt 4 \
  --num-shards 2
```

Optional overrides (useful for speed / memory):
- model flags: `--prompt-model`, `--trajectory-model`
- sampling flags: `--prompt-temperature`, `--prompt-top-p`, `--prompt-max-new-tokens`
- sampling flags: `--trajectory-temperature`, `--trajectory-top-p`, `--trajectory-max-new-tokens`
- batching flags: `--prompt-batch-size`, `--trajectory-batch-size`

## Tests on Generated Trajectories (text -> obs_seq -> filter step)
What it does: loads `trajectories_*.jsonl` from `--trajectories-dir`, embeds trajectory text into `obs_seq` via tokenizer/embedder, sets `dim_x=dim_y`, then runs one forward/backward + optimizer step using `DeepParticleFilter.initializer_pretrain_loss`.

```bash
uv run python tests/generated_trajectories_tests.py \
  --trajectories-dir outputs \
  --tokenizer-model sshleifer/tiny-gpt2 \
  --seq-len-obs 16 \
  --batch-size 2
```

