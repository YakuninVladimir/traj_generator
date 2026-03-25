from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs import GeneratorConfig, PromptSamplingConfig, TrajectorySamplingConfig

from generator_io import save_jsonl
from generator_llm import load_model
from generator_prompting import generate_prompts
from generator_sharding import shard_slices
from generator_trajectories import generate_trajectories


def _build_config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    prompt_sampling = PromptSamplingConfig(
        temperature=args.prompt_temperature,
        top_p=args.prompt_top_p,
        max_new_tokens=args.prompt_max_new_tokens,
    )
    trajectory_sampling = TrajectorySamplingConfig(
        temperature=args.trajectory_temperature,
        top_p=args.trajectory_top_p,
        max_new_tokens=args.trajectory_max_new_tokens,
    )

    cfg = GeneratorConfig(
        topic=args.topic,
        num_prompts=args.num_prompts,
        trajectories_per_prompt=args.trajectories_per_prompt,
        num_shards=args.num_shards,
        prompt_model_name=args.prompt_model,
        trajectory_model_name=args.trajectory_model,
        prompt_initial_batch_size=args.prompt_batch_size,
        trajectory_batch_size=args.trajectory_batch_size,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        prompt_sampling=prompt_sampling,
        trajectory_sampling=trajectory_sampling,
    )
    return cfg


def run_generation(cfg: GeneratorConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    out_dir: Path = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    p_tok, p_model = load_model(cfg.prompt_model_name, device)
    t_tok, t_model = load_model(cfg.trajectory_model_name, device)

    print("Generating prompts...")
    prompts = generate_prompts(
        cfg.topic,
        cfg.num_prompts,
        p_tok,
        p_model,
        batch_size=cfg.prompt_initial_batch_size,
        sampling=cfg.prompt_sampling,
    )
    prompt_records = [{"prompt_id": i, "prompt": p} for i, p in enumerate(prompts)]
    save_jsonl(out_dir / "prompts.jsonl", prompt_records)

    print("Generating trajectories by shards...")
    for shard_id, (start, end) in enumerate(shard_slices(cfg.num_prompts, cfg.num_shards)):
        if start == end:
            continue
        shard_prompts = prompts[start:end]
        print(f"generating shard {shard_id} from {cfg.num_shards} (prompts {start}:{end})")

        trajectories = generate_trajectories(
            shard_prompts,
            cfg.trajectories_per_prompt,
            t_tok,
            t_model,
            batch_size=cfg.trajectory_batch_size,
            sampling=cfg.trajectory_sampling,
            prompt_id_offset=start,
        )
        save_jsonl(out_dir / f"trajectories_{shard_id}.jsonl", trajectories)

    print("Generation finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompts + trajectories with Transformers (scripts).")
    default_cfg = GeneratorConfig()

    # Main counts / shards
    parser.add_argument("--topic", default=default_cfg.topic)
    parser.add_argument("--num-prompts", type=int, default=default_cfg.num_prompts)
    parser.add_argument(
        "--trajectories-per-prompt",
        type=int,
        default=default_cfg.trajectories_per_prompt,
    )
    parser.add_argument("--num-shards", type=int, default=default_cfg.num_shards)

    # Models
    parser.add_argument("--prompt-model", default=default_cfg.prompt_model_name)
    parser.add_argument("--trajectory-model", default=default_cfg.trajectory_model_name)

    # Batching
    parser.add_argument("--prompt-batch-size", type=int, default=default_cfg.prompt_initial_batch_size)
    parser.add_argument("--trajectory-batch-size", type=int, default=default_cfg.trajectory_batch_size)

    # Sampling (prompts)
    parser.add_argument("--prompt-temperature", type=float, default=0.8)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-max-new-tokens", type=int, default=64)

    # Sampling (trajectories)
    parser.add_argument("--trajectory-temperature", type=float, default=0.7)
    parser.add_argument("--trajectory-top-p", type=float, default=0.95)
    parser.add_argument("--trajectory-max-new-tokens", type=int, default=256)

    parser.add_argument("--output-dir", default=str(default_cfg.output_dir))
    parser.add_argument("--seed", type=int, default=default_cfg.seed)

    args = parser.parse_args()
    cfg = _build_config_from_args(args)
    run_generation(cfg)


if __name__ == "__main__":
    main()

