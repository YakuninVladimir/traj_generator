from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from filter.dpf import DeepParticleFilter
from filter.train_filter_rebuilt_config import build_stage_specs
from data.generated_trajectories import make_generated_trajectories_dataloader
from data.physical_datasets import make_physical_dataloader
from configs import GeneratedTrajectoriesDatasetConfig, PhysicalDatasetConfig


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def set_stage_trainable(dpf: DeepParticleFilter, stage_name: str) -> None:
    # Keep the mapping identical to `filter_rebuilt.ipynb`.
    modules = {
        "obs_encoder": dpf.obs_encoder,
        "init_sampler": dpf.init_sampler,
        "transition_det": dpf.transition_det,
        "transition_stoch": dpf.transition_stoch,
        "obs_residual_net": dpf.obs_residual_net,
        "likelihood_head": dpf.likelihood_head,
    }

    for m in modules.values():
        set_requires_grad(m, False)

    if stage_name == "init_pretrain":
        for name in ["obs_encoder", "init_sampler", "obs_residual_net"]:
            set_requires_grad(modules[name], True)

    elif stage_name.startswith("transition_pretrain"):
        for name in [
            "obs_encoder",
            "init_sampler",
            "transition_det",
            "transition_stoch",
            "obs_residual_net",
        ]:
            set_requires_grad(modules[name], True)

    elif stage_name.startswith("likelihood_pretrain"):
        for name in ["obs_encoder", "likelihood_head"]:
            set_requires_grad(modules[name], True)

    elif stage_name in {"e2e_warmup", "e2e_finetune"}:
        for m in modules.values():
            set_requires_grad(m, True)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def make_random_window_obs(obs_batch: torch.Tensor, window_len: int) -> torch.Tensor:
    total_len = obs_batch.shape[1]
    if window_len >= total_len:
        return obs_batch

    start = torch.randint(
        low=0,
        high=total_len - window_len + 1,
        size=(1,),
        device=obs_batch.device,
    ).item()
    end = start + window_len
    return obs_batch[:, start:end]


def _maybe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _configure_matplotlib() -> tuple[Any, Any]:
    """
    Returns (plt, sns) if available; otherwise raises.
    """
    # Always use headless backend; we only save plots.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")
    return plt, sns


@torch.no_grad()
def _plot_latest(
    stage_name: str,
    steps: list[int],
    metrics_history: dict[str, list[float]],
    latest_loss_path: Path,
    latest_components_path: Path,
    components: list[str],
) -> None:
    plt, sns = _configure_matplotlib()

    # 1) Loss curve
    plt.figure(figsize=(8, 4))
    loss_vals = metrics_history.get("loss", [])
    n = min(len(steps), len(loss_vals))
    if n > 0:
        x_steps = steps[-n:]
        y_steps = loss_vals[-n:]
        sns.lineplot(x=x_steps, y=y_steps, linewidth=2)
    plt.title(f"{stage_name}: loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    latest_loss_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(latest_loss_path, dpi=120)
    plt.close()

    # 2) Selected components
    if components:
        plt.figure(figsize=(10, 4))
        for name in components:
            vals = metrics_history.get(name, [])
            if not vals:
                continue
            n = min(len(steps), len(vals))
            if n <= 0:
                continue
            x_steps = steps[-n:]
            y_vals = vals[-n:]
            sns.lineplot(x=x_steps, y=y_vals, linewidth=2, label=name)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend()
        plt.title(f"{stage_name}: components")
        plt.tight_layout()
        plt.savefig(latest_components_path, dpi=120)
        plt.close()


def _select_components_for_stage(stage_name: str) -> list[str]:
    if stage_name == "init_pretrain":
        return ["recon", "diversity"]
    if stage_name.startswith("transition_pretrain"):
        return ["pred_recon", "diversity"]
    if stage_name.startswith("likelihood_pretrain"):
        return ["teacher", "nce", "lag", "spread"]
    if stage_name in {"e2e_warmup", "e2e_finetune"}:
        # Keep the number of curves small; the requirement is online visibility, not
        # exhaustive plotting for all scalar terms.
        return ["post_recon", "teacher", "nce", "lag"]
    return []


def _save_checkpoint(
    checkpoint_path: Path,
    dpf: DeepParticleFilter,
    optimizer: torch.optim.Optimizer,
    stage_name: str,
    stage_epoch: int,
    global_step: int,
    loss_value: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": dpf.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stage_name": stage_name,
            "stage_epoch": stage_epoch,
            "global_step": global_step,
            "loss": loss_value,
        },
        checkpoint_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="output/tblogs")

    p.add_argument("--dataset", type=str, choices=["physical", "generated"], default="physical")

    # --- dataset: physical
    p.add_argument("--n-sequences", type=int, default=256)
    p.add_argument("--spring-k", type=float, default=0.7)
    p.add_argument("--damping-c", type=float, default=0.12)
    p.add_argument("--process-std-1", type=float, default=0.02)
    p.add_argument("--process-std-2", type=float, default=0.03)
    p.add_argument("--obs-std", type=float, default=0.08)
    p.add_argument("--physical-device", type=str, default="cpu")

    # --- dataset: generated
    p.add_argument("--trajectories-dir", type=str, default="outputs")
    p.add_argument("--tokenizer-model", type=str, default="sshleifer/tiny-gpt2")
    # Tokenizer and embedder are the same model by project convention.
    p.add_argument("--generated-max-samples", type=int, default=None)
    # Keep embedder on CPU by default to reduce VRAM pressure.
    p.add_argument("--generated-device", type=str, default="cpu")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--dim-x",
        type=int,
        default=None,
        help="Particle state dimension (must be >= inferred dim_y). If omitted, uses inferred dim_y.",
    )
    p.add_argument("--seq-len-obs", type=int, default=100)
    p.add_argument("--n-pred", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-particles", type=int, default=256)

    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--obs-embed-dim", type=int, default=32)
    p.add_argument("--init-noise-dim", type=int, default=8)
    p.add_argument("--transition-noise-dim", type=int, default=8)

    p.add_argument("--likelihood-mode", type=str, default="hybrid")
    p.add_argument("--contrastive-temperature", type=float, default=0.10)
    p.add_argument("--ess-threshold", type=float, default=0.5)

    p.add_argument("--n-repeats", type=int, default=4)
    p.add_argument("--window-len", type=int, default=20)

    p.add_argument("--epochs-scale", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving the single best filter checkpoint.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    logdir = Path(args.logdir)
    tb_dir = logdir / "tblogs"
    plots_dir = logdir / "plots"
    tb_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tb_dir))
    except Exception:
        writer = None

    global_step = 0
    best_loss = float("inf")
    checkpoints_dir = Path(args.checkpoints_dir)
    # Stage-local plotting histories are created inside the stage loop.

    # -----------------------------
    # Dataset + infers dim_y
    # -----------------------------
    loader: DataLoader
    obs_batch_example: torch.Tensor
    if args.dataset == "physical":
        phys_cfg = PhysicalDatasetConfig(
            n_sequences=args.n_sequences,
            seq_len_obs=args.seq_len_obs,
            n_pred=args.n_pred,
            dt=args.dt,
            spring_k=args.spring_k,
            damping_c=args.damping_c,
            process_std=(args.process_std_1, args.process_std_2),
            obs_std=args.obs_std,
            device=args.physical_device,
        )
        loader = make_physical_dataloader(
            phys_cfg,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        (obs_batch_example,) = next(iter(loader))
        obs_batch_example = obs_batch_example.to("cpu")
    else:
        gen_cfg = GeneratedTrajectoriesDatasetConfig(
            trajectories_dir=Path(args.trajectories_dir),
            tokenizer_model_name=args.tokenizer_model,
            embedder_model_name=args.tokenizer_model,
            seq_len_obs=args.seq_len_obs,
            batch_size=args.batch_size,
            device=args.generated_device,
        )
        loader = make_generated_trajectories_dataloader(
            gen_cfg,
            shuffle=True,
            max_samples=args.generated_max_samples,
        )
        batch0 = next(iter(loader))
        obs_batch_example = batch0["obs_seq"].to("cpu")

    inferred_dim_y = int(obs_batch_example.shape[-1])
    dim_x = inferred_dim_y if args.dim_x is None else int(args.dim_x)
    if dim_x < inferred_dim_y:
        raise ValueError(f"--dim-x must be >= inferred dim_y, got dim_x={dim_x}, dim_y={inferred_dim_y}")

    # -----------------------------
    # Model
    # -----------------------------
    dpf = DeepParticleFilter(
        dim_x=dim_x,
        dim_y=inferred_dim_y,
        n_particles=args.n_particles,
        n_pred=args.n_pred,
        dt=args.dt,
        hidden_size=args.hidden_size,
        obs_embed_dim=args.obs_embed_dim,
        init_noise_dim=args.init_noise_dim,
        transition_noise_dim=args.transition_noise_dim,
        likelihood_mode=args.likelihood_mode,
        likelihood_match_dim=64,
        likelihood_num_heads=4,
        ess_threshold=args.ess_threshold,
        contrastive_temperature=args.contrastive_temperature,
    ).to(device)

    # -----------------------------
    # Stage specs
    # -----------------------------
    stage_specs = build_stage_specs(args.n_repeats)
    if args.lr is not None:
        for s in stage_specs:
            s["lr"] = args.lr

    for stage in stage_specs:
        stage_name = stage["name"]
        epochs = int(max(1, round(stage["epochs"] * args.epochs_scale)))

        set_stage_trainable(dpf, stage_name)

        optimizer = torch.optim.AdamW(
            filter(lambda p_: p_.requires_grad, dpf.parameters()),
            lr=stage["lr"],
            weight_decay=1e-5,
        )

        components = _select_components_for_stage(stage_name)

        # Stage-local plotting histories (so curves of different stages don't mix).
        stage_steps: list[int] = []
        stage_metrics_history: dict[str, list[float]] = {"loss": []}
        latest_loss_path = plots_dir / f"latest_loss_{stage_name}.png"
        latest_components_path = plots_dir / f"latest_components_{stage_name}.png"

        for epoch in range(epochs):
            dpf.train()

            pbar = tqdm(
                loader,
                desc=f"[{stage_name}] stage_epoch={epoch:02d}",
                leave=True,
            )
            for batch in pbar:
                out = None
                if args.dataset == "physical":
                    (obs_batch_cpu,) = batch
                    obs_batch = obs_batch_cpu.to(device)
                else:
                    obs_batch = batch["obs_seq"].to(device)

                if stage_name in {"e2e_warmup", "e2e_finetune"}:
                    obs_batch = make_random_window_obs(obs_batch, window_len=args.window_len)

                optimizer.zero_grad()

                if stage_name == "init_pretrain":
                    losses = dpf.initializer_pretrain_loss(
                        obs_seq=obs_batch,
                        n_obs_samples=64,
                    )
                elif stage_name.startswith("transition_pretrain"):
                    losses = dpf.transition_pretrain_loss(
                        obs_seq=obs_batch,
                        n_time_samples=2,
                    )
                elif stage_name.startswith("likelihood_pretrain"):
                    losses = dpf.likelihood_pretrain_loss(
                        obs_seq=obs_batch,
                        n_time_samples=2,
                    )
                elif stage_name == "e2e_warmup":
                    losses, out = dpf.end_to_end_selfsup_loss(
                        obs_seq=obs_batch,
                        resample=False,
                        n_nce_steps=2,
                    )
                elif stage_name == "e2e_finetune":
                    losses, out = dpf.end_to_end_selfsup_loss(
                        obs_seq=obs_batch,
                        resample=True,
                        n_nce_steps=2,
                    )
                else:
                    raise ValueError(stage_name)

                loss = losses["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dpf.parameters(), max_norm=5.0)
                optimizer.step()

                loss_val = _maybe_float(loss)
                global_step += 1

                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                stage_steps.append(global_step)
                stage_metrics_history["loss"].append(loss_val)

                for k, v in losses.items():
                    if k not in stage_metrics_history:
                        stage_metrics_history[k] = []
                    stage_metrics_history[k].append(_maybe_float(v))

                # TensorBoard (if available)
                if writer is not None:
                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", _maybe_float(v), global_step)
                    # Also log ESS mean from e2e
                    if stage_name in {"e2e_warmup", "e2e_finetune"} and out is not None:
                        writer.add_scalar(
                            "train/ess_mean", float(out.ess.mean().detach().cpu().item()), global_step
                        )

                # Online plots (overwrite after every batch)
                _plot_latest(
                    stage_name=stage_name,
                    steps=stage_steps,
                    metrics_history=stage_metrics_history,
                    latest_loss_path=latest_loss_path,
                    latest_components_path=latest_components_path,
                    components=components,
                )

            # Save only one best checkpoint for the whole training run.
            epoch_loss = stage_metrics_history["loss"][-1]
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_ckpt = checkpoints_dir / "best_filter.pt"
                _save_checkpoint(
                    best_ckpt,
                    dpf=dpf,
                    optimizer=optimizer,
                    stage_name=stage_name,
                    stage_epoch=epoch,
                    global_step=global_step,
                    loss_value=epoch_loss,
                )

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

