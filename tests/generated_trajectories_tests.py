from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from configs import DeepParticleFilterConfig, GeneratedTrajectoriesDatasetConfig
from data.generated_trajectories import make_generated_trajectories_dataloader
from filter.dpf import DeepParticleFilter


def _infer_dim_y(obs_seq: torch.Tensor) -> int:
    if obs_seq.ndim != 3:
        raise ValueError(f"obs_seq expected (B,T,Dy), got {obs_seq.shape}")
    return obs_seq.shape[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-dir", default="outputs")
    parser.add_argument("--tokenizer-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--embedder-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--seq-len-obs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)

    # DPF speed knobs for tests
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--n-pred", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--obs-embed-dim", type=int, default=8)

    args = parser.parse_args()

    torch.manual_seed(7)

    ds_cfg = GeneratedTrajectoriesDatasetConfig(
        trajectories_dir=args.trajectories_dir,
        seq_len_obs=args.seq_len_obs,
        batch_size=args.batch_size,
        tokenizer_model_name=args.tokenizer_model,
        embedder_model_name=args.embedder_model,
        device=args.device,
    )

    dl = make_generated_trajectories_dataloader(ds_cfg, shuffle=False, max_samples=32)
    batch = next(iter(dl))
    obs_seq: torch.Tensor = batch["obs_seq"]

    dim_y = _infer_dim_y(obs_seq)
    # DeepParticleFilter assumes dim_y <= dim_x because it uses particles[..., :dim_y]
    dim_x = dim_y

    dpf_cfg = DeepParticleFilterConfig(
        dim_x=dim_x,
        dim_y=dim_y,
        n_particles=args.n_particles,
        n_pred=args.n_pred,
        dt=args.dt,
        hidden_size=args.hidden_size,
        obs_embed_dim=args.obs_embed_dim,
        init_noise_dim=4,
        transition_noise_dim=4,
        likelihood_mode="mse",
        likelihood_match_dim=16,
        likelihood_num_heads=4,
        ess_threshold=0.5,
        contrastive_temperature=0.10,
    )

    dpf = DeepParticleFilter(
        dim_x=dpf_cfg.dim_x,
        dim_y=dpf_cfg.dim_y,
        n_particles=dpf_cfg.n_particles,
        n_pred=dpf_cfg.n_pred,
        dt=dpf_cfg.dt,
        hidden_size=dpf_cfg.hidden_size,
        obs_embed_dim=dpf_cfg.obs_embed_dim,
        init_noise_dim=dpf_cfg.init_noise_dim,
        transition_noise_dim=dpf_cfg.transition_noise_dim,
        likelihood_mode=dpf_cfg.likelihood_mode,
        likelihood_match_dim=dpf_cfg.likelihood_match_dim,
        likelihood_num_heads=dpf_cfg.likelihood_num_heads,
        ess_threshold=dpf_cfg.ess_threshold,
        contrastive_temperature=dpf_cfg.contrastive_temperature,
    )

    device = ds_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dpf = dpf.to(device)
    obs_seq = obs_seq.to(device)

    opt = AdamW(dpf.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)

    losses = dpf.initializer_pretrain_loss(obs_seq, n_obs_samples=32)
    loss = losses["loss"]
    assert torch.isfinite(loss).all(), f"loss is not finite: {loss}"

    loss.backward()
    opt.step()

    print("generated_trajectories_tests: OK", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()

