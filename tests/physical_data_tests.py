from __future__ import annotations

import torch
from torch.optim import AdamW

from filter.data_simulation import simulate_damped_oscillator_dataset
from filter.dpf import DeepParticleFilter


def main() -> None:
    torch.manual_seed(7)
    device = "cpu"

    # Parameters taken from `filter.ipynb` (see notebook cell with dim_x/dim_y, n_particles, n_pred, dt, etc.)
    dim_x = 2
    dim_y = 1
    n_particles = 256
    n_pred = 5
    dt = 0.05
    seq_len_obs = 100
    spring_k = 0.7
    damping_c = 0.12
    process_std = (0.02, 0.03)
    obs_std = 0.08
    hidden_size = 96
    obs_embed_dim = 32
    init_noise_dim = 8
    transition_noise_dim = 8
    likelihood_mode = "hybrid"
    likelihood_match_dim = 64
    likelihood_num_heads = 4
    ess_threshold = 0.5
    contrastive_temperature = 0.10

    # Keep sequences small enough for a single-step test.
    n_sequences = 8
    _, obs_seq = simulate_damped_oscillator_dataset(
        n_sequences=n_sequences,
        seq_len_obs=seq_len_obs,
        n_pred=n_pred,
        dt=dt,
        spring_k=spring_k,
        damping_c=damping_c,
        process_std=process_std,
        obs_std=obs_std,
        device=device,
    )

    dpf = DeepParticleFilter(
        dim_x=dim_x,
        dim_y=dim_y,
        n_particles=n_particles,
        n_pred=n_pred,
        dt=dt,
        hidden_size=hidden_size,
        obs_embed_dim=obs_embed_dim,
        init_noise_dim=init_noise_dim,
        transition_noise_dim=transition_noise_dim,
        likelihood_mode=likelihood_mode,
        likelihood_match_dim=likelihood_match_dim,
        likelihood_num_heads=likelihood_num_heads,
        ess_threshold=ess_threshold,
        contrastive_temperature=contrastive_temperature,
    ).to(device)

    opt = AdamW(dpf.parameters(), lr=8e-4)
    opt.zero_grad(set_to_none=True)

    losses = dpf.initializer_pretrain_loss(obs_seq, n_obs_samples=64)
    loss = losses["loss"]
    assert torch.isfinite(loss).all(), f"loss is not finite: {loss}"

    loss.backward()
    opt.step()

    print("physical_data_tests: OK", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()

