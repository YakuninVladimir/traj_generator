from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from filter.data_simulation import simulate_damped_oscillator_dataset
from filter.dpf import DeepParticleFilter


def main() -> None:
    torch.manual_seed(7)
    device = "cpu"

    # Very small setup to validate forward/backward quickly.
    _, train_obs = simulate_damped_oscillator_dataset(
        n_sequences=8,
        seq_len_obs=12,
        n_pred=2,
        dt=0.05,
        device=device,
    )

    loader = DataLoader(
        TensorDataset(train_obs),
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )
    (obs_seq,) = next(iter(loader))

    assert obs_seq.ndim == 3, f"obs_seq expected (B,T,Dy), got {obs_seq.shape}"
    assert obs_seq.shape[-1] == 1, f"expected dim_y=1 from physics dataset, got {obs_seq.shape[-1]}"

    dpf = DeepParticleFilter(
        dim_x=2,
        dim_y=1,
        n_particles=16,
        n_pred=2,
        dt=0.05,
        hidden_size=16,
        obs_embed_dim=8,
        init_noise_dim=4,
        transition_noise_dim=4,
        likelihood_mode="mse",
        likelihood_match_dim=16,
        likelihood_num_heads=4,
        ess_threshold=0.5,
        contrastive_temperature=0.10,
    ).to(device)

    opt = AdamW(dpf.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)

    losses = dpf.initializer_pretrain_loss(obs_seq, n_obs_samples=16)
    loss = losses["loss"]
    assert torch.isfinite(loss).all(), f"loss is not finite: {loss}"

    loss.backward()
    opt.step()

    print("smoke_pipeline: OK", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()

