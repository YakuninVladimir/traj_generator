from __future__ import annotations

import torch


@torch.no_grad()
def simulate_damped_oscillator_dataset(
    n_sequences: int,
    seq_len_obs: int,
    n_pred: int,
    dt: float,
    spring_k: float = 0.7,
    damping_c: float = 0.12,
    process_std: tuple[float, float] = (0.02, 0.03),
    obs_std: float = 0.08,
    device: str = "cpu",
):
    """
    Generates batched synthetic data.

    Important convention:
      - observation at index 0 is at the initial state
      - then between observation t and t+1 there are exactly n_pred latent steps
    """
    x = torch.zeros(n_sequences, 2, device=device)
    x[:, 0] = torch.empty(n_sequences, device=device).uniform_(-1.5, 1.5)
    x[:, 1] = torch.empty(n_sequences, device=device).uniform_(-0.5, 0.5)

    hidden_at_obs = [x.clone()]
    observations = [x[:, 0:1] + obs_std * torch.randn(n_sequences, 1, device=device)]

    process_std_tensor = torch.tensor(process_std, device=device)

    for _ in range(seq_len_obs - 1):
        for _ in range(n_pred):
            pos = x[:, 0]
            vel = x[:, 1]
            next_pos = pos + dt * vel
            next_vel = vel + dt * (-spring_k * pos - damping_c * vel)
            x = torch.stack([next_pos, next_vel], dim=-1)
            x = x + torch.randn_like(x) * process_std_tensor

        hidden_at_obs.append(x.clone())
        observations.append(x[:, 0:1] + obs_std * torch.randn(n_sequences, 1, device=device))

    hidden_at_obs = torch.stack(hidden_at_obs, dim=1)  # (N, T, 2)
    observations = torch.stack(observations, dim=1)  # (N, T, 1)
    return hidden_at_obs, observations

