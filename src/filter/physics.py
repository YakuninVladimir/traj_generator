from __future__ import annotations

import torch
import torch.nn as nn


class DampedOscillatorPhysics(nn.Module):
    """
    State: x = [position, velocity]

    One latent integration step:
        p_{t+1} = p_t + dt * v_t
        v_{t+1} = v_t + dt * (-k * p_t - c * v_t)
    """

    def __init__(self, dt: float = 0.05, spring_k: float = 0.7, damping_c: float = 0.12):
        super().__init__()
        self.dt = dt
        self.spring_k = spring_k
        self.damping_c = damping_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[..., 0]
        vel = x[..., 1]
        next_pos = pos + self.dt * vel
        next_vel = vel + self.dt * (-self.spring_k * pos - self.damping_c * vel)
        return torch.stack([next_pos, next_vel], dim=-1)

