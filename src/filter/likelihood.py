from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp


class ParticleLikelihoodHead(nn.Module):
    """
    Pluggable particle likelihood scorer.

    Modes:
      - "mse"
      - "cosine"
      - "bilinear"
      - "mlp"
      - "attention"
      - "hybrid"      (learned weighted combination of all of the above)

    Input:
      obs      : (B, Dy)
      obs_feat : (B, E)
      particles: (B, N, Dx)
      pred_obs : (B, N, Dy)

    Output:
      scores   : (B, N), unnormalized log-scores
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        obs_embed_dim: int,
        hidden_size: int,
        mode: str = "hybrid",
        match_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        assert match_dim % num_heads == 0

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.obs_embed_dim = obs_embed_dim
        self.hidden_size = hidden_size
        self.mode = mode
        self.match_dim = match_dim
        self.num_heads = num_heads
        self.head_dim = match_dim // num_heads

        self.particle_encoder = mlp(
            [dim_x, hidden_size, hidden_size],
            norm=False,
            dropout=0.0,
        )

        self.pred_obs_encoder = mlp(
            [dim_y, hidden_size, match_dim],
            norm=False,
            dropout=0.0,
        )

        self.obs_match_proj = nn.Linear(obs_embed_dim, match_dim)
        self.particle_match_proj = nn.Linear(hidden_size, match_dim)
        self.bilinear_proj = nn.Linear(match_dim, match_dim, bias=False)

        self.mlp_score = mlp(
            [hidden_size + obs_embed_dim + match_dim + 2 * dim_y, hidden_size, hidden_size, 1],
            norm=False,
            dropout=0.0,
        )

        self.attn_local = mlp(
            [match_dim + 2 * dim_y, hidden_size, 1],
            norm=False,
            dropout=0.0,
        )

        self.log_precision = nn.Parameter(torch.zeros(dim_y))
        self.mse_bias = nn.Parameter(torch.zeros(1))
        self.cos_scale = nn.Parameter(torch.tensor(1.0))
        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        self.hybrid_logits = nn.Parameter(torch.zeros(5))

    def _standardize(self, score: torch.Tensor) -> torch.Tensor:
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True).clamp_min(1e-4)
        return (score - mean) / std

    def forward(self, obs, obs_feat, particles, pred_obs):
        """
        Returns centered per-particle log-scores.
        """
        b, n, _ = particles.shape

        obs_expanded = obs[:, None, :].expand(-1, n, -1)
        obs_feat_expanded = obs_feat[:, None, :].expand(-1, n, -1)

        part_feat = self.particle_encoder(particles)  # (B, N, H)
        pred_feat = self.pred_obs_encoder(pred_obs)  # (B, N, M)
        residual = obs_expanded - pred_obs  # (B, N, Dy)

        # 1) MSE-like score with learned precision
        precision = F.softplus(self.log_precision)[None, None, :] + 1e-4
        score_mse = -(precision * residual.square()).sum(dim=-1) + self.mse_bias

        # 2) Cosine score between encoded observation and decoded particle observation
        obs_match = F.normalize(self.obs_match_proj(obs_feat), dim=-1)  # (B, M)
        pred_match = F.normalize(pred_feat, dim=-1)  # (B, N, M)
        score_cos = self.cos_scale * (obs_match[:, None, :] * pred_match).sum(dim=-1)

        # 3) Bilinear score between observation embedding and particle embedding
        particle_match = self.particle_match_proj(part_feat)  # (B, N, M)
        bilinear_q = self.bilinear_proj(obs_match)[:, None, :]  # (B, 1, M)
        score_bilinear = (bilinear_q * particle_match).sum(dim=-1) / math.sqrt(self.match_dim)

        # 4) MLP critic with explicit residual features
        mlp_inp = torch.cat(
            [part_feat, obs_feat_expanded, pred_feat, residual, residual.square()],
            dim=-1,
        )
        score_mlp = self.mlp_score(mlp_inp).squeeze(-1)

        # 5) Attention-style score
        q = self.obs_match_proj(obs_feat).view(b, self.num_heads, self.head_dim)  # (B, H, Dh)
        k = particle_match.view(b, n, self.num_heads, self.head_dim)  # (B, N, H, Dh)
        attn_logits = (q[:, None, :, :] * k).sum(dim=-1).mean(dim=-1) / math.sqrt(self.head_dim)
        local = self.attn_local(torch.cat([pred_feat, residual, residual.square()], dim=-1)).squeeze(-1)
        score_attention = self.attn_scale * attn_logits + 0.25 * local

        if self.mode == "mse":
            score = score_mse
        elif self.mode == "cosine":
            score = score_cos
        elif self.mode == "bilinear":
            score = score_bilinear
        elif self.mode == "mlp":
            score = score_mlp
        elif self.mode == "attention":
            score = score_attention
        elif self.mode == "hybrid":
            comps = torch.stack(
                [
                    self._standardize(score_mse),
                    self._standardize(score_cos),
                    self._standardize(score_bilinear),
                    self._standardize(score_mlp),
                    self._standardize(score_attention),
                ],
                dim=-1,
            )  # (B, N, 5)
            weights = torch.softmax(self.hybrid_logits, dim=0)
            score = (comps * weights[None, None, :]).sum(dim=-1)
        else:
            raise ValueError(f"Unknown likelihood mode: {self.mode}")

        # center only; do not squash through sigmoid
        score = score - score.mean(dim=1, keepdim=True)
        return score


@dataclass
class DPFOutput:
    prior_state_estimates: torch.Tensor
    posterior_state_estimates: torch.Tensor
    prior_obs_estimates: torch.Tensor
    posterior_obs_estimates: torch.Tensor
    log_evidence: torch.Tensor
    ess: torch.Tensor
    prior_particle_history: torch.Tensor
    posterior_particle_history: torch.Tensor
    prior_log_weight_history: torch.Tensor
    posterior_log_weight_history: torch.Tensor

