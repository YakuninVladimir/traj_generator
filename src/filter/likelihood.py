from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import mlp


class ParticleLikelihoodHead(nn.Module):
    """
    Innovation-aware particle likelihood scorer.

    The main difference from the previous version is that the scorer sees:
      - previous observation (`prev_obs`)
      - innovation in observation space
      - innovation mismatch between true obs and particle-decoded obs
      - global cloud context

    Inputs:
      obs       : (B, Dy)
      prev_obs  : (B, Dy)
      obs_feat  : (B, E)
      particles : (B, N, Dx)
      pred_obs  : (B, N, Dy)

    Output:
      scores    : (B, N), unnormalized log-scores
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
        self.prev_obs_proj = nn.Linear(dim_y, match_dim)

        # Particle-side matching uses particle + cloud context
        self.particle_match_proj = nn.Linear(2 * hidden_size, match_dim)
        self.bilinear_proj = nn.Linear(match_dim, match_dim, bias=False)

        # MLP critic
        self.mlp_score = mlp(
            [
                2 * hidden_size + obs_embed_dim + match_dim + 7 * dim_y,
                hidden_size,
                hidden_size,
                1,
            ],
            norm=False,
            dropout=0.0,
        )

        self.attn_local = mlp(
            [match_dim + 3 * dim_y, hidden_size, 1],
            norm=False,
            dropout=0.0,
        )

        # Learned precisions / scales
        self.log_precision_obs = nn.Parameter(torch.zeros(dim_y))
        self.log_precision_delta = nn.Parameter(torch.zeros(dim_y))
        self.obs_delta_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.cos_scale = nn.Parameter(torch.tensor(1.0))
        self.attn_scale = nn.Parameter(torch.tensor(1.0))

        # Hybrid combiner over 5 components
        self.hybrid_logits = nn.Parameter(torch.zeros(5))

    def _standardize(self, score: torch.Tensor) -> torch.Tensor:
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True).clamp_min(1e-4)
        return (score - mean) / std

    def forward(self, obs, prev_obs, obs_feat, particles, pred_obs):
        b, n, _ = particles.shape

        obs_expanded = obs[:, None, :].expand(-1, n, -1)
        prev_obs_expanded = prev_obs[:, None, :].expand(-1, n, -1)
        obs_feat_expanded = obs_feat[:, None, :].expand(-1, n, -1)

        # Per-particle latent features
        part_feat = self.particle_encoder(particles)  # (B, N, H)

        # Cloud context (mean over particles)
        cloud_ctx = part_feat.mean(dim=1, keepdim=True).expand(-1, n, -1)  # (B, N, H)

        # Encoded decoded-observation features
        pred_feat = self.pred_obs_encoder(pred_obs)  # (B, N, M)

        # Residuals / innovations
        residual = obs_expanded - pred_obs  # (B, N, Dy)
        delta_true = obs_expanded - prev_obs_expanded  # (B, N, Dy)
        delta_pred = pred_obs - prev_obs_expanded  # (B, N, Dy)
        delta_resid = delta_true - delta_pred  # (B, N, Dy)

        # 1) innovation-aware MSE score
        precision_obs = F.softplus(self.log_precision_obs)[None, None, :] + 1e-4
        precision_delta = F.softplus(self.log_precision_delta)[None, None, :] + 1e-4
        mix = torch.sigmoid(self.obs_delta_mix_logit)

        score_mse_obs = -(precision_obs * residual.square()).sum(dim=-1)
        score_mse_delta = -(precision_delta * delta_resid.square()).sum(dim=-1)
        score_mse = (1.0 - mix) * score_mse_obs + mix * score_mse_delta

        # 2) cosine score
        obs_query = self.obs_match_proj(obs_feat) + self.prev_obs_proj(obs - prev_obs)  # (B, M)
        obs_query = F.normalize(obs_query, dim=-1)
        pred_match = F.normalize(pred_feat, dim=-1)
        score_cos = self.cos_scale * (obs_query[:, None, :] * pred_match).sum(dim=-1)

        # 3) bilinear score
        particle_match = self.particle_match_proj(torch.cat([part_feat, cloud_ctx], dim=-1))  # (B, N, M)
        bilinear_q = self.bilinear_proj(obs_query)[:, None, :]  # (B, 1, M)
        score_bilinear = (bilinear_q * particle_match).sum(dim=-1) / math.sqrt(self.match_dim)

        # 4) explicit MLP critic
        mlp_inp = torch.cat(
            [
                part_feat,
                cloud_ctx,
                obs_feat_expanded,
                pred_feat,
                pred_obs,
                residual,
                delta_true,
                delta_pred,
                delta_resid,
                residual.square(),
                delta_resid.square(),
            ],
            dim=-1,
        )
        score_mlp = self.mlp_score(mlp_inp).squeeze(-1)

        # 5) lightweight attention-style score
        q = obs_query.view(b, self.num_heads, self.head_dim)  # (B, H, Dh)
        k = particle_match.view(b, n, self.num_heads, self.head_dim)  # (B, N, H, Dh)
        attn_logits = (q[:, None, :, :] * k).sum(dim=-1).mean(dim=-1) / math.sqrt(self.head_dim)

        local = self.attn_local(torch.cat([pred_feat, residual, delta_pred, delta_resid], dim=-1)).squeeze(-1)
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

        # Only center, do not squash
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

