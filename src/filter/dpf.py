from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .likelihood import DPFOutput, ParticleLikelihoodHead
from .mlp import mlp


class DeepParticleFilter(nn.Module):
    """
    Observation-only differentiable particle filter.

    Training uses observations only.
    True hidden states are never used inside the training losses.
    """

    def __init__(
        self,
        dim_x,
        dim_y,
        n_particles,
        n_pred,
        dt,
        hidden_size=96,
        obs_embed_dim=32,
        init_noise_dim=8,
        transition_noise_dim=8,
        likelihood_mode="hybrid",
        likelihood_match_dim=64,
        likelihood_num_heads=4,
        ess_threshold=0.5,
        contrastive_temperature=0.10,
    ):
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_particles = n_particles
        self.n_pred = n_pred
        self.dt = dt
        self.obs_embed_dim = obs_embed_dim
        self.init_noise_dim = init_noise_dim
        self.transition_noise_dim = transition_noise_dim
        self.ess_threshold = ess_threshold
        self.contrastive_temperature = contrastive_temperature

        # Shared observation encoder
        self.obs_encoder = mlp(
            [dim_y, hidden_size, hidden_size, obs_embed_dim],
            norm=False,
            dropout=0.0,
        )

        # Initial particle sampler x0 = g(y0, z0)
        self.init_sampler = mlp(
            [obs_embed_dim + init_noise_dim, hidden_size, hidden_size, hidden_size, dim_x],
            norm=False,
            dropout=0.0,
        )

        # Transition residuals
        self.transition_det = mlp(
            [dim_x + obs_embed_dim + 1, hidden_size, hidden_size, hidden_size, dim_x],
            norm=False,
            dropout=0.0,
            zero_last=True,
        )
        self.transition_stoch = mlp(
            [
                dim_x + obs_embed_dim + transition_noise_dim + 1,
                hidden_size,
                hidden_size,
                hidden_size,
                dim_x,
            ],
            norm=False,
            dropout=0.0,
            zero_last=True,
        )

        # Observation decoder for reconstruction / plotting
        # Soft anchor: first latent coordinates should carry observable quantity
        self.obs_residual_net = mlp(
            [dim_x, hidden_size, hidden_size, hidden_size, dim_y],
            norm=False,
            dropout=0.0,
            zero_last=True,
        )

        # Likelihood scorer
        self.likelihood_head = ParticleLikelihoodHead(
            dim_x=dim_x,
            dim_y=dim_y,
            obs_embed_dim=obs_embed_dim,
            hidden_size=hidden_size,
            mode=likelihood_mode,
            match_dim=likelihood_match_dim,
            num_heads=likelihood_num_heads,
        )

        self._particles = None
        self._log_weights = None
        self._last_obs = None

    @property
    def particles(self):
        return self._particles

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def weights(self):
        return self._log_weights.exp()

    def encode_obs(self, obs):
        return self.obs_encoder(obs)

    def _uniform_log_weights(self, batch_size, device, dtype):
        return torch.full(
            (batch_size, self.n_particles),
            -math.log(self.n_particles),
            device=device,
            dtype=dtype,
        )

    def _expand_obs_features(self, obs, n_particles):
        feat = self.encode_obs(obs)[:, None, :]
        return feat.expand(-1, n_particles, -1)

    def _dt_feature(self, batch_size, n_particles, device, dtype):
        return torch.full(
            (batch_size, n_particles, 1),
            self.dt,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------
    # Observation decoder / estimates
    # ------------------------------------------------------------
    def particle_observation_predictions(self, particles=None):
        if particles is None:
            particles = self._particles
        base = particles[..., : self.dim_y]
        residual = 0.10 * self.obs_residual_net(particles)
        return base + residual

    def estimate_state_from(self, particles, log_weights):
        weights = log_weights.exp().unsqueeze(-1)
        return torch.sum(weights * particles, dim=1)

    def estimate_observation_from(self, particles, log_weights):
        weights = log_weights.exp().unsqueeze(-1)
        particle_obs = self.particle_observation_predictions(particles)
        return torch.sum(weights * particle_obs, dim=1)

    def effective_sample_size_from(self, log_weights):
        weights = log_weights.exp()
        return 1.0 / torch.sum(weights.square(), dim=1).clamp_min(1e-12)

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def sample_initial_particles(self, init_obs):
        batch_size = init_obs.shape[0]
        device = init_obs.device
        dtype = init_obs.dtype

        obs_feat = self._expand_obs_features(init_obs, self.n_particles)
        z0 = torch.randn(
            batch_size,
            self.n_particles,
            self.init_noise_dim,
            device=device,
            dtype=dtype,
        )
        init_inp = torch.cat([obs_feat, z0], dim=-1)
        return self.init_sampler(init_inp)

    def reset(self, batch_size, device=None, init_obs=None):
        if device is None:
            device = next(self.parameters()).device

        if init_obs is None:
            self._particles = torch.randn(batch_size, self.n_particles, self.dim_x, device=device)
            self._last_obs = None
        else:
            self._particles = self.sample_initial_particles(init_obs)
            self._last_obs = init_obs

        self._log_weights = self._uniform_log_weights(
            batch_size=batch_size,
            device=self._particles.device,
            dtype=self._particles.dtype,
        )
        return self.estimate_state_from(self._particles, self._log_weights)

    # ------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------
    def transition_step_from_particles(self, particles, prev_obs):
        batch_size, n_particles, _ = particles.shape
        device = particles.device
        dtype = particles.dtype

        obs_feat = self._expand_obs_features(prev_obs, n_particles)
        dt_feat = self._dt_feature(batch_size, n_particles, device, dtype)

        det_inp = torch.cat([particles, obs_feat, dt_feat], dim=-1)
        det_delta = self.transition_det(det_inp)

        z = torch.randn(
            batch_size,
            n_particles,
            self.transition_noise_dim,
            device=device,
            dtype=dtype,
        )
        stoch_inp = torch.cat([particles, obs_feat, z, dt_feat], dim=-1)
        stoch_delta = self.transition_stoch(stoch_inp)

        # Center the stochastic part to prevent global drift of the whole cloud.
        stoch_delta = stoch_delta - stoch_delta.mean(dim=1, keepdim=True)

        return particles + self.dt * (det_delta + stoch_delta)

    def predict_particles(self, particles, prev_obs, n_steps=None):
        if n_steps is None:
            n_steps = self.n_pred
        x = particles
        for _ in range(n_steps):
            x = self.transition_step_from_particles(x, prev_obs)
        return x

    def predict_once(self):
        if self._last_obs is None:
            raise RuntimeError(
                "predict_once() requires previous observation. Call reset(..., init_obs=...) first."
            )
        self._particles = self.transition_step_from_particles(self._particles, self._last_obs)
        return self._particles

    def predict_n_steps(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_pred
        for _ in range(n_steps):
            self.predict_once()
        return self._particles

    # ------------------------------------------------------------
    # Likelihood / weight update
    # ------------------------------------------------------------
    def particle_log_scores(self, obs, particles):
        obs_feat = self.encode_obs(obs)
        pred_obs = self.particle_observation_predictions(particles)
        return self.likelihood_head(obs, obs_feat, particles, pred_obs)

    def cloud_score_matrix(self, particles, obs_candidates):
        b = particles.shape[0]
        scores = []
        for k in range(b):
            obs_k = obs_candidates[k : k + 1].expand(b, -1)
            part_scores = self.particle_log_scores(obs_k, particles)  # (B, N)
            cloud_scores = torch.logsumexp(part_scores, dim=1) - math.log(self.n_particles)
            scores.append(cloud_scores)
        return torch.stack(scores, dim=1)  # (B, B)

    def score_spread_loss(self, scores, min_std=0.25):
        std = scores.std(dim=1).mean()
        return F.relu(min_std - std)

    def systematic_resample(self, weights):
        b, n = weights.shape
        cdf = torch.cumsum(weights, dim=1)
        cdf[:, -1] = 1.0

        u0 = torch.rand(b, 1, device=weights.device) / n
        positions = u0 + torch.arange(n, device=weights.device).view(1, n) / n

        idx = (positions.unsqueeze(-1) > cdf.unsqueeze(1)).sum(dim=-1)
        return idx.clamp(max=n - 1)

    def maybe_resample(self, ess=None):
        if ess is None:
            ess = self.effective_sample_size_from(self._log_weights)

        need = ess < (self.ess_threshold * self.n_particles)
        if not need.any():
            return

        weights = self.weights[need]
        idx = self.systematic_resample(weights)

        selected_particles = self._particles[need].gather(
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, self.dim_x),
        )

        self._particles = self._particles.clone()
        self._particles[need] = selected_particles

        self._log_weights = self._log_weights.clone()
        self._log_weights[need] = -math.log(self.n_particles)

    def update(self, obs, resample=True):
        log_like = self.particle_log_scores(obs, self._particles)
        new_log_weights = self._log_weights + log_like
        log_norm = torch.logsumexp(new_log_weights, dim=1, keepdim=True)
        new_log_weights = new_log_weights - log_norm

        self._log_weights = new_log_weights

        posterior_particles = self._particles.clone()
        posterior_log_weights = self._log_weights.clone()

        posterior_state = self.estimate_state_from(posterior_particles, posterior_log_weights)
        posterior_obs = self.estimate_observation_from(posterior_particles, posterior_log_weights)
        ess = self.effective_sample_size_from(posterior_log_weights)

        if resample:
            self.maybe_resample(ess)

        self._last_obs = obs

        return (
            posterior_state,
            posterior_obs,
            log_norm.squeeze(1),
            ess,
            posterior_particles,
            posterior_log_weights,
        )

    def step(self, obs, n_pred=None, resample=True):
        self.predict_n_steps(n_pred)

        prior_particles = self._particles.clone()
        prior_log_weights = self._log_weights.clone()

        prior_state = self.estimate_state_from(prior_particles, prior_log_weights)
        prior_obs = self.estimate_observation_from(prior_particles, prior_log_weights)

        (
            posterior_state,
            posterior_obs,
            log_evidence,
            ess,
            posterior_particles,
            posterior_log_weights,
        ) = self.update(obs, resample=resample)

        return (
            prior_state,
            posterior_state,
            prior_obs,
            posterior_obs,
            log_evidence,
            ess,
            prior_particles,
            posterior_particles,
            prior_log_weights,
            posterior_log_weights,
        )

    def forward(self, obs_seq, resample=False):
        batch_size, seq_len, _ = obs_seq.shape
        device = obs_seq.device

        self.reset(batch_size=batch_size, device=device, init_obs=obs_seq[:, 0])

        init_particles = self._particles.clone()
        init_log_weights = self._log_weights.clone()

        prior_state_estimates = [self.estimate_state_from(init_particles, init_log_weights)]
        posterior_state_estimates = [self.estimate_state_from(init_particles, init_log_weights)]
        prior_obs_estimates = [self.estimate_observation_from(init_particles, init_log_weights)]
        posterior_obs_estimates = [self.estimate_observation_from(init_particles, init_log_weights)]
        log_evidence = [torch.zeros(batch_size, device=device, dtype=obs_seq.dtype)]
        ess_values = [self.effective_sample_size_from(init_log_weights)]
        prior_particle_history = [init_particles]
        posterior_particle_history = [init_particles]
        prior_log_weight_history = [init_log_weights]
        posterior_log_weight_history = [init_log_weights]

        for t in range(1, seq_len):
            (
                prior_state,
                posterior_state,
                prior_obs,
                posterior_obs,
                log_ev,
                ess,
                prior_particles,
                posterior_particles,
                prior_log_weights,
                posterior_log_weights,
            ) = self.step(
                obs=obs_seq[:, t],
                n_pred=self.n_pred,
                resample=resample,
            )

            prior_state_estimates.append(prior_state)
            posterior_state_estimates.append(posterior_state)
            prior_obs_estimates.append(prior_obs)
            posterior_obs_estimates.append(posterior_obs)
            log_evidence.append(log_ev)
            ess_values.append(ess)
            prior_particle_history.append(prior_particles)
            posterior_particle_history.append(posterior_particles)
            prior_log_weight_history.append(prior_log_weights)
            posterior_log_weight_history.append(posterior_log_weights)

        return DPFOutput(
            prior_state_estimates=torch.stack(prior_state_estimates, dim=1),
            posterior_state_estimates=torch.stack(posterior_state_estimates, dim=1),
            prior_obs_estimates=torch.stack(prior_obs_estimates, dim=1),
            posterior_obs_estimates=torch.stack(posterior_obs_estimates, dim=1),
            log_evidence=torch.stack(log_evidence, dim=1),
            ess=torch.stack(ess_values, dim=1),
            prior_particle_history=torch.stack(prior_particle_history, dim=1),
            posterior_particle_history=torch.stack(posterior_particle_history, dim=1),
            prior_log_weight_history=torch.stack(prior_log_weight_history, dim=1),
            posterior_log_weight_history=torch.stack(posterior_log_weight_history, dim=1),
        )

    # ------------------------------------------------------------
    # Self-supervised losses
    # ------------------------------------------------------------
    def particle_diversity_loss(self, particles, min_std=0.03, max_std=3.0):
        std = particles.std(dim=1)
        low = F.relu(min_std - std).mean()
        high = F.relu(std - max_std).mean()
        energy = 1e-4 * particles.square().mean()
        return low + 0.1 * high + energy

    def _sample_time_indices(self, seq_len, n_samples, device):
        if seq_len <= 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        n_samples = min(n_samples, seq_len - 1)
        return torch.randperm(seq_len - 1, device=device)[:n_samples] + 1

    def initializer_pretrain_loss(self, obs_seq, n_obs_samples=64):
        flat_obs = obs_seq.reshape(-1, self.dim_y)

        if flat_obs.shape[0] > n_obs_samples:
            idx = torch.randint(0, flat_obs.shape[0], (n_obs_samples,), device=flat_obs.device)
            obs = flat_obs[idx]
        else:
            obs = flat_obs

        particles = self.sample_initial_particles(obs)
        log_weights = self._uniform_log_weights(
            batch_size=obs.shape[0],
            device=obs.device,
            dtype=particles.dtype,
        )
        obs_hat = self.estimate_observation_from(particles, log_weights)

        recon = F.smooth_l1_loss(obs_hat, obs)
        diversity = self.particle_diversity_loss(particles)
        total = recon + 0.05 * diversity

        return {
            "loss": total,
            "recon": recon,
            "diversity": diversity,
        }

    def bootstrap_predict_from_observation(self, prev_obs, n_steps=None):
        particles = self.sample_initial_particles(prev_obs)
        particles = self.predict_particles(particles, prev_obs, n_steps=n_steps)
        log_weights = self._uniform_log_weights(
            batch_size=prev_obs.shape[0],
            device=prev_obs.device,
            dtype=particles.dtype,
        )
        return particles, log_weights

    def transition_pretrain_loss(self, obs_seq, n_time_samples=2):
        _, seq_len, _ = obs_seq.shape
        t_idx = self._sample_time_indices(seq_len, n_time_samples, obs_seq.device)

        pred_losses = []
        div_losses = []

        for t in t_idx:
            pred_particles, pred_log_weights = self.bootstrap_predict_from_observation(
                prev_obs=obs_seq[:, t - 1],
                n_steps=self.n_pred,
            )
            pred_obs = self.estimate_observation_from(pred_particles, pred_log_weights)

            pred_losses.append(F.smooth_l1_loss(pred_obs, obs_seq[:, t]))
            div_losses.append(self.particle_diversity_loss(pred_particles))

        pred_recon = torch.stack(pred_losses).mean()
        diversity = torch.stack(div_losses).mean()
        total = pred_recon + 0.05 * diversity

        return {
            "loss": total,
            "pred_recon": pred_recon,
            "diversity": diversity,
        }

    def likelihood_pretrain_loss(self, obs_seq, n_time_samples=2):
        batch_size, seq_len, _ = obs_seq.shape
        t_idx = self._sample_time_indices(seq_len, n_time_samples, obs_seq.device)

        nce_losses = []
        spread_losses = []

        for t in t_idx:
            pred_particles, _ = self.bootstrap_predict_from_observation(
                prev_obs=obs_seq[:, t - 1],
                n_steps=self.n_pred,
            )

            score_matrix = self.cloud_score_matrix(pred_particles, obs_seq[:, t])
            targets = torch.arange(batch_size, device=obs_seq.device)
            nce_losses.append(
                F.cross_entropy(score_matrix / self.contrastive_temperature, targets)
            )

            pos_scores = self.particle_log_scores(obs_seq[:, t], pred_particles)
            spread_losses.append(self.score_spread_loss(pos_scores))

        nce = torch.stack(nce_losses).mean()
        spread = torch.stack(spread_losses).mean()
        total = nce + 0.10 * spread

        return {
            "loss": total,
            "nce": nce,
            "spread": spread,
        }

    def end_to_end_selfsup_loss(self, obs_seq, resample=False, n_nce_steps=2):
        out = self(obs_seq, resample=resample)

        post_recon = F.smooth_l1_loss(out.posterior_obs_estimates, obs_seq)

        if obs_seq.shape[1] > 1:
            prior_pred = F.smooth_l1_loss(
                out.prior_obs_estimates[:, 1:],
                obs_seq[:, 1:],
            )

            prior_err = (out.prior_obs_estimates[:, 1:] - obs_seq[:, 1:]).abs().mean(dim=-1)
            post_err = (out.posterior_obs_estimates[:, 1:] - obs_seq[:, 1:]).abs().mean(dim=-1)
            innovation = F.relu(post_err - prior_err + 1e-3).mean()

            t_idx = self._sample_time_indices(obs_seq.shape[1], n_nce_steps, obs_seq.device)
            nce_losses = []
            spread_losses = []
            for t in t_idx:
                score_matrix = self.cloud_score_matrix(
                    out.prior_particle_history[:, t],
                    obs_seq[:, t],
                )
                targets = torch.arange(obs_seq.shape[0], device=obs_seq.device)
                nce_losses.append(
                    F.cross_entropy(score_matrix / self.contrastive_temperature, targets)
                )

                pos_scores = self.particle_log_scores(
                    obs_seq[:, t],
                    out.prior_particle_history[:, t],
                )
                spread_losses.append(self.score_spread_loss(pos_scores))

            nce = torch.stack(nce_losses).mean()
            spread = torch.stack(spread_losses).mean()

            ess_reg = ((out.ess[:, 1:] / self.n_particles) - 0.40).square().mean()

            diversity = self.particle_diversity_loss(
                out.prior_particle_history[:, 1:].reshape(-1, self.n_particles, self.dim_x)
            )
        else:
            prior_pred = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
            innovation = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
            nce = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
            spread = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
            ess_reg = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
            diversity = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)

        total = (
            1.00 * post_recon
            + 0.50 * prior_pred
            + 0.25 * nce
            + 0.10 * innovation
            + 0.05 * spread
            + 0.05 * diversity
            + 0.02 * ess_reg
        )

        return {
            "loss": total,
            "post_recon": post_recon,
            "prior_pred": prior_pred,
            "nce": nce,
            "spread": spread,
            "innovation": innovation,
            "ess_reg": ess_reg,
            "diversity": diversity,
        }, out

