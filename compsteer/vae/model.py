"""
Asymmetric VAE for aligning target robot action distributions to the
pre-trained GR00T action latent distribution (ATE align stage).

Architecture:
  Encoder  : MLP  →  (μ_q, log_σ_q)  in  ℝ^latent_dim
  Decoder  : MLP  →  reconstructed target action chunk

The "asymmetric" property comes from the KL term: instead of KL(q || N(0,I)),
we use KL(q || p_s) where p_s = N(μ_s, σ_s²) is the source action latent
distribution estimated from pre-training data.  This is a mode-seeking
(reverse KL) objective that pushes z into the modes of the source distribution
rather than covering it uniformly.

When source stats are unknown (source_mean=None), the model falls back to
the standard VAE KL, which is a reasonable initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Simple feed-forward block with LayerNorm and SiLU activations."""

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AsymmetricVAE(nn.Module):
    """
    Asymmetric VAE aligning target actions → source action latent distribution.

    Args:
        action_dim:   Dimensionality of one action step (target robot DoF).
        action_horizon: Number of consecutive action steps per chunk.
        latent_dim:   Dimensionality of latent code z.
        hidden_dims:  Hidden layer sizes for encoder and decoder.
        source_mean:  Optional μ_s from source latent distribution (ℝ^latent_dim).
        source_std:   Optional σ_s from source latent distribution (ℝ^latent_dim).
        dropout:      Dropout probability inside MLP blocks.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        source_mean: Tensor | None = None,
        source_std: Tensor | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        in_dim = action_dim * action_horizon

        # Encoder: action chunk → (μ_q, log_σ_q)
        self.encoder_body = MLP(in_dim, hidden_dims, hidden_dims[-1], dropout)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_std = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: z → reconstructed action chunk
        self.decoder = MLP(latent_dim, list(reversed(hidden_dims)), in_dim, dropout)

        # Source distribution parameters (for reverse KL).
        # Stored as buffers so they move with the model to the right device.
        if source_mean is not None:
            self.register_buffer("source_mean", source_mean.float())
        else:
            self.register_buffer("source_mean", torch.zeros(latent_dim))

        if source_std is not None:
            self.register_buffer("source_std", source_std.float())
        else:
            self.register_buffer("source_std", torch.ones(latent_dim))

    # ── Encoder ─────────────────────────────────────────────────────────
    def encode(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode a batch of action chunks.

        Args:
            actions: (B, T, action_dim)  or  (B, T*action_dim)

        Returns:
            mu:      (B, latent_dim)
            log_std: (B, latent_dim)
        """
        x = actions.reshape(actions.size(0), -1)   # (B, T*action_dim)
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        log_std = self.fc_log_std(h).clamp(-4.0, 2.0)
        return mu, log_std

    # ── Reparameterisation ───────────────────────────────────────────────
    @staticmethod
    def reparameterise(mu: Tensor, log_std: Tensor) -> Tensor:
        """z = μ + σ·ε,  ε ~ N(0, I)"""
        std = log_std.exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    # ── Decoder ─────────────────────────────────────────────────────────
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent code back to action chunk.

        Args:
            z: (B, latent_dim)

        Returns:
            actions_recon: (B, T, action_dim)
        """
        flat = self.decoder(z)                      # (B, T*action_dim)
        return flat.reshape(z.size(0), self.action_horizon, self.action_dim)

    # ── Forward ─────────────────────────────────────────────────────────
    def forward(self, actions: Tensor) -> dict[str, Tensor]:
        """
        Full encode-reparameterise-decode pass.

        Returns dict with keys:
            recon     – reconstructed action chunk (B, T, action_dim)
            mu        – encoder mean (B, latent_dim)
            log_std   – encoder log std (B, latent_dim)
            z         – sampled latent (B, latent_dim)
        """
        mu, log_std = self.encode(actions)
        z = self.reparameterise(mu, log_std)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "log_std": log_std, "z": z}

    # ── Steering vector utility ──────────────────────────────────────────
    @torch.no_grad()
    def extract_steering_vector(self, target_actions: Tensor) -> Tensor:
        """
        Compute the steering vector for a batch of target demos.

        Δv = mean(μ_q) - source_mean

        Args:
            target_actions: (N, T, action_dim)  —  all target demo action chunks

        Returns:
            delta_v: (latent_dim,)  steering vector in latent space
        """
        self.eval()
        mu, _ = self.encode(target_actions)
        delta_v = mu.mean(dim=0) - self.source_mean
        return delta_v

    def update_source_stats(self, source_latents: Tensor) -> None:
        """
        Update source distribution statistics from observed source latents.

        Args:
            source_latents: (N, latent_dim)  — latents extracted from source demos
        """
        self.source_mean.copy_(source_latents.mean(dim=0))
        self.source_std.copy_(source_latents.std(dim=0).clamp(min=1e-6))
