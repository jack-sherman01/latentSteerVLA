"""
Loss functions for the Asymmetric VAE.

The key distinction from a standard VAE:
  Standard: KL(q(z|x) || N(0,I))          — mean-seeking (forward KL)
  ATE:      KL(q(z|x) || p_s(z))          — mode-seeking (reverse KL)

where p_s = N(μ_s, σ_s²) is the source latent distribution.

If source_mean / source_std are not provided, the loss reduces to the
standard VAE ELBO (KL against N(0,I)), which is a safe default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reverse_kl_gaussian(
    mu_q: Tensor,
    log_std_q: Tensor,
    mu_p: Tensor,
    log_std_p: Tensor,
) -> Tensor:
    """
    Analytical KL(q || p) for two diagonal Gaussians, per sample.

    KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))
        = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²) / (2 σ_p²) - 0.5

    Args:
        mu_q, log_std_q:  (B, D) — encoder posterior
        mu_p, log_std_p:  (D,) or (B, D) — source distribution

    Returns:
        kl: (B,) — per-sample KL divergence (summed over dimensions)
    """
    std_q = log_std_q.exp()
    std_p = log_std_p.exp()
    var_q = std_q.pow(2)
    var_p = std_p.pow(2)

    kl = (
        log_std_p - log_std_q
        + (var_q + (mu_q - mu_p).pow(2)) / (2.0 * var_p)
        - 0.5
    )
    return kl.sum(dim=-1)   # (B,)


def vae_loss(
    recon: Tensor,
    target: Tensor,
    mu_q: Tensor,
    log_std_q: Tensor,
    source_mean: Tensor,
    source_std: Tensor,
    beta: float = 1.0,
    recon_weight: float = 1.0,
) -> dict[str, Tensor]:
    """
    Asymmetric VAE loss = recon_loss + β * KL(q || p_s).

    Args:
        recon:       (B, T, action_dim)  — VAE reconstruction
        target:      (B, T, action_dim)  — ground-truth target actions
        mu_q:        (B, latent_dim)     — encoder posterior mean
        log_std_q:   (B, latent_dim)     — encoder posterior log-std
        source_mean: (latent_dim,)       — source distribution mean
        source_std:  (latent_dim,)       — source distribution std  (> 0)
        beta:        KL weight
        recon_weight: reconstruction loss weight

    Returns:
        dict with keys: total, recon, kl   (all scalar tensors)
    """
    # Reconstruction loss (MSE over action chunk)
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # Reverse KL: KL(q(z|x) || p_s(z))
    log_std_p = source_std.clamp(min=1e-6).log().to(mu_q.device)
    mu_p = source_mean.to(mu_q.device)

    kl = reverse_kl_gaussian(mu_q, log_std_q, mu_p, log_std_p)
    kl_loss = kl.mean()

    total = recon_weight * recon_loss + beta * kl_loss
    return {"total": total, "recon": recon_loss, "kl": kl_loss}
