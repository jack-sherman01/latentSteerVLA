"""
VAE training loop — Phase 1 of CompSteer.

For each (embodiment, task) pair, this trains the AsymmetricVAE to align
target robot action chunks into the pre-trained GR00T action latent distribution.

Usage (via script):
    python scripts/02_train_vae.py embodiment=panda task=pick_cube

or directly:
    from compsteer.vae.train import train_vae
    vae = train_vae(cfg, target_actions, source_latent_stats)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import AsymmetricVAE
from .losses import vae_loss


def train_vae(
    target_actions: Tensor,
    action_dim: int,
    action_horizon: int,
    latent_dim: int = 256,
    hidden_dims: list[int] | None = None,
    source_mean: Tensor | None = None,
    source_std: Tensor | None = None,
    beta: float = 1.0,
    recon_weight: float = 1.0,
    lr: float = 3e-4,
    batch_size: int = 128,
    num_epochs: int = 100,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    device: str = "cuda",
    save_path: str | Path | None = None,
    log_interval: int = 10,
) -> AsymmetricVAE:
    """
    Train an AsymmetricVAE on target action chunks.

    Args:
        target_actions:  (N, T, action_dim)  — target robot demo action chunks
        action_dim:      DoF of target robot
        action_horizon:  T — chunk length
        latent_dim:      VAE latent dimension z
        hidden_dims:     MLP hidden layer sizes
        source_mean:     (latent_dim,) — source distribution mean; if None, use N(0,I)
        source_std:      (latent_dim,) — source distribution std; if None, use N(0,I)
        beta:            KL weight
        recon_weight:    Reconstruction loss weight
        lr:              Learning rate
        batch_size:      Mini-batch size
        num_epochs:      Training epochs
        warmup_steps:    Linear LR warmup steps
        grad_clip:       Gradient norm clip value
        device:          'cuda' or 'cpu'
        save_path:       If provided, save final checkpoint here
        log_interval:    Print loss every N epochs

    Returns:
        Trained AsymmetricVAE (on CPU, eval mode)
    """
    if hidden_dims is None:
        hidden_dims = [512, 512]

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────
    vae = AsymmetricVAE(
        action_dim=action_dim,
        action_horizon=action_horizon,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        source_mean=source_mean,
        source_std=source_std,
    ).to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    # target_actions: (N, T, action_dim)  →  ensure float32
    target_actions = target_actions.float()
    dataset = TensorDataset(target_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = num_epochs * len(loader)
    scheduler = _build_cosine_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────
    vae.train()
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        epoch_losses: dict[str, float] = {"total": 0.0, "recon": 0.0, "kl": 0.0}

        for (batch,) in loader:
            batch = batch.to(device)    # (B, T, action_dim)

            out = vae(batch)
            losses = vae_loss(
                recon=out["recon"],
                target=batch,
                mu_q=out["mu"],
                log_std_q=out["log_std"],
                source_mean=vae.source_mean,
                source_std=vae.source_std,
                beta=beta,
                recon_weight=recon_weight,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            for k, v in losses.items():
                epoch_losses[k] += v.item()

        if epoch % log_interval == 0:
            n = len(loader)
            print(
                f"Epoch {epoch:4d}/{num_epochs}  "
                f"total={epoch_losses['total']/n:.4f}  "
                f"recon={epoch_losses['recon']/n:.4f}  "
                f"kl={epoch_losses['kl']/n:.4f}"
            )

    # ── Save checkpoint ───────────────────────────────────────────────────
    vae.eval()
    vae = vae.cpu()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": vae.state_dict(),
                "action_dim": action_dim,
                "action_horizon": action_horizon,
                "latent_dim": latent_dim,
                "hidden_dims": hidden_dims,
            },
            save_path,
        )
        print(f"Saved VAE checkpoint → {save_path}")

    return vae


def load_vae(checkpoint_path: str | Path, device: str = "cpu") -> AsymmetricVAE:
    """Load a saved AsymmetricVAE checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    vae = AsymmetricVAE(
        action_dim=ckpt["action_dim"],
        action_horizon=ckpt["action_horizon"],
        latent_dim=ckpt["latent_dim"],
        hidden_dims=ckpt["hidden_dims"],
    )
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    return vae


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
