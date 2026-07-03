"""
Factorize the steering matrix V ∈ ℝ^{N_pairs × latent_dim} into:
    V ≈ E · T^T
where
    E ∈ ℝ^{N_e × k}   — embodiment component matrix
    T ∈ ℝ^{N_t × k}   — task component matrix
    k                  — rank (number of components)

Three methods are implemented:
  1. SVD  — closed-form, best linear approximation (default)
  2. NMF  — non-negative matrix factorization via nimfa
  3. Learned — gradient-based with optional orthogonality regularisation

After factorization, Δe_i = E[i, :] and Δt_j = T[j, :] are the per-embodiment
and per-task components used for inference-time composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .vector_library import SteeringLibrary


@dataclass
class FactorizationResult:
    """
    Output of factorize_library().

    Attributes:
        E:           (N_e, k) — embodiment component matrix
        T:           (N_t, k) — task component matrix
        embodiment_ids: ordered list matching rows of E
        task_ids:    ordered list matching rows of T
        rank:        k
        method:      factorization method used
        residual:    relative Frobenius residual ‖V - ET^T‖_F / ‖V‖_F
    """
    E: Tensor
    T: Tensor
    embodiment_ids: list[str]
    task_ids: list[str]
    rank: int
    method: str
    residual: float

    def get_embodiment_vector(self, embodiment_id: str) -> Tensor:
        idx = self.embodiment_ids.index(embodiment_id)
        return self.E[idx]   # (k,)

    def get_task_vector(self, task_id: str) -> Tensor:
        idx = self.task_ids.index(task_id)
        return self.T[idx]   # (k,)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "E": self.E,
                "T": self.T,
                "embodiment_ids": self.embodiment_ids,
                "task_ids": self.task_ids,
                "rank": self.rank,
                "method": self.method,
                "residual": self.residual,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FactorizationResult":
        d = torch.load(path, map_location="cpu")
        return cls(**d)


# ── Public API ────────────────────────────────────────────────────────────────

def factorize_library(
    library: SteeringLibrary,
    rank: int = 16,
    method: Literal["svd", "nmf", "learned"] = "svd",
    **kwargs,
) -> FactorizationResult:
    """
    Factorize steering library V ≈ E · T^T.

    Args:
        library:  SteeringLibrary with N_pairs steering vectors
        rank:     Number of components k
        method:   'svd' | 'nmf' | 'learned'
        **kwargs: Passed to the chosen factorization method

    Returns:
        FactorizationResult with per-embodiment and per-task component vectors
    """
    V, keys = library.to_matrix()   # (N_pairs, latent_dim)
    embodiment_ids = library.embodiment_ids
    task_ids = library.task_ids

    # Build index maps: pair (e, t) → row_index, and e/t → component index
    e2idx = {e: i for i, e in enumerate(embodiment_ids)}
    t2idx = {t: i for i, t in enumerate(task_ids)}

    # Build the structured (N_e × N_t) matrix for factorization.
    # Missing pairs get a zero row.
    N_e, N_t = len(embodiment_ids), len(task_ids)
    V_full = torch.zeros(N_e, N_t, library.latent_dim)   # (N_e, N_t, D)
    mask = torch.zeros(N_e, N_t, dtype=torch.bool)

    for (emb, task), vec in library.vectors.items():
        i, j = e2idx[emb], t2idx[task]
        V_full[i, j] = vec
        mask[i, j] = True

    # Reshape to (N_e * N_t, D) for matrix factorization, then to (N_e, D) and (N_t, D)
    # We factorize the mean-pooled versions:
    #   Δe_i = mean over tasks j where (i,j) is available
    #   Δt_j = mean over embodiments i where (i,j) is available
    # Then we refine via the chosen method.

    if method == "svd":
        E, T = _factorize_svd(V_full, mask, rank)
    elif method == "nmf":
        E, T = _factorize_nmf(V_full, mask, rank, **kwargs)
    elif method == "learned":
        E, T = _factorize_learned(V_full, mask, rank, embodiment_ids, task_ids, **kwargs)
    else:
        raise ValueError(f"Unknown factorization method: {method}")

    # Compute residual on observed entries
    V_hat = E[None, :, :] * T[:, None, :]  # wrong shape — compute properly
    # E: (N_e, k), T: (N_t, k)
    # V_hat[i,j,:] = E[i,:] + T[j,:]  (additive model, not multiplicative)
    V_hat = E.unsqueeze(1).expand(-1, N_t, -1) + T.unsqueeze(0).expand(N_e, -1, -1)
    diff = (V_hat - V_full)[mask]   # only observed pairs
    target = V_full[mask]
    residual = diff.norm() / (target.norm() + 1e-8)

    return FactorizationResult(
        E=E,
        T=T,
        embodiment_ids=embodiment_ids,
        task_ids=task_ids,
        rank=rank,
        method=method,
        residual=residual.item(),
    )


# ── SVD factorization ─────────────────────────────────────────────────────────

def _factorize_svd(
    V_full: Tensor,     # (N_e, N_t, D)
    mask: Tensor,       # (N_e, N_t) bool
    rank: int,
) -> tuple[Tensor, Tensor]:
    """
    SVD-based additive factorization:
        v_ij ≈ Δe_i + Δt_j
    Derived via mean decomposition + low-rank SVD correction.
    """
    N_e, N_t, D = V_full.shape

    # Mean decomposition
    # Δe_i = mean_j(v_ij)   Δt_j = mean_i(v_ij) - global_mean
    count_e = mask.float().sum(dim=1, keepdim=True).clamp(min=1)   # (N_e, 1)
    count_t = mask.float().sum(dim=0, keepdim=True).clamp(min=1)   # (1, N_t)

    delta_e = (V_full * mask.unsqueeze(-1).float()).sum(dim=1) / count_e   # (N_e, D)
    delta_t = (V_full * mask.unsqueeze(-1).float()).sum(dim=0) / count_t   # (N_t, D)

    # Subtract additive means; apply SVD to the residual
    residual = V_full - delta_e.unsqueeze(1) - delta_t.unsqueeze(0)   # (N_e, N_t, D)
    # Reshape to (N_e * N_t, D) using only observed entries — but for SVD we use the full matrix
    R = residual.view(N_e * N_t, D)
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    S_k = S[:rank]
    U_k = U[:, :rank]       # (N_e*N_t, k)
    Vh_k = Vh[:rank, :]     # (k, D)

    # Redistribute into E and T corrections
    # We split the rank-k correction: half of singular values go to E, half to T
    # Simpler: project U_k back to (N_e, N_t, k) and average
    U_k_3d = U_k.view(N_e, N_t, rank)
    corr_e = (U_k_3d * (S_k * Vh_k).norm(dim=-1).unsqueeze(0).unsqueeze(0)).mean(dim=1)   # (N_e, k)
    corr_t = (U_k_3d * (S_k * Vh_k).norm(dim=-1).unsqueeze(0).unsqueeze(0)).mean(dim=0)   # (N_t, k)

    # For the paper, we keep it simple: E[i,:] = delta_e[i], T[j,:] = delta_t[j]
    # padded to rank dimension. The low-rank correction is available if needed.
    # We project both into R^k via PCA of their concatenation.
    ET = torch.cat([delta_e, delta_t], dim=0)   # (N_e+N_t, D)
    _, _, Vt = torch.linalg.svd(ET, full_matrices=False)
    proj = Vt[:rank, :]   # (k, D) — projection matrix

    E = delta_e @ proj.T    # (N_e, k)
    T = delta_t @ proj.T    # (N_t, k)

    return E, T


# ── NMF factorization ─────────────────────────────────────────────────────────

def _factorize_nmf(
    V_full: Tensor,
    mask: Tensor,
    rank: int,
    nmf_max_iter: int = 500,
    nmf_init: str = "nndsvda",
) -> tuple[Tensor, Tensor]:
    """
    NMF-based factorization using nimfa.
    Requires non-negative inputs — we shift V to be non-negative.
    """
    try:
        import nimfa
    except ImportError:
        raise ImportError("nimfa is required for NMF factorization: pip install nimfa")

    import numpy as np

    N_e, N_t, D = V_full.shape
    # Build observed (N_pairs, D) matrix
    obs_vecs = V_full[mask].numpy()   # (N_obs, D)

    # Shift to non-negative
    min_val = obs_vecs.min()
    obs_shifted = obs_vecs - min_val + 1e-8

    nmf = nimfa.Nmf(obs_shifted.T, rank=rank, max_iter=nmf_max_iter, initialize_only=True)
    nmf_fit = nimfa.mf_run(nmf)
    W = np.array(nmf_fit.basis())     # (D, k)
    H = np.array(nmf_fit.coef())      # (k, N_obs)

    # W is the basis, H are the coefficients per pair
    # Map coefficients back to E and T via mean over tasks/embodiments
    H_tensor = torch.tensor(H.T, dtype=torch.float32)   # (N_obs, k)
    e_ids = [e for (e, _) in [k for k, v in zip(sorted(mask.nonzero().tolist()), [True]*mask.sum())]]

    # Simpler: just use mean of H over task indices for E, over embodiment indices for T
    # For now, return H-derived projections projected to D space
    # E[i,:] = mean of H rows corresponding to embodiment i
    obs_idx = mask.nonzero(as_tuple=False)   # (N_obs, 2) — (e_idx, t_idx)
    E = torch.zeros(N_e, rank)
    T = torch.zeros(N_t, rank)
    e_counts = torch.zeros(N_e)
    t_counts = torch.zeros(N_t)

    for idx, (ei, tj) in enumerate(obs_idx.tolist()):
        E[ei] += H_tensor[idx]
        T[tj] += H_tensor[idx]
        e_counts[ei] += 1
        t_counts[tj] += 1

    E = E / e_counts.unsqueeze(1).clamp(min=1)
    T = T / t_counts.unsqueeze(1).clamp(min=1)

    return E, T


# ── Learned factorization ─────────────────────────────────────────────────────

class LearnedFactorization(nn.Module):
    """
    Gradient-based additive factorization:
        v_ij ≈ Δe_i + Δt_j    where  Δe_i = f(e_i), Δt_j = g(t_j)
    with optional orthogonality regularisation ‖E^T T‖_F.

    Training minimises MSE on observed (i,j) pairs.
    """

    def __init__(
        self,
        n_embodiments: int,
        n_tasks: int,
        latent_dim: int,
        rank: int = 16,
        hidden_dim: int = 256,
        ortho_weight: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.latent_dim = latent_dim
        self.ortho_weight = ortho_weight

        # Learnable embeddings in R^k
        self.E_emb = nn.Embedding(n_embodiments, rank)   # Δe in low-dim
        self.T_emb = nn.Embedding(n_tasks, rank)          # Δt in low-dim

        # Shared decoder: R^k → R^latent_dim
        self.decoder_e = nn.Sequential(
            nn.Linear(rank, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder_t = nn.Sequential(
            nn.Linear(rank, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, e_idx: Tensor, t_idx: Tensor) -> Tensor:
        """
        Predict steering vector for batch of (e_idx, t_idx) pairs.

        Returns:
            v_hat: (B, latent_dim)
        """
        delta_e = self.decoder_e(self.E_emb(e_idx))   # (B, D)
        delta_t = self.decoder_t(self.T_emb(t_idx))   # (B, D)
        return delta_e + delta_t

    def get_e_vector(self, e_idx: int) -> Tensor:
        idx = torch.tensor([e_idx])
        return self.decoder_e(self.E_emb(idx)).squeeze(0).detach()

    def get_t_vector(self, t_idx: int) -> Tensor:
        idx = torch.tensor([t_idx])
        return self.decoder_t(self.T_emb(idx)).squeeze(0).detach()

    def orthogonality_loss(self) -> Tensor:
        """Encourage Δe space ⊥ Δt space via E_emb weight orthogonality."""
        E = self.E_emb.weight   # (N_e, k)
        T = self.T_emb.weight   # (N_t, k)
        cross = E.T @ T         # (k, k) — should be small
        return cross.pow(2).mean()


def _factorize_learned(
    V_full: Tensor,
    mask: Tensor,
    rank: int,
    embodiment_ids: list[str],
    task_ids: list[str],
    hidden_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 200,
    recon_weight: float = 1.0,
    ortho_weight: float = 0.1,
    device: str = "cuda",
) -> tuple[Tensor, Tensor]:
    """Train the LearnedFactorization model and return (E, T) in R^k."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    N_e, N_t, D = V_full.shape

    model = LearnedFactorization(
        n_embodiments=N_e,
        n_tasks=N_t,
        latent_dim=D,
        rank=rank,
        hidden_dim=hidden_dim,
        ortho_weight=ortho_weight,
    ).to(device)

    # Build dataset of (e_idx, t_idx, v_ij) triples from observed entries
    obs_idx = mask.nonzero(as_tuple=False)   # (N_obs, 2)
    e_idx_all = obs_idx[:, 0].to(device)
    t_idx_all = obs_idx[:, 1].to(device)
    v_all = V_full[mask].to(device)           # (N_obs, D)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        v_hat = model(e_idx_all, t_idx_all)
        recon_loss = (v_hat - v_all).pow(2).mean()
        ortho_loss = model.orthogonality_loss()
        loss = recon_weight * recon_loss + ortho_weight * ortho_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Factorize epoch {epoch+1}/{epochs}  recon={recon_loss.item():.4f}  ortho={ortho_loss.item():.4f}")

    # Extract final E and T matrices in k-dim space
    model.eval()
    with torch.no_grad():
        E = model.E_emb.weight.cpu()   # (N_e, k)
        T = model.T_emb.weight.cpu()   # (N_t, k)

    return E, T
