"""
Inference-time composition of embodiment and task steering components.

    Δz = α · Δe + β · Δt

where
    Δe ∈ ℝ^k — embodiment component (from library or f_emb encoder)
    Δt ∈ ℝ^k — task component       (from library or g_lang encoder)
    α, β      — scalar weights (fixed or adaptive)

The composed vector Δz is then injected into the GR00T denoising loop
by SteeringSchedule, which scales it at each denoising timestep t via λ(t).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .vector_library import SteeringLibrary
from .factorize import FactorizationResult


class ScheduleType(str, Enum):
    CONSTANT = "constant"
    COSINE = "cosine"
    LINEAR = "linear"
    EARLY_ONLY = "early_only"


class SteeringSchedule:
    """
    Step-dependent injection weight λ(t) for denoising step t ∈ [0, 1].

    In rectified flow (GR00T), t=1 is pure noise, t=0 is clean action.
    Larger λ at early steps (high t) means stronger steering when the
    action is still noisy; smaller λ at late steps preserves fine details.
    """

    def __init__(
        self,
        schedule: str | ScheduleType = "cosine",
        peak: float = 1.0,
        early_only_cutoff: float = 0.5,
    ):
        self.schedule = ScheduleType(schedule)
        self.peak = peak
        self.early_only_cutoff = early_only_cutoff

    def __call__(self, t: float) -> float:
        """
        Args:
            t: current denoising timestep ∈ [0, 1]  (1=noise, 0=clean)
        Returns:
            λ: injection weight ≥ 0
        """
        import math

        p = self.peak
        if self.schedule == ScheduleType.CONSTANT:
            return p
        elif self.schedule == ScheduleType.COSINE:
            # Peaks at t=1 (pure noise), decays to 0 at t=0 (clean)
            return p * 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))
        elif self.schedule == ScheduleType.LINEAR:
            return p * t
        elif self.schedule == ScheduleType.EARLY_ONLY:
            # Only inject when t > cutoff (early denoising steps)
            return p if t > self.early_only_cutoff else 0.0
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def as_tensor_fn(self, device: torch.device) -> Callable[[Tensor], Tensor]:
        """Return a version that operates on a (B,) timestep tensor."""
        def fn(t_batch: Tensor) -> Tensor:
            # t_batch: (B,) — normalised timestep in [0, 1]
            lam = torch.zeros_like(t_batch)
            for i, t_val in enumerate(t_batch.tolist()):
                lam[i] = self(float(t_val))
            return lam.to(device)   # (B,)
        return fn


class AdaptiveComposer(nn.Module):
    """
    Learns scalar weights (α, β) per (Δe, Δt) query via a tiny attention module.
    Used when compose.adaptive_weights=True in config.
    """

    def __init__(self, rank: int):
        super().__init__()
        # Takes [Δe; Δt] ∈ ℝ^{2k} → (α, β) ∈ ℝ²  (softplus to keep > 0)
        self.net = nn.Sequential(
            nn.Linear(rank * 2, rank),
            nn.SiLU(),
            nn.Linear(rank, 2),
            nn.Softplus(),
        )

    def forward(self, delta_e: Tensor, delta_t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            delta_e: (k,) or (B, k)
            delta_t: (k,) or (B, k)
        Returns:
            alpha, beta: scalars or (B,)
        """
        inp = torch.cat([delta_e, delta_t], dim=-1)
        weights = self.net(inp)   # (..., 2)
        return weights[..., 0], weights[..., 1]


def compose_steering_vector(
    embodiment_id: str,
    task_id: str,
    factorization: FactorizationResult,
    library: SteeringLibrary | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    adaptive_composer: AdaptiveComposer | None = None,
) -> Tensor:
    """
    Compose a steering vector Δz for a given (embodiment, task) query.

    Lookup order:
      1. If (embodiment_id, task_id) is in library → use stored vector directly
         (split A — seen pair, for sanity check)
      2. Else → compose from factorized Δe + Δt components

    Args:
        embodiment_id:      Target embodiment identifier
        task_id:            Target task identifier
        factorization:      FactorizationResult (contains E, T matrices)
        library:            Optional SteeringLibrary for direct lookup
        alpha:              Embodiment component weight
        beta:               Task component weight
        adaptive_composer:  Optional AdaptiveComposer for learned α, β

    Returns:
        delta_z: (k,) — composed steering vector (in factorization's rank-k space)
    """
    # Check if this exact pair is in the library (split A)
    if library is not None:
        try:
            return library.get(embodiment_id, task_id)
        except KeyError:
            pass

    # Determine which components are available
    e_in_lib = embodiment_id in factorization.embodiment_ids
    t_in_lib = task_id in factorization.task_ids

    if e_in_lib:
        delta_e = factorization.get_embodiment_vector(embodiment_id)
    else:
        raise ValueError(
            f"Embodiment '{embodiment_id}' not in factorization. "
            "Run with --use_f_emb_encoder to use the neural encoder."
        )

    if t_in_lib:
        delta_t = factorization.get_task_vector(task_id)
    else:
        raise ValueError(
            f"Task '{task_id}' not in factorization. "
            "Run with --use_g_lang_encoder to use the language encoder."
        )

    if adaptive_composer is not None:
        alpha, beta = adaptive_composer(delta_e, delta_t)
        delta_z = alpha * delta_e + beta * delta_t
    else:
        delta_z = alpha * delta_e + beta * delta_t

    return delta_z


def compose_from_encoders(
    robot_spec_vector: Tensor,
    task_description: str,
    f_emb: "EmbodimentEncoder",
    g_lang: "TaskEncoder",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    """
    Compose steering vector using neural encoders (for truly unseen pairs).

    Args:
        robot_spec_vector:  (input_dim,) — robot spec embedding
        task_description:   Natural language task description
        f_emb:              Trained EmbodimentEncoder
        g_lang:             Trained TaskEncoder
        alpha, beta:        Component weights

    Returns:
        delta_z: (k,)
    """
    with torch.no_grad():
        delta_e = f_emb(robot_spec_vector.unsqueeze(0)).squeeze(0)    # (k,)
        delta_t = g_lang([task_description]).squeeze(0)                # (k,)
    return alpha * delta_e + beta * delta_t
