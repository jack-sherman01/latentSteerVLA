"""
SteeringLibrary — stores and manages steering vectors v_ij for all
(embodiment_i, task_j) training pairs.

After extracting steering vectors from trained VAEs (script 03), they are
collected here as a matrix V ∈ ℝ^{N_pairs × latent_dim}, ready for
factorization into Δe + Δt components.

File layout on disk:
    steering_library/
        vectors/
            panda__pick_cube.pt          # single steering vector (latent_dim,)
            panda__stack_cube.pt
            ...
        library.pt                       # full matrix V + metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class SteeringLibrary:
    """
    Container for a collection of steering vectors.

    Attributes:
        vectors:       Dict mapping (embodiment_id, task_id) → Δv (latent_dim,)
        latent_dim:    Dimensionality of steering vectors
        embodiment_ids: Ordered list of embodiment identifiers
        task_ids:      Ordered list of task identifiers
    """

    vectors: dict[tuple[str, str], Tensor] = field(default_factory=dict)
    latent_dim: int | None = None
    embodiment_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)

    # ── Add / get ────────────────────────────────────────────────────────
    def add(self, embodiment_id: str, task_id: str, vector: Tensor) -> None:
        """Register a steering vector for a (embodiment, task) pair."""
        if self.latent_dim is None:
            self.latent_dim = vector.shape[0]
        else:
            assert vector.shape[0] == self.latent_dim, (
                f"Vector dim mismatch: expected {self.latent_dim}, got {vector.shape[0]}"
            )
        key = (embodiment_id, task_id)
        self.vectors[key] = vector.cpu().float()
        if embodiment_id not in self.embodiment_ids:
            self.embodiment_ids.append(embodiment_id)
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)

    def get(self, embodiment_id: str, task_id: str) -> Tensor:
        """Retrieve steering vector for (embodiment, task)."""
        key = (embodiment_id, task_id)
        if key not in self.vectors:
            raise KeyError(f"No steering vector for ({embodiment_id}, {task_id})")
        return self.vectors[key]

    # ── Matrix form ──────────────────────────────────────────────────────
    def to_matrix(self) -> tuple[Tensor, list[tuple[str, str]]]:
        """
        Assemble the steering matrix V ∈ ℝ^{N_pairs × latent_dim}.

        Returns:
            V:     (N_pairs, latent_dim)
            keys:  List of (embodiment_id, task_id) tuples, row order
        """
        keys = sorted(self.vectors.keys())
        V = torch.stack([self.vectors[k] for k in keys], dim=0)   # (N, D)
        return V, keys

    def num_pairs(self) -> int:
        return len(self.vectors)

    # ── I/O ──────────────────────────────────────────────────────────────
    def save(self, root: str | Path) -> None:
        """Save library to disk."""
        root = Path(root)
        vec_dir = root / "vectors"
        vec_dir.mkdir(parents=True, exist_ok=True)

        # Save individual vectors
        for (emb, task), vec in self.vectors.items():
            fname = vec_dir / f"{emb}__{task}.pt"
            torch.save(vec, fname)

        # Save metadata + full matrix
        V, keys = self.to_matrix()
        torch.save(
            {
                "V": V,
                "keys": keys,
                "latent_dim": self.latent_dim,
                "embodiment_ids": self.embodiment_ids,
                "task_ids": self.task_ids,
            },
            root / "library.pt",
        )

        # Human-readable index
        index = {f"{e}__{t}": f"vectors/{e}__{t}.pt" for (e, t) in keys}
        with open(root / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        print(f"Saved steering library ({len(keys)} pairs) → {root}")

    @classmethod
    def load(cls, root: str | Path) -> "SteeringLibrary":
        """Load a previously saved library."""
        root = Path(root)
        ckpt = torch.load(root / "library.pt", map_location="cpu")

        lib = cls(
            latent_dim=ckpt["latent_dim"],
            embodiment_ids=ckpt["embodiment_ids"],
            task_ids=ckpt["task_ids"],
        )
        for (emb, task), vec in zip(ckpt["keys"], ckpt["V"]):
            lib.vectors[(emb, task)] = vec

        return lib

    def __repr__(self) -> str:
        return (
            f"SteeringLibrary("
            f"{self.num_pairs()} pairs, "
            f"latent_dim={self.latent_dim}, "
            f"embodiments={self.embodiment_ids}, "
            f"tasks={self.task_ids})"
        )
