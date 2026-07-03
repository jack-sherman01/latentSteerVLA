"""
Lightweight retrieval encoders for generalising to novel embodiments/tasks
not in the steering library.

  f_emb : robot_spec → Δe   (embodiment component, ℝ^k)
  g_lang : task_text → Δt   (task component, ℝ^k)

These are trained (script 05) on the library entries so they can interpolate
or extrapolate to unseen (robot, task) pairs at inference time.
When library lookup suffices (seen embodiment / seen task), these encoders
are skipped and the stored Δe/Δt vectors are used directly.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


# ── EmbodimentEncoder (f_emb) ─────────────────────────────────────────────────

class EmbodimentEncoder(nn.Module):
    """
    Maps a fixed-length robot specification vector to an embodiment component
    vector Δe ∈ ℝ^k.

    The robot spec encodes:
        - Normalised joint limits (low + high) — 2 * max_dof floats
        - Gripper type one-hot                 — n_gripper_types floats
        - Max reach (metres, normalised)       — 1 float
        - Payload (kg, log-normalised)         — 1 float
        - DoF count (normalised)               — 1 float
    Total: see EmbodimentSpec.to_vector() in data/embodiment_specs.py

    Args:
        input_dim:  Dimensionality of the robot spec vector
        hidden_dims: Hidden layer sizes
        output_dim:  k — must match factorization rank
        dropout:    Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list[int] | None = None,
        output_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, robot_spec: Tensor) -> Tensor:
        """
        Args:
            robot_spec: (B, input_dim) or (input_dim,)
        Returns:
            delta_e: (B, output_dim) or (output_dim,)
        """
        return self.net(robot_spec)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self._config()}, path)

    def _config(self) -> dict:
        first = self.net[0]
        last = self.net[-1]
        return {"input_dim": first.in_features, "output_dim": last.out_features}

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "EmbodimentEncoder":
        ckpt = torch.load(path, map_location="cpu")
        cfg = {**ckpt["config"], **kwargs}
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ── TaskEncoder (g_lang) ──────────────────────────────────────────────────────

class TaskEncoder(nn.Module):
    """
    Maps a natural-language task description to a task component vector
    Δt ∈ ℝ^k by projecting frozen T5 embeddings via a learned linear layer.

    Architecture:
        frozen T5-base encoder  →  mean-pool over tokens  →  linear(768, k)

    Training only updates the linear projection.

    Args:
        t5_model:    HuggingFace model ID for T5 (default 't5-base')
        t5_hidden:   T5 hidden size (768 for t5-base)
        output_dim:  k — must match factorization rank
        freeze_t5:   If True, T5 weights are frozen (recommended)
    """

    def __init__(
        self,
        t5_model: str = "t5-base",
        t5_hidden: int = 768,
        output_dim: int = 16,
        freeze_t5: bool = True,
    ):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(t5_model)
        self.t5 = T5EncoderModel.from_pretrained(t5_model)
        self.t5_hidden = t5_hidden
        self.output_dim = output_dim
        self.freeze_t5 = freeze_t5

        if freeze_t5:
            for p in self.t5.parameters():
                p.requires_grad_(False)

        self.proj = nn.Linear(t5_hidden, output_dim)

    def encode_text(self, texts: list[str], device: torch.device) -> Tensor:
        """Run T5 encoder and mean-pool token representations."""
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.set_grad_enabled(not self.freeze_t5):
            outputs = self.t5(**tokens)
        # Mean-pool over sequence tokens (ignoring padding)
        mask = tokens["attention_mask"].unsqueeze(-1).float()   # (B, L, 1)
        hidden = outputs.last_hidden_state                       # (B, L, D)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)   # (B, D)
        return pooled

    def forward(self, texts: list[str]) -> Tensor:
        """
        Args:
            texts: list of B task descriptions

        Returns:
            delta_t: (B, output_dim)
        """
        device = self.proj.weight.device
        pooled = self.encode_text(texts, device)   # (B, t5_hidden)
        return self.proj(pooled)                   # (B, output_dim)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"proj_state_dict": self.proj.state_dict(), "output_dim": self.output_dim}, path)

    def load_proj(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.proj.load_state_dict(ckpt["proj_state_dict"])


# ── Training utilities ────────────────────────────────────────────────────────

def train_embodiment_encoder(
    encoder: EmbodimentEncoder,
    spec_vectors: Tensor,         # (N_e, input_dim)
    target_e_vectors: Tensor,     # (N_e, k)  — from FactorizationResult.E
    lr: float = 1e-3,
    epochs: int = 200,
    device: str = "cuda",
    save_path: str | Path | None = None,
) -> EmbodimentEncoder:
    """
    Train f_emb on (robot_spec, Δe) pairs.

    Args:
        encoder:          EmbodimentEncoder instance
        spec_vectors:     (N_e, input_dim) — robot spec vectors
        target_e_vectors: (N_e, k) — ground-truth Δe from factorization
        lr:               Learning rate
        epochs:           Training epochs
        device:           'cuda' or 'cpu'
        save_path:        Optional checkpoint save path

    Returns:
        Trained EmbodimentEncoder (eval mode, on CPU)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    X = spec_vectors.float().to(device)
    Y = target_e_vectors.float().to(device)

    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)

    encoder.train()
    for epoch in range(1, epochs + 1):
        pred = encoder(X)
        loss = (pred - Y).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  f_emb epoch {epoch}/{epochs}  mse={loss.item():.5f}")

    encoder.eval()
    encoder = encoder.cpu()

    if save_path is not None:
        encoder.save(save_path)
        print(f"Saved EmbodimentEncoder → {save_path}")

    return encoder


def train_task_encoder(
    encoder: TaskEncoder,
    task_descriptions: list[str],  # N_t strings
    target_t_vectors: Tensor,       # (N_t, k) — from FactorizationResult.T
    lr: float = 5e-4,
    epochs: int = 100,
    batch_size: int = 16,
    device: str = "cuda",
    save_path: str | Path | None = None,
) -> TaskEncoder:
    """
    Train g_lang on (task_description, Δt) pairs.

    Args:
        encoder:            TaskEncoder instance
        task_descriptions:  List of N_t task description strings
        target_t_vectors:   (N_t, k) — ground-truth Δt from factorization
        lr:                 Learning rate (only updates T5 projection layer)
        epochs:             Training epochs
        batch_size:         Mini-batch size
        device:             'cuda' or 'cpu'
        save_path:          Optional checkpoint save path

    Returns:
        Trained TaskEncoder (eval mode, on CPU)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    Y = target_t_vectors.float().to(device)

    # Only train the projection layer
    optimizer = optim.AdamW(encoder.proj.parameters(), lr=lr, weight_decay=1e-4)
    N = len(task_descriptions)

    encoder.train()
    for epoch in range(1, epochs + 1):
        # Mini-batch over tasks
        perm = torch.randperm(N)
        epoch_loss = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size].tolist()
            texts_batch = [task_descriptions[i] for i in idx]
            y_batch = Y[idx]

            pred = encoder(texts_batch)
            loss = (pred - y_batch).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"  g_lang epoch {epoch}/{epochs}  mse={epoch_loss / (N / batch_size):.5f}")

    encoder.eval()
    encoder = encoder.cpu()

    if save_path is not None:
        encoder.save(save_path)
        print(f"Saved TaskEncoder → {save_path}")

    return encoder
