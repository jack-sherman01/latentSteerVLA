"""
Robot embodiment specification vectors for the f_emb encoder.

Each robot is described by a fixed-length vector encoding:
  - Joint limits (low + high), padded to MAX_DOF dimensions each
  - Gripper type one-hot
  - Max reach (normalised)
  - Log-normalised payload
  - Normalised DoF count

These vectors are the input to EmbodimentEncoder (f_emb) for generalising
to novel robot embodiments not in the steering library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import Tensor

MAX_DOF = 10          # Pad all joint limit arrays to this length
GRIPPER_TYPES = ["parallel_jaw", "robotiq_85", "robotiq_140", "suction", "dexterous"]
MAX_REACH_M = 2.0     # Normalisation constant for reach
MAX_PAYLOAD_KG = 50.0 # Normalisation constant for payload


@dataclass
class EmbodimentSpec:
    """
    All relevant geometric/physical properties of a robot arm.

    Args:
        name:             Unique identifier (must match embodiments.yaml)
        dof:              Number of controllable joints
        joint_limits_low: List of lower joint limits (radians), length == dof
        joint_limits_high: List of upper joint limits (radians), length == dof
        gripper_type:     One of GRIPPER_TYPES
        max_reach_m:      Maximum end-effector reach in metres
        payload_kg:       Maximum payload in kilograms
    """
    name: str
    dof: int
    joint_limits_low: list[float]
    joint_limits_high: list[float]
    gripper_type: str
    max_reach_m: float
    payload_kg: float

    # ── Vector representation ────────────────────────────────────────────
    def to_vector(self) -> Tensor:
        """
        Convert to a fixed-length float32 vector for f_emb.

        Dimensions:
            [0 : MAX_DOF]           — normalised joint limits low  (padded)
            [MAX_DOF : 2*MAX_DOF]   — normalised joint limits high (padded)
            [2*MAX_DOF : 2*MAX_DOF + len(GRIPPER_TYPES)]  — gripper one-hot
            [-3]                    — normalised max reach
            [-2]                    — log-normalised payload
            [-1]                    — normalised DoF count

        Total: 2*MAX_DOF + len(GRIPPER_TYPES) + 3
        """
        import math

        # Joint limits — padded to MAX_DOF, normalised by π
        def _pad(lst: list[float]) -> list[float]:
            normed = [v / math.pi for v in lst]
            padded = normed + [0.0] * (MAX_DOF - len(normed))
            return padded[:MAX_DOF]

        low_pad = _pad(self.joint_limits_low)
        high_pad = _pad(self.joint_limits_high)

        # Gripper one-hot
        gripper_vec = [0.0] * len(GRIPPER_TYPES)
        if self.gripper_type in GRIPPER_TYPES:
            gripper_vec[GRIPPER_TYPES.index(self.gripper_type)] = 1.0

        # Scalar features
        reach_norm = self.max_reach_m / MAX_REACH_M
        payload_log = math.log1p(self.payload_kg) / math.log1p(MAX_PAYLOAD_KG)
        dof_norm = self.dof / MAX_DOF

        vec = low_pad + high_pad + gripper_vec + [reach_norm, payload_log, dof_norm]
        return torch.tensor(vec, dtype=torch.float32)

    @staticmethod
    def vector_dim() -> int:
        return 2 * MAX_DOF + len(GRIPPER_TYPES) + 3


def load_embodiment_specs(config_path: str | Path) -> dict[str, EmbodimentSpec]:
    """
    Load all embodiment specs from configs/embodiments.yaml.

    Returns:
        Dict mapping embodiment_id → EmbodimentSpec
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    specs: dict[str, EmbodimentSpec] = {}
    for emb_id, props in cfg["embodiments"].items():
        spec_data = props.get("spec", {})
        specs[emb_id] = EmbodimentSpec(
            name=emb_id,
            dof=props["dof"],
            joint_limits_low=spec_data.get("joint_limits_low", [0.0] * props["dof"]),
            joint_limits_high=spec_data.get("joint_limits_high", [1.0] * props["dof"]),
            gripper_type=props.get("gripper_type", "parallel_jaw"),
            max_reach_m=spec_data.get("max_reach_m", 1.0),
            payload_kg=spec_data.get("payload_kg", 1.0),
        )

    return specs


def build_spec_matrix(
    specs: dict[str, EmbodimentSpec],
    embodiment_ids: list[str],
) -> Tensor:
    """
    Build a (N_e, vector_dim) matrix of spec vectors.

    Args:
        specs:          Dict from load_embodiment_specs()
        embodiment_ids: Ordered list of embodiment identifiers

    Returns:
        X: (N_e, vector_dim) float32 tensor
    """
    rows = [specs[emb_id].to_vector() for emb_id in embodiment_ids]
    return torch.stack(rows, dim=0)
