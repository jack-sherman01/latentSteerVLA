"""
Inference-time steering injection for GR00T N1 / N1.6 and RDT-1B.

GR00T denoising loop (from gr00t/model/gr00t_n1d6/gr00t_n1d6.py):

    actions = torch.randn(B, T, action_dim)
    for t in linspace(1.0, 0.0, num_inference_steps):
        encoded = action_encoder(actions, timestep=t, embodiment_id=emb_id)
        dit_out  = dit(encoded, encoder_hidden_states=vl_features, timestep=t)
        velocity = action_decoder(dit_out)
        actions  = actions + dt * velocity           # Euler step

CompSteer injection (hidden-space mode):
    encoded_steered = encoded + λ(t) * Δz.unsqueeze(0)   # (B, T, hidden_size)

CompSteer injection (velocity-space mode):
    velocity_steered = velocity + λ(t) * Δz_action.unsqueeze(0)   # (B, T, action_dim)

We implement this by subclassing Gr00tN1d6Pipeline and overriding
get_action_with_features() to intercept the denoising loop.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..steering.compose import SteeringSchedule


# ── GR00T steerable policy ────────────────────────────────────────────────────

class SteerableGr00tPolicy:
    """
    Wraps Gr00tPolicy (from gr00t.eval.service) to inject a steering vector
    into the flow matching denoising loop at inference time.

    No fine-tuning or gradient computation is performed.

    Usage:
        from gr00t.eval.service import Gr00tPolicy
        base_policy = Gr00tPolicy(
            model_path="nvidia/GR00T-N1-2B",
            embodiment_tag="panda",
            ...
        )
        steerable = SteerableGr00tPolicy(base_policy)
        steerable.set_steering_vector(delta_z, schedule="cosine", alpha=1.0)
        action = steerable.get_action(observations)
    """

    def __init__(self, base_policy: Any, injection_mode: str = "hidden"):
        """
        Args:
            base_policy:     gr00t.eval.service.Gr00tPolicy instance
            injection_mode:  'hidden' (action encoder output) or 'velocity' (predicted velocity)
        """
        self.base_policy = base_policy
        self.injection_mode = injection_mode
        self._delta_z: Tensor | None = None
        self._schedule: SteeringSchedule | None = None
        self._hooks: list = []

    # ── Steering vector control ──────────────────────────────────────────
    def set_steering_vector(
        self,
        delta_z: Tensor,
        schedule: str = "cosine",
        schedule_peak: float = 1.0,
        early_only_cutoff: float = 0.5,
        alpha: float = 1.0,
    ) -> None:
        """
        Set the steering vector to inject at inference time.

        Args:
            delta_z:           (hidden_size,) or (action_dim,) depending on injection_mode
            schedule:          λ(t) schedule: 'constant' | 'cosine' | 'linear' | 'early_only'
            schedule_peak:     Maximum λ value
            early_only_cutoff: Cutoff for 'early_only' schedule
            alpha:             Global scaling factor applied to delta_z
        """
        self._delta_z = delta_z.float() * alpha
        self._schedule = SteeringSchedule(
            schedule=schedule,
            peak=schedule_peak,
            early_only_cutoff=early_only_cutoff,
        )
        self._install_hooks()

    def clear_steering(self) -> None:
        """Remove steering vector and hooks."""
        self._delta_z = None
        self._schedule = None
        self._remove_hooks()

    # ── Hook installation ────────────────────────────────────────────────
    def _install_hooks(self) -> None:
        """Register forward hooks on the relevant GR00T sub-module."""
        self._remove_hooks()

        if self._delta_z is None:
            return

        model = self._get_gr00t_model()
        if model is None:
            raise RuntimeError(
                "Could not locate GR00T model from base_policy. "
                "Make sure base_policy has a .model or .gr00t_model attribute."
            )

        if self.injection_mode == "hidden":
            # Hook on action encoder output
            target_module = self._find_module(model, ["action_head.action_encoder",
                                                       "action_encoder",
                                                       "model.action_encoder"])
            if target_module is None:
                raise RuntimeError(
                    "Could not find action_encoder module in GR00T model. "
                    "Check gr00t/model/modules/embodiment_conditioned_mlp.py "
                    "and update the module path in groot_hook.py."
                )
            hook = target_module.register_forward_hook(self._make_hidden_hook())
            self._hooks.append(hook)

        elif self.injection_mode == "velocity":
            # Hook on action decoder output
            target_module = self._find_module(model, ["action_head.action_decoder",
                                                       "action_decoder",
                                                       "model.action_decoder"])
            if target_module is None:
                raise RuntimeError(
                    "Could not find action_decoder module in GR00T model."
                )
            hook = target_module.register_forward_hook(self._make_velocity_hook())
            self._hooks.append(hook)

        else:
            raise ValueError(f"Unknown injection_mode: {self.injection_mode}")

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Hook factories ───────────────────────────────────────────────────
    def _make_hidden_hook(self):
        """
        Forward hook that adds λ(t) · Δz to the action encoder output.

        GR00T's ActionEncoder outputs (B, T, hidden_size).
        The timestep t is passed as an input argument.
        """
        delta_z = self._delta_z
        schedule = self._schedule

        def hook(module, inputs, output):
            # output: (B, T, hidden_size)
            # inputs[1] is typically the timestep tensor (B,) or scalar
            t_val = _extract_timestep(inputs)
            lam = schedule(t_val) if t_val is not None else schedule.peak

            device = output.device
            dz = delta_z.to(device)

            if dz.dim() == 1:
                # (hidden_size,) → broadcast to (1, 1, hidden_size)
                dz = dz.unsqueeze(0).unsqueeze(0)

            return output + lam * dz

        return hook

    def _make_velocity_hook(self):
        """
        Forward hook that adds λ(t) · Δz to the predicted velocity / action_decoder output.

        GR00T's action decoder outputs (B, T, action_dim).
        """
        delta_z = self._delta_z
        schedule = self._schedule

        def hook(module, inputs, output):
            t_val = _extract_timestep(inputs)
            lam = schedule(t_val) if t_val is not None else schedule.peak

            device = output.device
            dz = delta_z.to(device)

            if dz.dim() == 1:
                dz = dz.unsqueeze(0).unsqueeze(0)

            return output + lam * dz

        return hook

    # ── Pass-through to base policy ──────────────────────────────────────
    def get_action(self, observations: dict) -> Any:
        """
        Run inference with steering injected via forward hooks.
        The hooks are active for the duration of this call.
        """
        return self.base_policy.get_action(observations)

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attributes to base policy
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.base_policy, name)

    # ── Internal helpers ─────────────────────────────────────────────────
    def _get_gr00t_model(self) -> Any:
        """Find the underlying nn.Module in the base policy."""
        for attr in ["model", "gr00t_model", "pipeline", "_model"]:
            if hasattr(self.base_policy, attr):
                return getattr(self.base_policy, attr)
        return None

    @staticmethod
    def _find_module(root_model: Any, candidate_paths: list[str]) -> Any:
        """Try multiple dotted attribute paths to find a sub-module."""
        for path in candidate_paths:
            parts = path.split(".")
            obj = root_model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return None


# ── RDT-1B steerable policy ───────────────────────────────────────────────────

class SteerableRDTPolicy:
    """
    Wraps an RDT-1B policy to inject a steering vector into the score-based
    diffusion denoising loop.

    RDT uses DDPM/DDIM-style denoising. The steering vector is injected into
    the noise prediction network output (classifier guidance style):
        noise_pred_steered = noise_pred + λ(t) * Δz_noise

    This follows ATE's original steering mechanism for score-based VLAs.

    Usage:
        from compsteer.injection.groot_hook import SteerableRDTPolicy
        steerable = SteerableRDTPolicy(rdt_policy)
        steerable.set_steering_vector(delta_z, schedule="cosine")
        action = steerable.get_action(observations)
    """

    def __init__(self, base_policy: Any):
        self.base_policy = base_policy
        self._delta_z: Tensor | None = None
        self._schedule: SteeringSchedule | None = None
        self._hooks: list = []

    def set_steering_vector(
        self,
        delta_z: Tensor,
        schedule: str = "cosine",
        schedule_peak: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        self._delta_z = delta_z.float() * alpha
        self._schedule = SteeringSchedule(schedule=schedule, peak=schedule_peak)
        self._install_noise_hook()

    def clear_steering(self) -> None:
        self._delta_z = None
        self._schedule = None
        self._remove_hooks()

    def _install_noise_hook(self) -> None:
        self._remove_hooks()
        model = getattr(self.base_policy, "model", None) or getattr(self.base_policy, "_model", None)
        if model is None:
            raise RuntimeError("Cannot find RDT model from base_policy.")

        # For RDT, hook on the denoising UNet / transformer noise prediction output
        # The module name may vary — check the actual RDT codebase
        noise_pred_module = None
        for name, mod in model.named_modules():
            if "noise_pred" in name.lower() or "denoise" in name.lower():
                noise_pred_module = mod
                break

        if noise_pred_module is None:
            # Fallback: hook on the whole model's forward output
            noise_pred_module = model

        delta_z = self._delta_z
        schedule = self._schedule

        def hook(module, inputs, output):
            t_val = _extract_timestep(inputs)
            lam = schedule(t_val) if t_val is not None else schedule.peak
            dz = delta_z.to(output.device)
            if dz.dim() < output.dim():
                while dz.dim() < output.dim():
                    dz = dz.unsqueeze(0)
            return output + lam * dz

        h = noise_pred_module.register_forward_hook(hook)
        self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_action(self, observations: dict) -> Any:
        return self.base_policy.get_action(observations)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.base_policy, name)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_timestep(inputs: tuple) -> float | None:
    """
    Try to extract a scalar timestep value from a module's input arguments.
    GR00T's action encoder receives (actions, timesteps, embodiment_id).
    Returns None if no timestep found.
    """
    for inp in inputs:
        if isinstance(inp, Tensor) and inp.numel() <= 32:
            # Likely a timestep tensor; normalise to [0, 1]
            val = inp.float().mean().item()
            if 0.0 <= val <= 1.0:
                return val
            # GR00T uses discrete timestep buckets (0..999) — normalise
            if 1.0 < val <= 1000.0:
                return val / 1000.0
    return None


def build_steerable_policy(
    backbone: str,
    model_path: str,
    embodiment_tag: str,
    injection_mode: str = "hidden",
    device: str = "cuda",
    **model_kwargs,
) -> SteerableGr00tPolicy | SteerableRDTPolicy:
    """
    Factory function: load the base policy and wrap it with steering support.

    Args:
        backbone:       'groot' | 'groot_1b' | 'rdt1b'
        model_path:     HuggingFace model ID or local path
        embodiment_tag: GR00T embodiment tag or RDT embodiment name
        injection_mode: 'hidden' | 'velocity' (GR00T only)
        device:         'cuda' or 'cpu'
        **model_kwargs: Extra kwargs for the base policy constructor

    Returns:
        SteerableGr00tPolicy or SteerableRDTPolicy
    """
    if backbone in ("groot", "groot_1b"):
        try:
            from gr00t.eval.service import Gr00tPolicy
        except ImportError:
            raise ImportError(
                "GR00T not installed. Run: pip install git+https://github.com/NVIDIA/Isaac-GR00T.git"
            )
        base = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            device=device,
            **model_kwargs,
        )
        return SteerableGr00tPolicy(base, injection_mode=injection_mode)

    elif backbone == "rdt1b":
        try:
            from rdt.policy import RDTPolicy  # adjust import path as needed
        except ImportError:
            raise ImportError(
                "RDT not installed. Clone TeleHuman/Align-Then-Steer and install."
            )
        base = RDTPolicy(model_path=model_path, embodiment=embodiment_tag, **model_kwargs)
        return SteerableRDTPolicy(base)

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'groot', 'groot_1b', or 'rdt1b'.")
