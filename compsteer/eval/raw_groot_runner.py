"""
Raw GR00T benchmark runner for ManiSkill — no steering, no fine-tuning.

Loads a stock Gr00tPolicy and rolls it out in ManiSkill 3 environments, purely
to sanity-check that a given GR00T checkpoint + embodiment tag can drive a
given robot/task before any CompSteer machinery is layered on top.

GR00T's observation modality keys (video/state/action) are embodiment- and
checkpoint-specific, so this runner queries policy.get_modality_config() at
load time and maps ManiSkill's obs dict onto whatever keys the policy
actually expects, instead of hardcoding names.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class RawEvalResult:
    """Result of rolling out one (embodiment, task) pair with raw GR00T."""
    robot_uid: str
    env_id: str
    task_description: str
    num_episodes: int
    successes: list[bool] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(self.successes) / max(len(self.successes), 1)

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    def to_dict(self) -> dict:
        return {
            "robot_uid": self.robot_uid,
            "env_id": self.env_id,
            "task_description": self.task_description,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "mean_episode_length": self.mean_length,
            "successes": self.successes,
            "episode_lengths": self.episode_lengths,
        }


class RawGr00tManiSkillRunner:
    """
    Rolls out a stock GR00T policy in a single ManiSkill environment.

    Usage:
        runner = RawGr00tManiSkillRunner(
            model_path="nvidia/GR00T-N1.6-3B",
            embodiment_tag="NEW_EMBODIMENT",
            robot_uid="panda",
            env_id="PickCube-v1",
            task_description="Pick up the red cube and lift it above the table",
        )
        result = runner.run(num_episodes=20, max_steps=200, seed=100)
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str,
        robot_uid: str,
        env_id: str,
        task_description: str,
        video_key_map: dict[str, str] | None = None,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.embodiment_tag = embodiment_tag
        self.robot_uid = robot_uid
        self.env_id = env_id
        self.task_description = task_description
        # Optional override: ManiSkill camera name -> GR00T video modality key.
        # If not given, cameras are assigned to GR00T's video keys in order.
        self.video_key_map = video_key_map or {}
        self.device = device

        self._policy: Any = None
        self._video_keys: list[str] = []
        self._state_keys: list[str] = []
        self._action_key: str = ""
        self._env: Any = None

    # ── Policy loading ───────────────────────────────────────────────────
    def _load_policy(self) -> None:
        if self._policy is not None:
            return

        try:
            from gr00t.policy import Gr00tPolicy
            from gr00t.data.embodiment_tags import EmbodimentTag

            tag = getattr(EmbodimentTag, self.embodiment_tag, self.embodiment_tag)
        except ImportError:
            # Older Isaac-GR00T releases exposed the policy under
            # gr00t.eval.service instead of gr00t.policy.
            from gr00t.eval.service import Gr00tPolicy  # type: ignore

            tag = self.embodiment_tag

        self._policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag=tag,
            device=self.device,
        )

        config = self._policy.get_modality_config()
        self._video_keys = list(config["video"].modality_keys)
        self._state_keys = list(config["state"].modality_keys)
        self._action_key = list(config["action"].modality_keys)[0]
        print(
            f"[raw_groot_runner] modality config: "
            f"video={self._video_keys} state={self._state_keys} action={self._action_key}"
        )

    # ── Environment lifecycle ────────────────────────────────────────────
    def _make_env(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers ManiSkill envs

        self._env = gym.make(
            self.env_id,
            obs_mode="rgbd",
            robot_uids=self.robot_uid,
            render_mode="rgb_array",
            num_envs=1,
        )

    def _close_env(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── Rollout ───────────────────────────────────────────────────────────
    def run(self, num_episodes: int, max_steps: int, seed: int = 100) -> RawEvalResult:
        self._load_policy()
        self._make_env()

        result = RawEvalResult(
            robot_uid=self.robot_uid,
            env_id=self.env_id,
            task_description=self.task_description,
            num_episodes=num_episodes,
        )

        try:
            for ep_i in range(num_episodes):
                obs, info = self._env.reset(seed=seed + ep_i)
                done = False
                step = 0
                success = False

                while not done and step < max_steps:
                    obs_dict = self._build_obs(obs)
                    action, _info = self._policy.get_action(obs_dict)
                    action_np = self._parse_action(action)

                    obs, reward, terminated, truncated, info = self._env.step(action_np)
                    step += 1
                    done = bool(terminated) or bool(truncated)
                    if done:
                        success = bool(terminated)

                result.successes.append(success)
                result.episode_lengths.append(step)

                sr = result.success_rate
                print(f"  Episode {ep_i + 1}/{num_episodes}  success={success}  steps={step}  running SR={sr:.2%}")
        finally:
            self._close_env()

        print(f"→ {self.robot_uid} × {self.env_id}: success_rate={result.success_rate:.2%}  mean_len={result.mean_length:.1f}")
        return result

    # ── Obs / action conversion ──────────────────────────────────────────
    def _build_obs(self, obs: Any) -> dict:
        """Map a ManiSkill observation dict onto GR00T's expected modality dict."""
        video: dict[str, np.ndarray] = {}
        sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
        cam_names = list(sensor_data.keys())

        for cam_name in cam_names:
            groot_key = self.video_key_map.get(cam_name)
            if groot_key is None:
                idx = cam_names.index(cam_name)
                if idx >= len(self._video_keys):
                    continue
                groot_key = self._video_keys[idx]
            rgb = sensor_data[cam_name].get("rgb", None)
            if rgb is None:
                continue
            rgb_np = self._to_numpy(rgb)  # (num_envs, H, W, 3) uint8
            frame = rgb_np[0]
            # GR00T expects (B, T, H, W, 3) uint8
            video[groot_key] = frame[None, None, ...].astype(np.uint8)

        state: dict[str, np.ndarray] = {}
        agent_obs = obs.get("agent", {}) if isinstance(obs, dict) else {}
        parts = []
        for key in ["qpos", "qvel"]:
            if key in agent_obs:
                parts.append(self._to_numpy(agent_obs[key])[0])
        if parts and self._state_keys:
            state_vec = np.concatenate(parts, axis=-1).astype(np.float32)
            state[self._state_keys[0]] = state_vec[None, None, ...]  # (B, T, D)

        return {
            "video": video,
            "state": state,
            "language": {"task": [[self.task_description]]},
        }

    def _parse_action(self, action: Any) -> np.ndarray:
        """Extract the first timestep of the predicted action chunk."""
        if isinstance(action, dict):
            arr = action.get(self._action_key)
            if arr is None:
                arr = next(iter(action.values()))
        else:
            arr = action

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)

        # (B, T, action_dim) -> first batch, first timestep
        while arr.ndim > 1:
            arr = arr[0]
        return arr

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)


def save_raw_results(results: list[RawEvalResult], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = [r.to_dict() for r in results]
    with open(output_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'Robot':10} {'Env':24} {'Episodes':9} {'Success Rate':13} {'Mean Length':12}")
    print("-" * 72)
    for r in results:
        print(f"{r.robot_uid:10} {r.env_id:24} {r.num_episodes:9} {r.success_rate:13.2%} {r.mean_length:12.1f}")

    print(f"\nResults saved → {output_dir / 'results.json'}")
