"""
CompSteer evaluator — runs all four evaluation splits in ManiSkill.

Split definitions:
    A — Seen embodiment + seen task      (sanity check: composition ≈ ATE oracle)
    B — New embodiment + seen task       (embodiment generalisation)
    C — Seen embodiment + new task       (task generalisation)
    D — New embodiment + new task        (full zero-shot — key contribution)

For each split and each baseline, the evaluator runs num_episodes rollouts
and records success rate, episode length, and trajectory smoothness.
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
class EvalResult:
    """Results for one (embodiment, task, method) evaluation cell."""
    embodiment_id: str
    task_id: str
    method: str
    split: str              # A | B | C | D | baseline_nonadapt | baseline_ate

    num_episodes: int
    successes: list[bool]
    episode_lengths: list[int]
    smoothness_scores: list[float]   # Mean jerk (lower = smoother)

    # ── Derived metrics ──────────────────────────────────────────────────
    @property
    def success_rate(self) -> float:
        return sum(self.successes) / max(len(self.successes), 1)

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_smoothness(self) -> float:
        return float(np.mean(self.smoothness_scores)) if self.smoothness_scores else 0.0

    def to_dict(self) -> dict:
        return {
            "embodiment_id": self.embodiment_id,
            "task_id": self.task_id,
            "method": self.method,
            "split": self.split,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "mean_episode_length": self.mean_length,
            "mean_smoothness": self.mean_smoothness,
            "successes": self.successes,
        }


class CompSteerEvaluator:
    """
    Evaluates CompSteer (and baselines) across all four splits in ManiSkill 3.

    Usage:
        evaluator = CompSteerEvaluator(cfg, factorization, library)
        results = evaluator.run_all_splits()
        evaluator.save_results(results, "results/")
        evaluator.print_table(results)
    """

    def __init__(
        self,
        embodiments_cfg: dict,
        tasks_cfg: dict,
        factorization: Any,             # FactorizationResult
        library: Any,                   # SteeringLibrary
        num_episodes: int = 50,
        num_envs: int = 8,
        eval_seed: int = 100,
        backbone: str = "groot",
        model_path: str = "nvidia/GR00T-N1.6-3B",  # NOT N1-2B, see scripts/run_groot_server.sh
        injection_mode: str = "hidden",
        compose_alpha: float = 1.0,
        compose_beta: float = 1.0,
        schedule: str = "cosine",
        schedule_peak: float = 1.0,
        device: str = "cuda",
        render_videos: bool = False,
        video_dir: str | None = None,
    ):
        self.embodiments_cfg = embodiments_cfg
        self.tasks_cfg = tasks_cfg
        self.factorization = factorization
        self.library = library
        self.num_episodes = num_episodes
        self.num_envs = num_envs
        self.eval_seed = eval_seed
        self.backbone = backbone
        self.model_path = model_path
        self.injection_mode = injection_mode
        self.compose_alpha = compose_alpha
        self.compose_beta = compose_beta
        self.schedule = schedule
        self.schedule_peak = schedule_peak
        self.device = device
        self.render_videos = render_videos
        self.video_dir = Path(video_dir) if video_dir else None

        self._policy = None   # lazy init

    # ── Main eval entry points ───────────────────────────────────────────

    def run_all_splits(self) -> list[EvalResult]:
        """
        Run all four splits plus the no-adaptation baseline.
        Returns a flat list of EvalResult objects.
        """
        train_embodiments = [k for k, v in self.embodiments_cfg.items() if v["split"] == "train"]
        test_embodiments  = [k for k, v in self.embodiments_cfg.items() if v["split"] == "test"]
        train_tasks       = [k for k, v in self.tasks_cfg.items() if v["split"] == "train"]
        test_tasks        = [k for k, v in self.tasks_cfg.items() if v["split"] == "test"]

        results: list[EvalResult] = []

        # Split A: seen × seen
        for emb in train_embodiments:
            for task in train_tasks:
                r = self.run_single(emb, task, method="compsteer", split="A")
                results.append(r)

        # Split B: new embodiment × seen task
        for emb in test_embodiments:
            for task in train_tasks:
                r = self.run_single(emb, task, method="compsteer", split="B")
                results.append(r)

        # Split C: seen embodiment × new task
        for emb in train_embodiments:
            for task in test_tasks:
                r = self.run_single(emb, task, method="compsteer", split="C")
                results.append(r)

        # Split D: new × new
        for emb in test_embodiments:
            for task in test_tasks:
                r = self.run_single(emb, task, method="compsteer", split="D")
                results.append(r)

        # Baseline: no adaptation
        for emb in test_embodiments + train_embodiments[:1]:
            for task in test_tasks + train_tasks[:1]:
                r = self.run_single(emb, task, method="no_adaptation", split="baseline")
                results.append(r)

        return results

    def run_single(
        self,
        embodiment_id: str,
        task_id: str,
        method: str = "compsteer",
        split: str = "D",
    ) -> EvalResult:
        """
        Evaluate one (embodiment, task, method) cell.

        Args:
            embodiment_id:  Target embodiment
            task_id:        Target task
            method:         'compsteer' | 'no_adaptation'
            split:          'A' | 'B' | 'C' | 'D' | 'baseline'

        Returns:
            EvalResult
        """
        print(f"\nEval [{split}] {method} | {embodiment_id} × {task_id}")

        # Load / configure policy
        policy = self._get_policy(embodiment_id, task_id, method)

        # Build ManiSkill eval env
        env_id = self.tasks_cfg[task_id]["maniskill_env"]
        env = self._make_eval_env(embodiment_id, env_id)

        successes: list[bool] = []
        ep_lengths: list[int] = []
        smoothness_scores: list[float] = []

        try:
            for ep_i in range(self.num_episodes):
                obs, info = env.reset(seed=self.eval_seed + ep_i)
                done = False
                step = 0
                action_history: list[np.ndarray] = []
                success = False

                while not done and step < self.tasks_cfg[task_id].get("max_steps", 400):
                    obs_dict = self._build_policy_obs(obs, task_id)
                    action_chunk = policy.get_action(obs_dict)
                    action = self._parse_action(action_chunk, embodiment_id)

                    obs, reward, terminated, truncated, info = env.step(action)
                    action_history.append(np.array(action))
                    step += 1
                    done = bool(terminated) or bool(truncated)
                    if done:
                        success = bool(terminated)

                successes.append(success)
                ep_lengths.append(step)
                smoothness_scores.append(compute_smoothness(action_history))

                if (ep_i + 1) % 10 == 0:
                    sr = sum(successes) / len(successes)
                    print(f"  Episode {ep_i+1}/{self.num_episodes}  SR={sr:.2%}")

        finally:
            env.close()

        result = EvalResult(
            embodiment_id=embodiment_id,
            task_id=task_id,
            method=method,
            split=split,
            num_episodes=self.num_episodes,
            successes=successes,
            episode_lengths=ep_lengths,
            smoothness_scores=smoothness_scores,
        )
        print(f"  → success_rate={result.success_rate:.2%}  mean_len={result.mean_length:.1f}")
        return result

    # ── Policy management ────────────────────────────────────────────────

    def _get_policy(self, embodiment_id: str, task_id: str, method: str) -> Any:
        """Lazily load and configure the steerable policy."""
        from compsteer.injection.groot_hook import build_steerable_policy
        from compsteer.steering.compose import compose_steering_vector

        emb_cfg = self.embodiments_cfg[embodiment_id]
        groot_tag = emb_cfg.get("groot_tag", "NEW_EMBODIMENT")

        if self._policy is None:
            self._policy = build_steerable_policy(
                backbone=self.backbone,
                model_path=self.model_path,
                embodiment_tag=groot_tag,
                injection_mode=self.injection_mode,
                device=self.device,
            )

        if method == "compsteer":
            delta_z = compose_steering_vector(
                embodiment_id=embodiment_id,
                task_id=task_id,
                factorization=self.factorization,
                library=self.library,
                alpha=self.compose_alpha,
                beta=self.compose_beta,
            )
            self._policy.set_steering_vector(
                delta_z=delta_z,
                schedule=self.schedule,
                schedule_peak=self.schedule_peak,
            )
        else:
            self._policy.clear_steering()

        return self._policy

    # ── Environment management ───────────────────────────────────────────

    def _make_eval_env(self, embodiment_id: str, env_id: str) -> Any:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        robot_uid = self.embodiments_cfg[embodiment_id]["robot_uid"]
        return gym.make(
            env_id,
            obs_mode="rgbd",
            robot_uids=robot_uid,
            render_mode="rgb_array",
            num_envs=1,
        )

    def _build_policy_obs(self, obs: Any, task_id: str) -> dict:
        """Convert ManiSkill obs to GR00T-compatible observation dict."""
        task_desc = self.tasks_cfg[task_id]["description"]
        obs_dict: dict = {}

        if isinstance(obs, dict):
            # Images
            sensor_data = obs.get("sensor_data", {})
            for cam_name, cam_data in sensor_data.items():
                rgb = cam_data.get("rgb", None)
                if rgb is not None:
                    if isinstance(rgb, torch.Tensor):
                        rgb = rgb.float() / 255.0
                    else:
                        rgb = torch.tensor(rgb, dtype=torch.float32) / 255.0
                    # GR00T expects (1, 1, C, H, W) — batch=1, T=1
                    if rgb.dim() == 3:
                        rgb = rgb.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                    obs_dict[f"video.{cam_name}"] = rgb

            # State
            agent_data = obs.get("agent", {})
            parts = []
            for key in ["qpos", "qvel"]:
                if key in agent_data:
                    v = agent_data[key]
                    if isinstance(v, torch.Tensor):
                        parts.append(v.float())
                    else:
                        parts.append(torch.tensor(v, dtype=torch.float32))
            if parts:
                state = torch.cat(parts, dim=-1)
                if state.dim() == 1:
                    state = state.unsqueeze(0).unsqueeze(0)   # (1, 1, state_dim)
                obs_dict["state.joint_state"] = state

        obs_dict["annotation.human.action.task_description"] = [task_desc]
        return obs_dict

    def _parse_action(self, action_chunk: Any, embodiment_id: str) -> np.ndarray:
        """Extract the first action from the predicted chunk."""
        if isinstance(action_chunk, dict):
            action = action_chunk.get("action_pred", action_chunk.get("actions", None))
        else:
            action = action_chunk

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if action is None:
            action_dim = self.embodiments_cfg[embodiment_id]["total_action_dim"]
            return np.zeros(action_dim)

        # action shape: (B, T, action_dim) → take first batch, first step
        while action.ndim > 1:
            action = action[0]

        return action

    # ── Results I/O ──────────────────────────────────────────────────────

    @staticmethod
    def save_results(results: list[EvalResult], output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_dicts = [r.to_dict() for r in results]
        with open(output_dir / "results.json", "w") as f:
            json.dump(all_dicts, f, indent=2)

        # Per-split summary
        for split in ["A", "B", "C", "D"]:
            split_results = [r for r in results if r.split == split and r.method == "compsteer"]
            if split_results:
                mean_sr = np.mean([r.success_rate for r in split_results])
                print(f"Split {split}: mean success rate = {mean_sr:.2%} ({len(split_results)} pairs)")

        print(f"Results saved → {output_dir}")

    @staticmethod
    def print_table(results: list[EvalResult]) -> None:
        """Print a compact success-rate table grouped by split."""
        print("\n" + "=" * 70)
        print(f"{'Split':6} {'Method':20} {'Pairs':6} {'Success Rate':12} {'Mean Length':12}")
        print("-" * 70)

        methods = sorted(set(r.method for r in results))
        splits = ["A", "B", "C", "D", "baseline"]

        for split in splits:
            for method in methods:
                subset = [r for r in results if r.split == split and r.method == method]
                if not subset:
                    continue
                sr = np.mean([r.success_rate for r in subset])
                ml = np.mean([r.mean_length for r in subset])
                print(f"{split:6} {method:20} {len(subset):6} {sr:12.2%} {ml:12.1f}")

        print("=" * 70)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_smoothness(action_history: list[np.ndarray]) -> float:
    """
    Compute trajectory smoothness as mean squared jerk (finite differences).
    Lower is smoother.
    """
    if len(action_history) < 3:
        return 0.0

    actions = np.stack(action_history, axis=0)   # (T, action_dim)
    vel   = np.diff(actions, axis=0)              # (T-1, action_dim)
    accel = np.diff(vel, axis=0)                  # (T-2, action_dim)
    jerk  = np.diff(accel, axis=0)                # (T-3, action_dim)
    return float(np.mean(jerk ** 2))
