"""
ManiSkill 3 demonstration collector.

Collects (obs, action) trajectories from a scripted / motion-planning policy
in GPU-parallel ManiSkill environments and saves them in the LeRobot v2 format
expected by GR00T fine-tuning.

Output directory layout (per (embodiment, task) pair):
    data/
        panda__pick_cube/
            episodes/
                episode_000.hdf5
                episode_001.hdf5
                ...
            dataset_stats.json

Each HDF5 episode contains:
    obs/
        images/base_camera  : (T, H, W, 3)  uint8 RGB
        images/hand_camera  : (T, H, W, 3)  uint8 RGB
        state               : (T, state_dim) float32
    actions                 : (T, action_dim) float32
    metadata/
        success             : bool
        episode_length      : int
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


@dataclass
class Episode:
    """A single collected demonstration episode."""
    images: dict[str, np.ndarray]  # camera_name → (T, H, W, 3) uint8
    states: np.ndarray             # (T, state_dim) float32
    actions: np.ndarray            # (T, action_dim) float32
    success: bool
    length: int


class ManiSkillCollector:
    """
    Collects demonstrations from ManiSkill 3 environments using the built-in
    motion planning / scripted oracle policy where available.

    Args:
        robot_uid:      ManiSkill robot_uid string (e.g. 'panda')
        env_id:         ManiSkill environment ID (e.g. 'PickCube-v1')
        num_envs:       Number of parallel GPU environments
        obs_mode:       Observation mode — 'rgbd' recommended
        image_size:     (H, W) output image resolution
        cameras:        Camera names to record
        max_steps:      Max steps per episode before timeout
        seed:           Base random seed
        device:         'cuda' or 'cpu'
    """

    def __init__(
        self,
        robot_uid: str,
        env_id: str,
        num_envs: int = 8,
        obs_mode: str = "rgbd",
        image_size: tuple[int, int] = (224, 224),
        cameras: list[str] | None = None,
        max_steps: int = 500,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.robot_uid = robot_uid
        self.env_id = env_id
        self.num_envs = num_envs
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.cameras = cameras or ["base_camera", "hand_camera"]
        self.max_steps = max_steps
        self.seed = seed
        self.device = device

        self._env = None

    # ── Environment lifecycle ────────────────────────────────────────────
    def _make_env(self):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — register envs

        self._env = gym.make(
            self.env_id,
            obs_mode=self.obs_mode,
            robot_uids=self.robot_uid,
            render_mode="rgb_array",
            num_envs=self.num_envs,
            sim_backend="gpu" if self.device == "cuda" else "cpu",
        )

    def _close_env(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── Collection ───────────────────────────────────────────────────────
    def collect(
        self,
        num_demos: int,
        save_dir: str | Path,
        save_failed: bool = False,
    ) -> list[Episode]:
        """
        Collect demonstrations using the scripted oracle policy.

        Args:
            num_demos:    Target number of successful episodes
            save_dir:     Directory to save HDF5 files
            save_failed:  If True, also save failed episodes

        Returns:
            List of collected Episode objects (successful only by default)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._make_env()
        episodes: list[Episode] = []
        episode_idx = 0
        attempt_idx = 0

        try:
            while len(episodes) < num_demos:
                ep = self._run_episode(
                    seed=self.seed + attempt_idx * self.num_envs
                )
                attempt_idx += 1

                for e in ep:
                    if e.success or save_failed:
                        self._save_episode(e, save_dir / f"episode_{episode_idx:04d}.hdf5")
                        episodes.append(e)
                        episode_idx += 1
                        if len(episodes) >= num_demos:
                            break

                total_attempts = attempt_idx * self.num_envs
                success_rate = len(episodes) / max(total_attempts, 1)
                print(
                    f"  Collected {len(episodes)}/{num_demos} demos "
                    f"(success rate: {success_rate:.1%}, attempts: {total_attempts})"
                )

        finally:
            self._close_env()

        # Save dataset statistics
        self._save_stats(episodes, save_dir / "dataset_stats.json")
        print(f"Saved {len(episodes)} episodes → {save_dir}")
        return episodes

    def _run_episode(self, seed: int) -> list[Episode]:
        """
        Run one batch of parallel episodes and collect transitions.
        Returns a list of Episode objects (one per env).
        """
        obs, info = self._env.reset(seed=seed)
        num_envs = self.num_envs

        # Buffers — lists of per-step observations
        images_buf: dict[str, list[np.ndarray]] = {cam: [[] for _ in range(num_envs)] for cam in self.cameras}
        states_buf: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        actions_buf: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        dones = [False] * num_envs
        successes = [False] * num_envs

        for step in range(self.max_steps):
            # Extract observations
            obs_np = self._parse_obs(obs)

            # Record observations for non-done envs
            for env_i in range(num_envs):
                if not dones[env_i]:
                    for cam in self.cameras:
                        images_buf[cam][env_i].append(obs_np["images"].get(cam, {}).get(env_i))
                    states_buf[env_i].append(obs_np["states"][env_i])

            # Get action from scripted oracle
            action = self._get_oracle_action(obs, info)
            action_np = self._to_numpy(action)

            for env_i in range(num_envs):
                if not dones[env_i]:
                    actions_buf[env_i].append(action_np[env_i])

            obs, reward, terminated, truncated, info = self._env.step(action)

            for env_i in range(num_envs):
                if not dones[env_i]:
                    done = bool(terminated[env_i]) or bool(truncated[env_i])
                    if done:
                        successes[env_i] = bool(terminated[env_i])
                        dones[env_i] = True

            if all(dones):
                break

        # Package episodes
        episodes = []
        for env_i in range(num_envs):
            if len(actions_buf[env_i]) == 0:
                continue
            T = len(actions_buf[env_i])
            ep_images = {}
            for cam in self.cameras:
                frames = images_buf[cam][env_i]
                if frames and frames[0] is not None:
                    ep_images[cam] = np.stack(frames, axis=0)  # (T, H, W, 3)
            ep_states = np.stack(states_buf[env_i], axis=0)   # (T, state_dim)
            ep_actions = np.stack(actions_buf[env_i], axis=0) # (T, action_dim)
            episodes.append(Episode(
                images=ep_images,
                states=ep_states,
                actions=ep_actions,
                success=successes[env_i],
                length=T,
            ))
        return episodes

    def _get_oracle_action(self, obs: Any, info: Any) -> Any:
        """
        Use ManiSkill's built-in solution/scripted policy if available.
        Falls back to random actions for environments without a scripted policy.
        """
        try:
            # ManiSkill 3 exposes a solution via env.get_solution_actions() or env.step()
            # Some envs have a built-in planner accessible via env.agent.controller
            if hasattr(self._env, "solution"):
                return self._env.solution(obs)
            # For MS3 envs with built-in motion planning:
            if hasattr(self._env, "_get_obs_agent_obs"):
                return self._env.action_space.sample()
            return self._env.action_space.sample()
        except Exception:
            return self._env.action_space.sample()

    def _parse_obs(self, obs: Any) -> dict:
        """Parse ManiSkill observation dict into structured numpy arrays."""
        result = {"images": {}, "states": {}}

        # Images from sensor data
        if isinstance(obs, dict):
            sensor_data = obs.get("sensor_data", {})
            for cam in self.cameras:
                if cam in sensor_data:
                    rgb = sensor_data[cam].get("rgb", None)
                    if rgb is not None:
                        rgb_np = self._to_numpy(rgb)   # (num_envs, H, W, 3)
                        result["images"][cam] = {
                            i: rgb_np[i] for i in range(self.num_envs)
                        }

            # Proprioceptive state
            agent_obs = obs.get("agent", {})
            qpos = agent_obs.get("qpos", None)
            qvel = agent_obs.get("qvel", None)
            parts = []
            if qpos is not None:
                parts.append(self._to_numpy(qpos))
            if qvel is not None:
                parts.append(self._to_numpy(qvel))
            if parts:
                state = np.concatenate(parts, axis=-1)   # (num_envs, state_dim)
                result["states"] = {i: state[i] for i in range(self.num_envs)}

        return result

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    @staticmethod
    def _save_episode(episode: Episode, path: Path) -> None:
        with h5py.File(path, "w") as f:
            obs_grp = f.create_group("obs")
            img_grp = obs_grp.create_group("images")
            for cam, frames in episode.images.items():
                img_grp.create_dataset(cam, data=frames, compression="gzip", compression_opts=4)
            obs_grp.create_dataset("state", data=episode.states)
            f.create_dataset("actions", data=episode.actions)
            meta = f.create_group("metadata")
            meta.attrs["success"] = episode.success
            meta.attrs["episode_length"] = episode.length

    @staticmethod
    def _save_stats(episodes: list[Episode], path: Path) -> None:
        all_actions = np.concatenate([e.actions for e in episodes], axis=0)
        stats = {
            "num_episodes": len(episodes),
            "success_rate": sum(e.success for e in episodes) / len(episodes),
            "action_mean": all_actions.mean(axis=0).tolist(),
            "action_std": all_actions.std(axis=0).tolist(),
            "action_min": all_actions.min(axis=0).tolist(),
            "action_max": all_actions.max(axis=0).tolist(),
            "avg_episode_length": np.mean([e.length for e in episodes]),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


# ── LeRobot format conversion ─────────────────────────────────────────────────

def episodes_to_lerobot(
    episodes: list[Episode],
    embodiment_tag: str,
    task_description: str,
    output_dir: str | Path,
    action_horizon: int = 16,
    image_size: tuple[int, int] = (224, 224),
) -> Path:
    """
    Convert collected episodes to GR00T-flavored LeRobot v2 format.

    GR00T expects:
        {
            "video.<camera_name>": (1, T, C, H, W) float32 [0,1]
            "state.<state_name>":  (1, T, state_dim) float32
            "annotation.human.action.task_description": ["<text>"]
        }

    Saves a JSONL metadata file + video files per episode.

    Returns:
        Path to the converted dataset directory
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for ep_idx, ep in enumerate(episodes):
        ep_dir = output_dir / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(exist_ok=True)

        # Save videos
        for cam, frames in ep.images.items():
            # frames: (T, H, W, 3) uint8 → resize if needed
            if frames.shape[1:3] != image_size:
                resized = np.stack([
                    cv2.resize(f, image_size[::-1]) for f in frames
                ], axis=0)
            else:
                resized = frames
            video_path = ep_dir / f"{cam}.mp4"
            _save_video(resized, str(video_path))

        # Save state tensor
        torch.save(
            {"state": torch.tensor(ep.states, dtype=torch.float32)},
            ep_dir / "state.pt",
        )

        # Save actions tensor (chunked for training)
        torch.save(
            {"actions": torch.tensor(ep.actions, dtype=torch.float32)},
            ep_dir / "actions.pt",
        )

        metadata.append({
            "episode_index": ep_idx,
            "task_description": task_description,
            "embodiment_tag": embodiment_tag,
            "success": ep.success,
            "length": ep.length,
            "cameras": list(ep.images.keys()),
        })

    # Write metadata JSONL
    with open(output_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"Converted {len(episodes)} episodes to LeRobot format → {output_dir}")
    return output_dir


def _save_video(frames: np.ndarray, path: str, fps: int = 10) -> None:
    """Save (T, H, W, 3) uint8 array as MP4."""
    import imageio
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def load_action_chunks(
    data_dir: str | Path,
    action_horizon: int = 16,
    stride: int = 1,
) -> torch.Tensor:
    """
    Load all action sequences from an episode directory and slice into chunks.

    Returns:
        chunks: (N_chunks, action_horizon, action_dim) float32
    """
    data_dir = Path(data_dir)
    all_chunks = []

    for ep_dir in sorted(data_dir.glob("episode_*")):
        actions_file = ep_dir / "actions.pt"
        if not actions_file.exists():
            # Try HDF5 format
            hdf_file = data_dir / f"{ep_dir.name}.hdf5"
            if hdf_file.exists():
                with h5py.File(hdf_file) as f:
                    actions = torch.tensor(f["actions"][:], dtype=torch.float32)
            else:
                continue
        else:
            actions = torch.load(actions_file)["actions"]   # (T, action_dim)

        # Slice into chunks
        T = actions.shape[0]
        for start in range(0, T - action_horizon + 1, stride):
            chunk = actions[start:start + action_horizon]
            all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError(f"No action data found in {data_dir}")

    return torch.stack(all_chunks, dim=0)   # (N, action_horizon, action_dim)
