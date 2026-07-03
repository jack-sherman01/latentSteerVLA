"""
Phase 0 — Collect ManiSkill demonstrations.

Collects scripted oracle demos for a single (embodiment, task) pair and saves
them in HDF5 + LeRobot v2 format.  Designed to run as a SLURM array job
over all training pairs.

Usage:
    python scripts/01_collect_demos.py --embodiment panda --task pick_cube
    python scripts/01_collect_demos.py --embodiment xarm6 --task stack_cube --num_demos 100

SLURM array usage (see slurm/01_collect_demos.slurm):
    sbatch --array=0-29 slurm/01_collect_demos.slurm
    # Array index encodes the (embodiment, task) pair
"""

import argparse
import sys
from pathlib import Path

import yaml

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.data.maniskill_collector import (
    ManiSkillCollector,
    episodes_to_lerobot,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect ManiSkill demos for CompSteer")
    p.add_argument("--embodiment", required=True, help="Embodiment ID (must be in configs/embodiments.yaml)")
    p.add_argument("--task",       required=True, help="Task ID (must be in configs/tasks.yaml)")
    p.add_argument("--num_demos",  type=int, default=50)
    p.add_argument("--num_envs",   type=int, default=8, help="Parallel ManiSkill envs")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--data_root",  default="data", help="Root directory for saving demos")
    p.add_argument("--save_failed", action="store_true")
    p.add_argument("--no_lerobot", action="store_true", help="Skip LeRobot format conversion")
    p.add_argument("--configs_dir", default="configs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "embodiments.yaml") as f:
        embodiments_cfg = yaml.safe_load(f)["embodiments"]

    with open(cfg_dir / "tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["tasks"]

    if args.embodiment not in embodiments_cfg:
        print(f"ERROR: embodiment '{args.embodiment}' not found in embodiments.yaml")
        print(f"Available: {list(embodiments_cfg.keys())}")
        sys.exit(1)

    if args.task not in tasks_cfg:
        print(f"ERROR: task '{args.task}' not found in tasks.yaml")
        print(f"Available: {list(tasks_cfg.keys())}")
        sys.exit(1)

    emb_cfg  = embodiments_cfg[args.embodiment]
    task_cfg = tasks_cfg[args.task]

    pair_name = f"{args.embodiment}__{args.task}"
    hdf5_dir  = Path(args.data_root) / pair_name / "raw"
    lr_dir    = Path(args.data_root) / pair_name / "lerobot"

    print(f"\n{'='*60}")
    print(f"Collecting demos: {args.embodiment} × {args.task}")
    print(f"  Robot UID:   {emb_cfg['robot_uid']}")
    print(f"  Env ID:      {task_cfg['maniskill_env']}")
    print(f"  Num demos:   {args.num_demos}")
    print(f"  Num envs:    {args.num_envs}")
    print(f"  Output dir:  {hdf5_dir}")
    print(f"{'='*60}\n")

    # ── Collect ──────────────────────────────────────────────────────────
    collector = ManiSkillCollector(
        robot_uid=emb_cfg["robot_uid"],
        env_id=task_cfg["maniskill_env"],
        num_envs=args.num_envs,
        obs_mode="rgbd",
        image_size=(224, 224),
        cameras=["base_camera", "hand_camera"],
        max_steps=task_cfg.get("max_steps", 500),
        seed=args.seed,
    )

    episodes = collector.collect(
        num_demos=args.num_demos,
        save_dir=hdf5_dir,
        save_failed=args.save_failed,
    )

    # ── Convert to LeRobot format ─────────────────────────────────────────
    if not args.no_lerobot:
        episodes_to_lerobot(
            episodes=episodes,
            embodiment_tag=emb_cfg["groot_tag"],
            task_description=task_cfg["description"],
            output_dir=lr_dir,
            action_horizon=16,
            image_size=(224, 224),
        )

    print(f"\nDone! {len(episodes)} demos saved for {pair_name}")


if __name__ == "__main__":
    main()
