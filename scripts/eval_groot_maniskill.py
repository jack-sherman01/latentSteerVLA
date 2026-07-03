"""
Raw GR00T evaluation on ManiSkill 3 — sanity check with no CompSteer steering.

Rolls a stock GR00T checkpoint out in ManiSkill environments defined in
configs/tasks.yaml, against a robot defined in configs/embodiments.yaml
(or an ad-hoc robot_uid/embodiment_tag pair passed on the command line).

Usage:
    # Panda on all training tasks
    python scripts/eval_groot_maniskill.py --embodiment panda --tasks pick_cube stack_cube

    # All embodiments x all tasks in the registry
    python scripts/eval_groot_maniskill.py --embodiment all --tasks all

    # Ad-hoc robot not in the registry
    python scripts/eval_groot_maniskill.py \\
        --robot_uid xarm6_robotiq85 --embodiment_tag NEW_EMBODIMENT \\
        --tasks pick_cube

Docker:
    docker compose run --rm groot-server \\
        python scripts/eval_groot_maniskill.py --embodiment panda --tasks pick_cube
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.eval.raw_groot_runner import RawGr00tManiSkillRunner, save_raw_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw GR00T rollout on ManiSkill (no steering)")
    p.add_argument("--model_path", default="nvidia/GR00T-N1.7-3B")  # NOT N1-2B, see slurm/eval_groot_robocasa.sh
    p.add_argument("--embodiment", default="panda",
                    help="Key in configs/embodiments.yaml, or 'all'. Ignored if --robot_uid is set.")
    p.add_argument("--robot_uid", default=None, help="Ad-hoc ManiSkill robot_uid, bypasses --embodiment")
    p.add_argument("--embodiment_tag", default=None, help="Ad-hoc GR00T embodiment tag, requires --robot_uid")
    p.add_argument("--tasks", nargs="+", default=["pick_cube"],
                    help="Keys in configs/tasks.yaml, or 'all'")
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--results_root", default="results/raw_groot/maniskill")
    p.add_argument("--configs_dir", default="configs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["tasks"]

    if args.robot_uid is not None:
        if args.embodiment_tag is None:
            print("ERROR: --embodiment_tag is required when --robot_uid is set.")
            sys.exit(1)
        embodiments = {"custom": {"robot_uid": args.robot_uid, "groot_tag": args.embodiment_tag}}
        embodiment_ids = ["custom"]
    else:
        with open(cfg_dir / "embodiments.yaml") as f:
            embodiments = yaml.safe_load(f)["embodiments"]
        embodiment_ids = list(embodiments.keys()) if args.embodiment == "all" else [args.embodiment]

    task_ids = list(tasks_cfg.keys()) if args.tasks == ["all"] else args.tasks

    results = []
    for emb_id in embodiment_ids:
        emb_cfg = embodiments[emb_id]
        for task_id in task_ids:
            task_cfg = tasks_cfg[task_id]
            print(f"\n=== {emb_id} × {task_id} ({task_cfg['maniskill_env']}) ===")

            runner = RawGr00tManiSkillRunner(
                model_path=args.model_path,
                embodiment_tag=emb_cfg["groot_tag"],
                robot_uid=emb_cfg["robot_uid"],
                env_id=task_cfg["maniskill_env"],
                task_description=task_cfg["description"],
                device=args.device,
            )
            result = runner.run(
                num_episodes=args.num_episodes,
                max_steps=task_cfg.get("max_steps", 400),
                seed=args.seed,
            )
            results.append(result)

    results_dir = Path(args.results_root)
    save_raw_results(results, results_dir)


if __name__ == "__main__":
    main()
