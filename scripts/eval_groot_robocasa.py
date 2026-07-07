"""
Raw GR00T evaluation on RoboCasa — sanity check with no CompSteer steering.

Isaac-GR00T ships its own official RoboCasa evaluation harness
(gr00t/eval/rollout_policy.py + gr00t/eval/run_gr00t_server.py) built on a
ZMQ client/server split: the GR00T policy runs in the main uv environment,
while the RoboCasa client runs in an isolated venv because `robosuite`
(a RoboCasa dependency) conflicts with GR00T's own pinned dependencies.

Rather than re-implement that client/server protocol here, this script
drives the official rollout_policy.py once per task from
configs/robocasa_tasks.yaml, using this repo's task registry as the source
of truth, and aggregates the per-task results into one JSON report.

Prerequisites:
    1. A GR00T policy server is already running and reachable
       (see scripts/run_groot_server.sh / the `groot-server` compose service).
    2. The isolated RoboCasa venv has been set up
       (see scripts/setup_robocasa_env.sh / the `robocasa-client` compose service).

Usage:
    python scripts/eval_groot_robocasa.py \\
        --tasks open_drawer coffee_press_button \\
        --policy_host groot-server --policy_port 5555

    # All tasks in the registry
    python scripts/eval_groot_robocasa.py --tasks all

Docker:
    docker compose run --rm robocasa-client \\
        python scripts/eval_groot_robocasa.py --tasks all --policy_host groot-server
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Best-effort patterns for pulling a success rate out of rollout_policy.py's
# stdout. The exact format may drift across Isaac-GR00T releases — raw stdout
# is always preserved per task so results can be checked by hand if this
# fails to match.
_SUCCESS_RATE_PATTERNS = [
    re.compile(r"success[_ ]rate[:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"success[:\s]+([0-9]*\.?[0-9]+)\s*%", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw GR00T rollout on RoboCasa via the official client/server harness")
    p.add_argument("--gr00t_root", default="/workspace/Isaac-GR00T",
                    help="Path to the cloned Isaac-GR00T repo")
    p.add_argument("--robocasa_venv", default=None,
                    help="Path to the isolated RoboCasa venv "
                         "(default: <gr00t_root>/gr00t/eval/sim/robocasa/robocasa_uv/.venv)")
    p.add_argument("--tasks", nargs="+", default=["open_drawer"],
                    help="Keys in configs/robocasa_tasks.yaml, or 'all'")
    p.add_argument("--env_prefix", default="robocasa_panda_omron", help="RoboCasa sim group name")
    p.add_argument("--env_suffix", default="PandaOmron_Env", help="Suffix appended to each env_class")
    p.add_argument("--policy_host", default="127.0.0.1")
    p.add_argument("--policy_port", type=int, default=5555)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--results_root", default="results/raw_groot/robocasa")
    p.add_argument("--configs_dir", default="configs")
    return p.parse_args()


def extract_success_rate(stdout: str) -> float | None:
    for pattern in _SUCCESS_RATE_PATTERNS:
        m = pattern.search(stdout)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val
    return None


def run_task(args: argparse.Namespace, task_id: str, task_cfg: dict, robocasa_python: str, log_dir: Path) -> dict:
    env_name = f"{args.env_prefix}/{task_cfg['env_class']}_{args.env_suffix}"
    max_steps = task_cfg.get("max_episode_steps", 500)

    cmd = [
        robocasa_python,
        "gr00t/eval/rollout_policy.py",
        "--n_episodes", str(args.n_episodes),
        "--policy_client_host", args.policy_host,
        "--policy_client_port", str(args.policy_port),
        "--max_episode_steps", str(max_steps),
        "--env_name", env_name,
        "--n_action_steps", str(args.n_action_steps),
        "--n_envs", str(args.n_envs),
    ]

    print(f"\n=== {task_id} ({env_name}) ===")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, cwd=args.gr00t_root, capture_output=True, text=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{task_id}.stdout.log").write_text(proc.stdout)
    (log_dir / f"{task_id}.stderr.log").write_text(proc.stderr)

    success_rate = extract_success_rate(proc.stdout)
    ok = proc.returncode == 0

    if not ok:
        print(f"  ✗ rollout_policy.py exited with code {proc.returncode} — see {log_dir}/{task_id}.stderr.log")
    elif success_rate is None:
        print(f"  ⚠ could not parse success rate from stdout — see {log_dir}/{task_id}.stdout.log")
    else:
        print(f"  → success_rate={success_rate:.2%}")

    return {
        "task_id": task_id,
        "env_name": env_name,
        "n_episodes": args.n_episodes,
        "returncode": proc.returncode,
        "success_rate": success_rate,
    }


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "robocasa_tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["robocasa_tasks"]

    task_ids = list(tasks_cfg.keys()) if args.tasks == ["all"] else args.tasks

    robocasa_venv = args.robocasa_venv or f"{args.gr00t_root}/gr00t/eval/sim/robocasa/robocasa_uv/.venv"
    robocasa_python = f"{robocasa_venv}/bin/python"
    if not Path(robocasa_python).exists():
        print(f"ERROR: RoboCasa venv python not found at {robocasa_python}")
        print("Run scripts/setup_robocasa_env.sh first (or the robocasa-client compose service).")
        sys.exit(1)

    results_dir = Path(args.results_root)
    log_dir = results_dir / "logs"

    results = [run_task(args, tid, tasks_cfg[tid], robocasa_python, log_dir) for tid in task_ids]

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Task':28} {'Success Rate':13} {'Status':8}")
    print("-" * 52)
    for r in results:
        sr = f"{r['success_rate']:.2%}" if r["success_rate"] is not None else "n/a"
        status = "ok" if r["returncode"] == 0 else "FAILED"
        print(f"{r['task_id']:28} {sr:13} {status:8}")

    print(f"\nResults saved → {results_dir / 'results.json'}")


if __name__ == "__main__":
    main()
