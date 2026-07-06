"""
Phase 4 — Zero-shot evaluation of CompSteer on all four splits.

Runs CompSteer and baselines in ManiSkill across the full (embodiment, task)
evaluation matrix. Designed to run on HPC with a single GPU per job.

Usage:
    # Full eval (all splits, all baselines)
    python scripts/06_eval_compsteer.py --backbone groot

    # Single split
    python scripts/06_eval_compsteer.py --splits D --backbone groot

    # Override model
    python scripts/06_eval_compsteer.py --model_path /path/to/local/groot_checkpoint

SLURM usage:
    sbatch slurm/06_eval_compsteer.slurm
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.eval.evaluator import CompSteerEvaluator
from compsteer.steering.factorize import FactorizationResult
from compsteer.steering.vector_library import SteeringLibrary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CompSteer on all splits")
    p.add_argument("--backbone",        default="groot", choices=["groot", "groot_1b", "rdt1b"])
    p.add_argument("--model_path",      default="nvidia/GR00T-N1.6-3B")  # NOT N1-2B, see scripts/run_groot_server.sh
    p.add_argument("--library_root",    default="steering_library")
    p.add_argument("--rank",            type=int,   default=16)
    p.add_argument("--factorize_method",default="svd", choices=["svd", "nmf", "learned"])
    p.add_argument("--splits",          nargs="+",  default=["A", "B", "C", "D"],
                   choices=["A", "B", "C", "D", "baseline"])
    p.add_argument("--num_episodes",    type=int,   default=50)
    p.add_argument("--num_envs",        type=int,   default=8)
    p.add_argument("--injection_mode",  default="hidden", choices=["hidden", "velocity"])
    p.add_argument("--schedule",        default="cosine",
                   choices=["constant", "cosine", "linear", "early_only"])
    p.add_argument("--schedule_peak",   type=float, default=1.0)
    p.add_argument("--alpha",           type=float, default=1.0)
    p.add_argument("--beta",            type=float, default=1.0)
    p.add_argument("--eval_seed",       type=int,   default=100)
    p.add_argument("--results_root",    default="results/eval")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--render_videos",   action="store_true")
    p.add_argument("--configs_dir",     default="configs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "embodiments.yaml") as f:
        embodiments_cfg = yaml.safe_load(f)["embodiments"]
    with open(cfg_dir / "tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["tasks"]

    # ── Load library + factorization ──────────────────────────────────────
    library = SteeringLibrary.load(args.library_root)
    print(f"Loaded: {library}")

    fact_path = (Path(args.library_root) / "factorizations"
                 / args.factorize_method / f"rank_{args.rank}.pt")
    if not fact_path.exists():
        print(f"ERROR: factorization not found at {fact_path}")
        print("Run scripts/04_verify_linearity.py first.")
        sys.exit(1)

    factorization = FactorizationResult.load(fact_path)
    print(f"Factorization: rank={args.rank}, method={args.factorize_method}, "
          f"residual={factorization.residual:.4f}")

    # ── Run evaluation ────────────────────────────────────────────────────
    results_dir = (Path(args.results_root) / args.backbone
                   / f"{args.factorize_method}_rank{args.rank}"
                   / f"schedule_{args.schedule}")
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluator = CompSteerEvaluator(
        embodiments_cfg=embodiments_cfg,
        tasks_cfg=tasks_cfg,
        factorization=factorization,
        library=library,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        eval_seed=args.eval_seed,
        backbone=args.backbone,
        model_path=args.model_path,
        injection_mode=args.injection_mode,
        compose_alpha=args.alpha,
        compose_beta=args.beta,
        schedule=args.schedule,
        schedule_peak=args.schedule_peak,
        device=args.device,
        render_videos=args.render_videos,
        video_dir=str(results_dir / "videos") if args.render_videos else None,
    )

    print(f"\nRunning splits: {args.splits}")
    results = evaluator.run_all_splits()

    # Filter to requested splits
    filtered = [r for r in results if r.split in args.splits or r.split == "baseline"]

    # Save + display
    CompSteerEvaluator.save_results(filtered, results_dir)
    CompSteerEvaluator.print_table(filtered)

    print(f"\nResults saved → {results_dir}")


if __name__ == "__main__":
    main()
