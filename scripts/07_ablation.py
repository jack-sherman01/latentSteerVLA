"""
Phase 5 — Ablation study sweep.

Sweeps over all ablation dimensions defined in configs/compsteer.yaml:
  - Factorization methods: svd, nmf, learned
  - Ranks: 4, 8, 16, 32
  - Library sizes: 5, 10, 20, 30 training pairs
  - Injection schedules: constant, cosine, linear, early_only
  - Retrieval modes: nearest_neighbor, f_emb_encoder

Each condition is evaluated on Split D (new × new, the key zero-shot split).
Results are saved to results/ablation/ in a structured format for easy plotting.

Usage:
    python scripts/07_ablation.py --ablation_type rank
    python scripts/07_ablation.py --ablation_type schedule
    python scripts/07_ablation.py --ablation_type all  # runs everything (slow)

SLURM usage:
    sbatch slurm/06_eval_compsteer.slurm --ablation_type rank
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.eval.evaluator import CompSteerEvaluator, EvalResult
from compsteer.steering.factorize import FactorizationResult, factorize_library
from compsteer.steering.vector_library import SteeringLibrary


ABLATION_GRID = {
    "rank": {
        "param":  "rank",
        "values": [4, 8, 16, 32],
        "fixed":  {"factorize_method": "svd", "schedule": "cosine"},
    },
    "factorize_method": {
        "param":  "factorize_method",
        "values": ["svd", "nmf", "learned"],
        "fixed":  {"rank": 16, "schedule": "cosine"},
    },
    "schedule": {
        "param":  "schedule",
        "values": ["constant", "cosine", "linear", "early_only"],
        "fixed":  {"rank": 16, "factorize_method": "svd"},
    },
    "library_size": {
        "param":  "library_size",
        "values": [5, 10, 20, 30],
        "fixed":  {"rank": 16, "factorize_method": "svd", "schedule": "cosine"},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation sweeps for CompSteer")
    p.add_argument("--ablation_type",   required=True,
                   choices=["rank", "factorize_method", "schedule", "library_size", "all"])
    p.add_argument("--backbone",        default="groot", choices=["groot", "groot_1b", "rdt1b"])
    p.add_argument("--model_path",      default="nvidia/GR00T-N1-2B")
    p.add_argument("--library_root",    default="steering_library")
    p.add_argument("--num_episodes",    type=int,   default=20,
                   help="Episodes per cell (use fewer for faster ablation sweep)")
    p.add_argument("--num_envs",        type=int,   default=8)
    p.add_argument("--eval_seed",       type=int,   default=200)
    p.add_argument("--results_root",    default="results/ablation")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--configs_dir",     default="configs")
    return p.parse_args()


def run_ablation_condition(
    library: SteeringLibrary,
    embodiments_cfg: dict,
    tasks_cfg: dict,
    condition: dict,
    num_episodes: int,
    backbone: str,
    model_path: str,
    results_dir: Path,
    device: str,
) -> list[EvalResult]:
    """Run one ablation condition and return split-D results."""
    rank            = condition.get("rank", 16)
    factorize_method= condition.get("factorize_method", "svd")
    schedule        = condition.get("schedule", "cosine")
    library_size    = condition.get("library_size", None)

    # Subsample library if library_size ablation
    active_library = library
    if library_size is not None and library_size < library.num_pairs():
        # Take the first `library_size` pairs (sorted for reproducibility)
        active_library = SteeringLibrary()
        for (emb, task), vec in sorted(library.vectors.items())[:library_size]:
            active_library.add(emb, task, vec)
        # Re-populate metadata
        active_library.embodiment_ids = list(set(e for e, _ in sorted(library.vectors.keys())[:library_size]))
        active_library.task_ids       = list(set(t for _, t in sorted(library.vectors.keys())[:library_size]))

    # Re-factorize with this condition's parameters
    factorization = factorize_library(
        library=active_library,
        rank=rank,
        method=factorize_method,
        device=device,
    )

    evaluator = CompSteerEvaluator(
        embodiments_cfg=embodiments_cfg,
        tasks_cfg=tasks_cfg,
        factorization=factorization,
        library=active_library,
        num_episodes=num_episodes,
        num_envs=1,
        backbone=backbone,
        model_path=model_path,
        schedule=schedule,
        device=device,
    )

    # Only run Split D (new × new) for speed
    test_embodiments = [k for k, v in embodiments_cfg.items() if v["split"] == "test"]
    test_tasks       = [k for k, v in tasks_cfg.items()       if v["split"] == "test"]

    results = []
    for emb in test_embodiments:
        for task in test_tasks:
            r = evaluator.run_single(emb, task, method="compsteer", split="D")
            results.append(r)

    return results


def main():
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "embodiments.yaml") as f:
        embodiments_cfg = yaml.safe_load(f)["embodiments"]
    with open(cfg_dir / "tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["tasks"]

    library = SteeringLibrary.load(args.library_root)

    ablation_types = list(ABLATION_GRID.keys()) if args.ablation_type == "all" else [args.ablation_type]

    all_ablation_results: dict = {}

    for abl_type in ablation_types:
        grid = ABLATION_GRID[abl_type]
        param = grid["param"]
        values = grid["values"]
        fixed = grid["fixed"]

        print(f"\n{'='*60}")
        print(f"Ablation: {abl_type}  (sweep {param} over {values})")
        print(f"Fixed: {fixed}")
        print(f"{'='*60}")

        abl_results: dict[str, float] = {}

        for val in values:
            condition = {**fixed, param: val}
            cond_name = f"{param}={val}"
            print(f"\n  Condition: {cond_name}")

            results_dir = Path(args.results_root) / abl_type / str(val)
            results_dir.mkdir(parents=True, exist_ok=True)

            split_d_results = run_ablation_condition(
                library=library,
                embodiments_cfg=embodiments_cfg,
                tasks_cfg=tasks_cfg,
                condition=condition,
                num_episodes=args.num_episodes,
                backbone=args.backbone,
                model_path=args.model_path,
                results_dir=results_dir,
                device=args.device,
            )

            sr = sum(r.success_rate for r in split_d_results) / max(len(split_d_results), 1)
            abl_results[cond_name] = round(sr, 4)
            print(f"  Split D mean success rate: {sr:.2%}")

            CompSteerEvaluator.save_results(split_d_results, results_dir)

        all_ablation_results[abl_type] = abl_results

    # Save summary
    summary_path = Path(args.results_root) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_ablation_results, f, indent=2)
    print(f"\nAblation summary → {summary_path}")

    # Print summary table
    print("\n" + "="*50)
    for abl_type, results in all_ablation_results.items():
        print(f"\n{abl_type}:")
        for cond, sr in results.items():
            print(f"  {cond:30s}: {sr:.2%}")


if __name__ == "__main__":
    main()
