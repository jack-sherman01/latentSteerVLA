"""
Phase 2b — Verify linearity hypothesis (go/no-go gate).

Tests Hypothesis H1:
    V ≈ E · T^T  (additive model: v_ij ≈ Δe_i + Δt_j)

Runs SVD, NMF, and learned factorization at multiple ranks k and reports:
  - Relative Frobenius residual ‖V - V_hat‖_F / ‖V‖_F  (lower = more linear)
  - Percentage of variance explained (higher = better)
  - Per-pair residuals (to identify outlier pairs)

Decision rule: if rank-16 SVD gives residual < 0.20 → proceed with CompSteer.

Usage:
    python scripts/04_verify_linearity.py
    python scripts/04_verify_linearity.py --ranks 4 8 16 32 64
    python scripts/04_verify_linearity.py --methods svd nmf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.steering.factorize import factorize_library, FactorizationResult
from compsteer.steering.vector_library import SteeringLibrary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify linearity of steering library")
    p.add_argument("--library_root",  default="steering_library")
    p.add_argument("--results_root",  default="results/linearity")
    p.add_argument("--ranks",         type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--methods",       nargs="+", default=["svd"],
                   choices=["svd", "nmf", "learned"])
    p.add_argument("--learned_epochs", type=int, default=200)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    library = SteeringLibrary.load(args.library_root)
    print(f"Loaded steering library: {library}")

    if library.num_pairs() < 2:
        print("ERROR: need at least 2 pairs to test linearity")
        sys.exit(1)

    V, keys = library.to_matrix()
    print(f"\nSteering matrix V: {V.shape}  (N_pairs × latent_dim)")
    print(f"  V norm: {V.norm():.4f}")
    print(f"  V mean: {V.mean():.4f}")
    print(f"  V std:  {V.std():.4f}")

    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for method in args.methods:
        all_results[method] = {}
        print(f"\n{'='*50}")
        print(f"Method: {method}")
        print(f"{'='*50}")

        for rank in args.ranks:
            print(f"\n  Rank k = {rank}:")

            factorization = factorize_library(
                library=library,
                rank=rank,
                method=method,
                device=args.device,
                learned_epochs=args.learned_epochs if method == "learned" else 200,
            )

            residual = factorization.residual
            variance_explained = 1.0 - residual ** 2

            print(f"    Relative residual:     {residual:.4f}")
            print(f"    Variance explained:    {variance_explained:.2%}")

            # Per-pair residuals
            E = factorization.E     # (N_e, k)
            T = factorization.T     # (N_t, k)
            e2i = {e: i for i, e in enumerate(factorization.embodiment_ids)}
            t2i = {t: i for i, t in enumerate(factorization.task_ids)}

            per_pair_residuals = {}
            for (emb, task), vec in library.vectors.items():
                v_hat = E[e2i[emb]] + T[t2i[task]]
                err = (vec - v_hat).norm().item() / (vec.norm().item() + 1e-8)
                per_pair_residuals[f"{emb}__{task}"] = round(err, 4)

            # Sort by residual to identify worst-fit pairs
            sorted_pairs = sorted(per_pair_residuals.items(), key=lambda x: x[1], reverse=True)
            print(f"    Worst-fit pairs:")
            for pair_name, err in sorted_pairs[:3]:
                print(f"      {pair_name}: {err:.4f}")

            all_results[method][f"rank_{rank}"] = {
                "relative_residual": round(residual, 6),
                "variance_explained": round(float(variance_explained), 6),
                "per_pair_residuals": per_pair_residuals,
            }

            # Save factorization for further use
            fact_path = Path(args.library_root) / "factorizations" / method / f"rank_{rank}.pt"
            fact_path.parent.mkdir(parents=True, exist_ok=True)
            factorization.save(fact_path)

    # Save all results to JSON
    with open(results_dir / "linearity_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_dir / 'linearity_results.json'}")

    # ── Go/No-Go decision ─────────────────────────────────────────────────
    print("\n" + "="*50)
    print("GO / NO-GO DECISION")
    print("="*50)
    svd_r16 = all_results.get("svd", {}).get("rank_16", {}).get("relative_residual", None)
    if svd_r16 is not None:
        if svd_r16 < 0.20:
            print(f"✓ GO: SVD rank-16 residual = {svd_r16:.4f} < 0.20")
            print("  Linear structure confirmed. Proceed with CompSteer.")
        elif svd_r16 < 0.35:
            print(f"~ CONDITIONAL GO: SVD rank-16 residual = {svd_r16:.4f}")
            print("  Moderate linearity. Try learned factorization for better fit.")
        else:
            print(f"✗ NO-GO: SVD rank-16 residual = {svd_r16:.4f} >= 0.35")
            print("  Poor linearity. Consider: more pairs, different latent space, or")
            print("  non-linear composition (AdaptiveComposer).")
    else:
        print("SVD rank-16 result not available.")

    # ── Plot (if matplotlib available) ───────────────────────────────────
    try:
        _plot_linearity_curve(all_results, args.ranks, results_dir)
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")


def _plot_linearity_curve(all_results: dict, ranks: list[int], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for method, method_results in all_results.items():
        residuals = [method_results.get(f"rank_{r}", {}).get("relative_residual", None) for r in ranks]
        var_exp   = [method_results.get(f"rank_{r}", {}).get("variance_explained", None) for r in ranks]

        valid_r = [(r, v) for r, v in zip(ranks, residuals) if v is not None]
        valid_v = [(r, v) for r, v in zip(ranks, var_exp) if v is not None]

        if valid_r:
            ax1.plot([x[0] for x in valid_r], [x[1] for x in valid_r], marker="o", label=method)
        if valid_v:
            ax2.plot([x[0] for x in valid_v], [x[1] for x in valid_v], marker="o", label=method)

    ax1.axhline(y=0.20, color="green",  linestyle="--", alpha=0.7, label="Go threshold (0.20)")
    ax1.axhline(y=0.35, color="red",    linestyle="--", alpha=0.7, label="No-go threshold (0.35)")
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("Relative Frobenius Residual")
    ax1.set_title("Linearity Test: Residual vs Rank")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Rank k")
    ax2.set_ylabel("Variance Explained")
    ax2.set_title("Linearity Test: Variance Explained vs Rank")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "linearity_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {output_dir / 'linearity_curve.png'}")


if __name__ == "__main__":
    main()
