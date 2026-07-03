"""
Phase 3 — Train lightweight retrieval encoders.

Trains:
  f_emb  : robot_spec → Δe  (EmbodimentEncoder)
  g_lang : task_text  → Δt  (TaskEncoder)

These encoders enable generalisation to novel (embodiment, task) pairs
not in the steering library, completing the full zero-shot CompSteer pipeline.

Usage:
    python scripts/05_train_encoders.py --rank 16
    python scripts/05_train_encoders.py --rank 16 --skip_f_emb  # only g_lang
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.data.embodiment_specs import load_embodiment_specs, build_spec_matrix
from compsteer.steering.encoders import (
    EmbodimentEncoder,
    TaskEncoder,
    train_embodiment_encoder,
    train_task_encoder,
)
from compsteer.steering.factorize import FactorizationResult
from compsteer.steering.vector_library import SteeringLibrary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train f_emb and g_lang encoders")
    p.add_argument("--library_root",    default="steering_library")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--rank",            type=int, default=16)
    p.add_argument("--factorize_method",default="svd", choices=["svd", "nmf", "learned"])
    p.add_argument("--f_emb_lr",        type=float, default=1e-3)
    p.add_argument("--f_emb_epochs",    type=int,   default=200)
    p.add_argument("--g_lang_lr",       type=float, default=5e-4)
    p.add_argument("--g_lang_epochs",   type=int,   default=100)
    p.add_argument("--t5_model",        default="t5-base")
    p.add_argument("--skip_f_emb",      action="store_true")
    p.add_argument("--skip_g_lang",     action="store_true")
    p.add_argument("--device",          default="cuda")
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
    print(f"Library: {library}")

    fact_path = (Path(args.library_root) / "factorizations"
                 / args.factorize_method / f"rank_{args.rank}.pt")
    if not fact_path.exists():
        print(f"Factorization not found at {fact_path}")
        print("Run scripts/04_verify_linearity.py first.")
        sys.exit(1)

    factorization = FactorizationResult.load(fact_path)
    print(f"Factorization: rank={factorization.rank}, method={factorization.method}")

    encoder_dir = Path(args.checkpoint_root) / "encoders" / f"rank_{args.rank}"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    # ── Train f_emb ───────────────────────────────────────────────────────
    if not args.skip_f_emb:
        print(f"\n{'='*50}")
        print("Training EmbodimentEncoder (f_emb)")
        print(f"{'='*50}")

        # Load embodiment specs for training embodiments
        specs = load_embodiment_specs(cfg_dir / "embodiments.yaml")
        train_emb_ids = factorization.embodiment_ids

        spec_matrix = build_spec_matrix(specs, train_emb_ids)   # (N_e, spec_dim)
        target_E    = factorization.E                            # (N_e, k)

        print(f"  Training on {len(train_emb_ids)} embodiments: {train_emb_ids}")
        print(f"  Spec vector dim: {spec_matrix.shape[1]}")

        from compsteer.data.embodiment_specs import EmbodimentSpec
        spec_dim = EmbodimentSpec.vector_dim()

        encoder = EmbodimentEncoder(
            input_dim=spec_dim,
            hidden_dims=[128, 128],
            output_dim=args.rank,
        )

        encoder = train_embodiment_encoder(
            encoder=encoder,
            spec_vectors=spec_matrix,
            target_e_vectors=target_E,
            lr=args.f_emb_lr,
            epochs=args.f_emb_epochs,
            device=args.device,
            save_path=encoder_dir / "f_emb.pt",
        )

        # Quick eval: predict on training set
        with torch.no_grad():
            pred = encoder(spec_matrix)
            mse = (pred - target_E).pow(2).mean().item()
        print(f"  f_emb train MSE: {mse:.6f}")

    # ── Train g_lang ──────────────────────────────────────────────────────
    if not args.skip_g_lang:
        print(f"\n{'='*50}")
        print("Training TaskEncoder (g_lang)")
        print(f"{'='*50}")

        train_task_ids = factorization.task_ids
        task_descriptions = [tasks_cfg[t]["description"] for t in train_task_ids]
        target_T = factorization.T   # (N_t, k)

        print(f"  Training on {len(train_task_ids)} tasks: {train_task_ids}")

        encoder_lang = TaskEncoder(
            t5_model=args.t5_model,
            output_dim=args.rank,
            freeze_t5=True,
        )

        encoder_lang = train_task_encoder(
            encoder=encoder_lang,
            task_descriptions=task_descriptions,
            target_t_vectors=target_T,
            lr=args.g_lang_lr,
            epochs=args.g_lang_epochs,
            device=args.device,
            save_path=encoder_dir / "g_lang.pt",
        )

        # Quick eval
        with torch.no_grad():
            pred = encoder_lang(task_descriptions)
            mse  = (pred.cpu() - target_T).pow(2).mean().item()
        print(f"  g_lang train MSE: {mse:.6f}")

    print(f"\nEncoders saved → {encoder_dir}")


if __name__ == "__main__":
    main()
