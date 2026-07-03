"""
Phase 2a — Extract steering vectors from trained VAEs.

For each (embodiment, task) pair, loads the trained VAE and computes:
    Δv_ij = mean(VAE.encode(target_demos)) - source_mean

All vectors are saved into a SteeringLibrary for factorization.

Usage:
    python scripts/03_extract_vectors.py
    python scripts/03_extract_vectors.py --pairs panda__pick_cube xarm6__stack_cube
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.data.maniskill_collector import load_action_chunks
from compsteer.steering.vector_library import SteeringLibrary
from compsteer.vae.model import AsymmetricVAE
from compsteer.vae.train import load_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract steering vectors from trained VAEs")
    p.add_argument("--pairs",         nargs="*", default=None,
                   help="Specific 'embodiment__task' pairs to process. Default: all in checkpoint_root/vae/")
    p.add_argument("--data_root",         default="data")
    p.add_argument("--checkpoint_root",   default="checkpoints")
    p.add_argument("--library_root",      default="steering_library")
    p.add_argument("--action_horizon",    type=int, default=16)
    p.add_argument("--source_stats_root", default=None,
                   help="Directory with per-embodiment source_stats.pt files")
    p.add_argument("--configs_dir",       default="configs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "embodiments.yaml") as f:
        embodiments_cfg = yaml.safe_load(f)["embodiments"]
    with open(cfg_dir / "tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["tasks"]

    # ── Discover pairs to process ─────────────────────────────────────────
    vae_root = Path(args.checkpoint_root) / "vae"
    if args.pairs is not None:
        pairs = args.pairs
    else:
        pairs = [p.name for p in vae_root.iterdir() if p.is_dir()]

    print(f"Extracting steering vectors for {len(pairs)} pairs...")

    library = SteeringLibrary()

    for pair_name in sorted(pairs):
        parts = pair_name.split("__", 1)
        if len(parts) != 2:
            print(f"  SKIP: cannot parse pair name '{pair_name}'")
            continue

        embodiment_id, task_id = parts

        if embodiment_id not in embodiments_cfg:
            print(f"  SKIP: embodiment '{embodiment_id}' not in config")
            continue
        if task_id not in tasks_cfg:
            print(f"  SKIP: task '{task_id}' not in config")
            continue

        ckpt_path = vae_root / pair_name / "vae_best.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: no VAE checkpoint at {ckpt_path}")
            continue

        # ── Load data ────────────────────────────────────────────────────
        data_dir = Path(args.data_root) / pair_name / "lerobot"
        if not data_dir.exists():
            data_dir = Path(args.data_root) / pair_name / "raw"

        if not data_dir.exists():
            print(f"  SKIP: no data at {data_dir}")
            continue

        target_actions = load_action_chunks(data_dir, action_horizon=args.action_horizon)
        print(f"  {pair_name}: {target_actions.shape[0]} chunks")

        # ── Load VAE + extract vector ─────────────────────────────────────
        vae = load_vae(ckpt_path, device="cpu")

        # Optionally update source stats
        if args.source_stats_root is not None:
            stats_path = Path(args.source_stats_root) / f"{embodiment_id}_stats.pt"
            if stats_path.exists():
                stats = torch.load(stats_path, map_location="cpu")
                vae.update_source_stats(stats["latents"])
                print(f"    Updated source stats from {stats_path}")

        delta_v = vae.extract_steering_vector(target_actions)
        library.add(embodiment_id, task_id, delta_v)
        print(f"    Δv norm = {delta_v.norm().item():.4f}")

    # ── Save library ─────────────────────────────────────────────────────
    lib_path = Path(args.library_root)
    library.save(lib_path)

    print(f"\nExtracted {library.num_pairs()} steering vectors → {lib_path}")
    print(library)


if __name__ == "__main__":
    main()
