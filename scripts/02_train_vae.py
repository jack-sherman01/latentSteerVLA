"""
Phase 1 — Train Asymmetric VAE for one (embodiment, task) pair.

This is the "align" stage from ATE, extended to GR00T's action space.
Designed to run as a SLURM array job over all training pairs.

Usage:
    python scripts/02_train_vae.py --embodiment panda --task pick_cube
    python scripts/02_train_vae.py --embodiment panda --task pick_cube --beta 2.0 --latent_dim 512

SLURM array usage:
    sbatch --array=0-29 slurm/02_train_vae.slurm
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from compsteer.data.maniskill_collector import load_action_chunks
from compsteer.vae.train import train_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train asymmetric VAE for CompSteer")
    p.add_argument("--embodiment",    required=True)
    p.add_argument("--task",          required=True)
    p.add_argument("--data_root",     default="data")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--latent_dim",    type=int,   default=256)
    p.add_argument("--hidden_dims",   type=int,   nargs="+", default=[512, 512])
    p.add_argument("--beta",          type=float, default=1.0)
    p.add_argument("--recon_weight",  type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_epochs",    type=int,   default=100)
    p.add_argument("--action_horizon",type=int,   default=16)
    p.add_argument("--source_stats",  default=None, help="Path to source_stats.pt (optional)")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--configs_dir",   default="configs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dir = Path(args.configs_dir)

    with open(cfg_dir / "embodiments.yaml") as f:
        embodiments_cfg = yaml.safe_load(f)["embodiments"]
    with open(cfg_dir / "tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)["tasks"]

    if args.embodiment not in embodiments_cfg:
        print(f"ERROR: embodiment '{args.embodiment}' not in embodiments.yaml")
        sys.exit(1)
    if args.task not in tasks_cfg:
        print(f"ERROR: task '{args.task}' not in tasks.yaml")
        sys.exit(1)

    emb_cfg = embodiments_cfg[args.embodiment]
    action_dim = emb_cfg["total_action_dim"]
    pair_name  = f"{args.embodiment}__{args.task}"

    # ── Load action chunks ────────────────────────────────────────────────
    data_dir  = Path(args.data_root) / pair_name / "lerobot"
    if not data_dir.exists():
        data_dir = Path(args.data_root) / pair_name / "raw"

    print(f"\nLoading action data from: {data_dir}")
    target_actions = load_action_chunks(
        data_dir=data_dir,
        action_horizon=args.action_horizon,
        stride=1,
    )
    print(f"  Action chunks: {target_actions.shape}  (N, T, action_dim)")

    # ── Load source distribution stats (if provided) ─────────────────────
    source_mean = source_std = None
    if args.source_stats is not None:
        stats = torch.load(args.source_stats, map_location="cpu")
        source_mean = stats["mean"]
        source_std  = stats["std"]
        print(f"  Source stats loaded from: {args.source_stats}")
    else:
        print("  No source stats provided — using N(0,I) as source prior")

    # ── Train VAE ─────────────────────────────────────────────────────────
    ckpt_dir  = Path(args.checkpoint_root) / "vae" / pair_name
    ckpt_path = ckpt_dir / "vae_best.pt"

    print(f"\nTraining VAE:")
    print(f"  action_dim={action_dim}, action_horizon={args.action_horizon}")
    print(f"  latent_dim={args.latent_dim}, hidden_dims={args.hidden_dims}")
    print(f"  beta={args.beta}, lr={args.lr}, epochs={args.num_epochs}")

    vae = train_vae(
        target_actions=target_actions,
        action_dim=action_dim,
        action_horizon=args.action_horizon,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        source_mean=source_mean,
        source_std=source_std,
        beta=args.beta,
        recon_weight=args.recon_weight,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        save_path=ckpt_path,
    )

    print(f"\nVAE training complete → {ckpt_path}")


if __name__ == "__main__":
    main()
