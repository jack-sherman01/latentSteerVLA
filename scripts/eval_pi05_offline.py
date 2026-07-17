"""Offline sanity check for a fine-tuned π0.5 checkpoint.

Replays dataset episodes through the policy open-loop (feeding it the recorded
observations frame by frame) and reports the error between predicted and
recorded actions. This does not measure closed-loop task success, but it is a
cheap smoke test that the checkpoint learned the demonstrations before
deploying on the robot.

Run inside the pi05-finetune container:

    docker compose run --rm pi05-finetune python scripts/eval_pi05_offline.py \
        --checkpoint outputs/pi05_circular_obj/checkpoints/last/pretrained_model \
        --episodes 45 46 47 48 49

`--checkpoint` also accepts a HF Hub id (e.g. lerobot/pi05_base to eyeball the
un-fine-tuned baseline).
"""

import argparse
from collections import defaultdict

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-loop eval of a pi05 checkpoint on dataset episodes")
    p.add_argument("--checkpoint", required=True, help="Path or HF id of the pretrained_model dir")
    p.add_argument("--repo_id", default="lucazanett/circular_obj_30fps")
    p.add_argument("--episodes", type=int, nargs="+", default=[45, 46, 47, 48, 49],
                   help="Episode indices to replay (default: last 5 of the 50)")
    p.add_argument("--max_frames_per_episode", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    print(f"Loading policy from {args.checkpoint} ...")
    policy = PI05Policy.from_pretrained(args.checkpoint)
    policy.to(args.device).eval()

    dataset = LeRobotDataset(args.repo_id, episodes=args.episodes)
    action_names = dataset.meta.features["action"]["names"]

    # Iterate frames in order; the policy is reset at every episode boundary so
    # its internal action-chunk queue never straddles two episodes.
    errors_by_ep: dict[int, list[np.ndarray]] = defaultdict(list)
    current_ep = None
    for i in range(len(dataset)):
        item = dataset[i]
        ep = int(item["episode_index"])
        if ep != current_ep:
            policy.reset()
            current_ep = ep
        if len(errors_by_ep[ep]) >= args.max_frames_per_episode:
            continue

        batch = {k: v.unsqueeze(0).to(args.device) for k, v in item.items() if isinstance(v, torch.Tensor)}
        batch["task"] = [item["task"]]

        pred = policy.select_action(batch)[0].float().cpu().numpy()
        gt = item["action"].float().cpu().numpy()
        errors_by_ep[ep].append(np.abs(pred - gt))

    for ep, errs in sorted(errors_by_ep.items()):
        errs = np.stack(errs, axis=0)
        print(f"episode {ep:3d}: {errs.shape[0]:4d} frames, "
              f"mean |err| = {errs.mean():.4f}, max |err| = {errs.max():.4f}")

    errors = np.concatenate([np.stack(e) for e in errors_by_ep.values()], axis=0)
    print("\nPer-dimension mean absolute error:")
    for name, err in zip(action_names, errors.mean(axis=0)):
        print(f"  {name:20s} {err:.4f}")
    print(f"\nOverall MAE: {errors.mean():.4f}   MSE: {(errors ** 2).mean():.6f}")


if __name__ == "__main__":
    main()
