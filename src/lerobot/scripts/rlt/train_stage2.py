# Copyright 2026 The RL-Token-Pi05 authors.
# SPDX-License-Identifier: Apache-2.0
"""Train/resume the reference Stage-2 learner on normalized replay, without hardware."""

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.pi05_rlt.learning import RLTLearner, RLTLearnerConfig
from lerobot.policies.pi05_rlt.replay import ReplayBuffer
from lerobot.policies.pi05_rlt.training import load_stage1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    initial = p.add_mutually_exclusive_group(required=True)
    initial.add_argument("--stage1_checkpoint")
    initial.add_argument("--resume_from")
    p.add_argument(
        "--replay_path", required=True, help="NPZ ReplayBuffer snapshot with per-step source/validity labels"
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--steps", type=int, default=1000, help="Critic updates (actor updates use the delayed ratio)"
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="One real learner update; never opens hardware")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.save_every) < 1:
        p.error("Steps, batch size and save frequency must be positive")
    torch.manual_seed(args.seed)
    out = Path(args.output_dir).expanduser()
    if out.exists() and any(out.iterdir()) and not args.resume_from:
        raise FileExistsError("Use a fresh output directory or resume explicitly")
    replay = ReplayBuffer.load(args.replay_path, seed=args.seed)
    if len(replay) < args.batch_size:
        raise ValueError("Replay has fewer transitions than batch_size")
    if args.resume_from:
        learner = RLTLearner.load(args.resume_from, args.device)
    else:
        policy, _stage1 = load_stage1(args.stage1_checkpoint, device=args.device)
        learner = RLTLearner(policy, RLTLearnerConfig())
    out.mkdir(parents=True, exist_ok=True)
    updates = 1 if args.dry_run else args.steps
    with (out / "metrics.jsonl").open("a", buffering=1) as log:
        for index in range(updates):
            metrics = learner.update(replay.sample(args.batch_size))
            log.write(json.dumps(metrics) + "\n")
            if index % 50 == 0 or index + 1 == updates:
                print(json.dumps(metrics), flush=True)
            if (index + 1) % args.save_every == 0 or index + 1 == updates:
                learner.save(
                    out / "last_checkpoint.pt",
                    stage1_checkpoint=args.stage1_checkpoint,
                    replay_path=str(Path(args.replay_path).resolve()),
                )
    print("Stage 2 complete: " + str(out / "last_checkpoint.pt"), flush=True)


if __name__ == "__main__":
    main()
