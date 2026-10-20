# Copyright 2026 The RL-Token-Pi05 authors.
# SPDX-License-Identifier: Apache-2.0
"""Encode cached prefixes to z_rl for replay preparation and critical-phase labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.pi05_rlt.training import load_prefix_cache, load_stage1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1_checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--labels", help="Optional binary .npy critical labels, one per cache row")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch_size must be positive")
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(output)
    policy, checkpoint = load_stage1(args.stage1_checkpoint, mode="inference", device=args.device)
    policy.eval()
    dataset, metas = load_prefix_cache([args.cache])
    meta = metas[0]
    if checkpoint["provenance"] != {k: v for k, v in meta.items() if k not in ("root", "n_samples")}:
        raise ValueError("Stage-1 checkpoint and cache provenance differ")
    cache = Path(args.cache).expanduser()
    if not meta.get("has_frame_index"):
        raise ValueError(
            "Use a cache produced by the AgileX precompute command with proprio/actions/frame indices"
        )
    values = []
    with torch.inference_mode():
        for embeddings, mask in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            values.append(
                policy.rlt_encoder(embeddings.to(args.device, dtype=torch.float32), mask.to(args.device))
                .cpu()
                .numpy()
            )
    n = len(dataset)
    payload = {
        "z_rl": np.concatenate(values),
        "proprio": np.memmap(
            cache / "proprio.mmap", dtype=np.float32, mode="r", shape=(n, meta["state_dim"])
        ),
        "actions": np.memmap(
            cache / "actions.mmap", dtype=np.float32, mode="r", shape=(n, meta["action_dim"])
        ),
        "episode_index": np.memmap(cache / "episode_index.mmap", dtype=np.int64, mode="r", shape=(n,)),
        "frame_index": np.memmap(cache / "frame_index.mmap", dtype=np.int64, mode="r", shape=(n,)),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
                    "checkpoint_signature": meta.get("checkpoint_signature"),
                }
            )
        ),
    }
    if args.labels:
        labels = np.load(args.labels, allow_pickle=False)
        if labels.shape != (n,) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Critical labels must be binary and match the cache row count")
        payload["labels"] = labels.astype(np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **payload)


if __name__ == "__main__":
    main()
