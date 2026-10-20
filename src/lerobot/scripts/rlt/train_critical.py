# Copyright 2026 The RL-Token-Pi05 authors.
# SPDX-License-Identifier: Apache-2.0
"""Train the optional reference critical-phase classifier with episode-disjoint validation."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from lerobot.policies.pi05_rlt.critical import CriticalPhaseClassifier
from lerobot.policies.pi05_rlt.training import atomic_torch_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="NPZ from encode_cache, including labels")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        p.error("epochs/batch_size must be positive")
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(args.seed)
    with np.load(args.features, allow_pickle=False) as data:
        z = torch.from_numpy(data["z_rl"].copy()).float()
        state = torch.from_numpy(data["proprio"].copy()).float()
        labels = torch.from_numpy(data["labels"].copy()).long()
        episodes = data["episode_index"].copy()
    if (
        z.ndim != 2
        or state.ndim != 2
        or labels.shape != (len(z),)
        or len(state) != len(z)
        or episodes.shape != (len(z),)
    ):
        raise ValueError("Classifier data shape mismatch")
    if (
        not torch.isfinite(z).all()
        or not torch.isfinite(state).all()
        or not torch.isin(labels, torch.tensor([0, 1])).all()
    ):
        raise ValueError("Invalid classifier features/labels")
    ids = np.unique(episodes)
    if len(ids) < 2:
        raise ValueError("At least two episodes are required for disjoint validation")
    np.random.default_rng(args.seed).shuffle(ids)
    train_ids = ids[: max(1, int(0.8 * len(ids)))]
    train = torch.from_numpy(np.isin(episodes, train_ids))
    valid = ~train
    counts = torch.bincount(labels[train], minlength=2).float()
    if (counts == 0).any():
        raise ValueError("Training episodes must contain critical and noncritical labels")
    mean = state[train].mean(0)
    std = state[train].std(0, unbiased=False).clamp_min(1e-6)
    norm = (state - mean) / std
    model = CriticalPhaseClassifier(z.shape[-1], state.shape[-1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weight = (counts.sum() / (2 * counts)).to(args.device)
    for _ in range(args.epochs):
        model.train()
        for indices in torch.where(train)[0][torch.randperm(int(train.sum()))].split(args.batch_size):
            logits = model(z[indices].to(args.device), norm[indices].to(args.device))
            loss = functional.cross_entropy(logits, labels[indices].to(args.device), weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(z[valid].to(args.device), norm[valid].to(args.device)).argmax(-1).cpu()
        accuracy = float((prediction == labels[valid]).float().mean())
    atomic_torch_save(
        {
            "classifier_state_dict": model.cpu().state_dict(),
            "z_dim": z.shape[-1],
            "proprio_dim": state.shape[-1],
            "hidden_dim": 256,
            "num_layers": 2,
            "proprio_mean": mean,
            "proprio_std": std,
            "train_episode_ids": train_ids.tolist(),
            "validation_episode_ids": ids[len(train_ids) :].tolist(),
            "validation_accuracy": accuracy,
        },
        output,
    )
    print(f"Critical classifier saved: {output}; held-out accuracy={accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
