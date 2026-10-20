# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted from scripts/precompute_pi05_embeddings.py; see pi05_rlt/NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Cache frozen PI05 image-prefix tokens using the checkpoint's own processors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_rlt.training import encode_policy_config
from lerobot.policies.pi05_rlt.vla_compat import extract_embeddings
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pi05_path", required=True, help="Local PI05 pretrained_model directory")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_repo_id", default="local/piper")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", help="Local tokenizer; otherwise use the saved processor")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, nargs="+", help="Optional episode IDs, for cache sharding")
    p.add_argument("--max_frames", type=int, help="Debug cap, not a complete cache of the input dataset")
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_frames is not None and args.max_frames < 1):
        p.error("Invalid batch/workers/frame count")
    path = Path(args.pi05_path).expanduser().resolve()
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError("Expected a local config.json and model.safetensors")
    cfg = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    if cfg.type != "pi05":
        raise ValueError("Precompute requires a PI05 checkpoint")
    if any(
        getattr(cfg, key, False)
        for key in ("use_relative_actions", "use_visual_memory", "use_proprioceptive_memory")
    ):
        raise ValueError("Reference RLT requires absolute-action PI05 without observation memory")
    cfg.device = args.device
    if args.tokenizer_path:
        cfg.text_tokenizer_name = str(Path(args.tokenizer_path).expanduser().resolve())
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=Path(args.dataset_root).expanduser(),
        episodes=args.episodes,
        video_backend="pyav",
    )
    if not cfg.image_features or any(key not in dataset.meta.features for key in cfg.image_features):
        raise ValueError("Dataset does not contain every camera feature required by PI05")
    output = Path(args.output_dir).expanduser()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prefix cache output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    weights = path / "model.safetensors"
    signature = hashlib.sha256(
        (path / "config.json").read_bytes()
        + str((weights.stat().st_size, weights.stat().st_mtime_ns)).encode()
    ).hexdigest()
    n = min(len(dataset), args.max_frames) if args.max_frames else len(dataset)
    if n < 1:
        raise ValueError("Dataset is empty")
    meta = {
        "schema_version": 2,
        "complete": False,
        "n_samples": n,
        "dtype": "float16",
        "has_mask": True,
        "image_only": True,
        "embedding_stage": "post_transformer",
        "language_positions_dropped": True,
        "pi05_path": str(path),
        "checkpoint_signature": signature,
        "pi05_config": encode_policy_config(cfg),
        "checkpoint_image_features": list(cfg.image_features),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "episodes": args.episodes,
        "max_frames": args.max_frames,
    }

    def save_meta():
        temp = output / "meta.json.tmp"
        temp.write_text(json.dumps(meta, indent=2))
        temp.replace(output / "meta.json")

    save_meta()
    policy = (
        PI05Policy.from_pretrained(path, config=cfg, local_files_only=True)
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    # Pretrained normalization and tokenizer must agree with the frozen VLA.
    overrides = {"device_processor": {"device": args.device}}
    if args.tokenizer_path:
        overrides["tokenizer_processor"] = {"tokenizer_name": cfg.text_tokenizer_name}
    pre, _ = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides=overrides)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    arrays = None
    written = 0
    with torch.inference_mode():
        for batch in loader:
            count = min(len(batch["observation.state"]), n - written)
            if count <= 0:
                break
            processed = pre(batch)
            images, img_masks = policy._preprocess_images(processed)
            actions = torch.zeros(len(images[0]), cfg.chunk_size, cfg.max_action_dim, device=args.device)
            prefix, _, mask = extract_embeddings(
                policy,
                images,
                img_masks,
                processed[OBS_LANGUAGE_TOKENS],
                processed[OBS_LANGUAGE_ATTENTION_MASK],
                actions,
                chunk_size=cfg.chunk_size,
                max_action_dim=cfg.max_action_dim,
                noise=torch.zeros_like(actions),
                time=torch.ones(len(actions), device=args.device),
                image_only=True,
            )
            if not torch.isfinite(prefix).all() or not mask.any(-1).all():
                raise ValueError("Invalid image-prefix embeddings")
            if arrays is None:
                meta.update(
                    seq_len=prefix.shape[1],
                    vlm_hidden_dim=prefix.shape[2],
                    state_dim=processed["observation.state"].shape[-1],
                    action_dim=processed["action"].shape[-1],
                    has_frame_index=True,
                )
                arrays = {
                    "prefix": np.memmap(
                        output / "prefix_out.mmap", dtype=np.float16, mode="w+", shape=(n, *prefix.shape[1:])
                    ),
                    "mask": np.memmap(
                        output / "prefix_mask.mmap", dtype=np.bool_, mode="w+", shape=(n, prefix.shape[1])
                    ),
                    "proprio": np.memmap(
                        output / "proprio.mmap", dtype=np.float32, mode="w+", shape=(n, meta["state_dim"])
                    ),
                    "actions": np.memmap(
                        output / "actions.mmap", dtype=np.float32, mode="w+", shape=(n, meta["action_dim"])
                    ),
                    "episode_index": np.memmap(
                        output / "episode_index.mmap", dtype=np.int64, mode="w+", shape=(n,)
                    ),
                    "frame_index": np.memmap(
                        output / "frame_index.mmap", dtype=np.int64, mode="w+", shape=(n,)
                    ),
                }
            sl = slice(written, written + count)
            arrays["prefix"][sl] = prefix[:count].float().cpu().numpy()
            arrays["mask"][sl] = mask[:count].cpu().numpy()
            arrays["proprio"][sl] = processed["observation.state"][:count].float().cpu().numpy()
            arrays["actions"][sl] = processed["action"][:count].float().cpu().numpy()
            arrays["episode_index"][sl] = batch["episode_index"][:count].reshape(-1).cpu().numpy()
            arrays["frame_index"][sl] = batch["frame_index"][:count].reshape(-1).cpu().numpy()
            written += count
            if written % 100 == 0 or written == n:
                print(f"Cached {written}/{n} frames", flush=True)
    if written != n:
        raise RuntimeError("Dataset ended before the declared cache length")
    for array in arrays.values():
        array.flush()
    meta["complete"] = True
    save_meta()


if __name__ == "__main__":
    main()
