# Copyright 2026 The RL-Token-Pi05 authors.
# SPDX-License-Identifier: Apache-2.0
"""Export a trained RLT checkpoint and its PI05 processors to pretrained format."""

import argparse
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05_rlt.learning import RLTLearner
from lerobot.policies.pi05_rlt.training import load_stage1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pi05_path", help="Override frozen VLA location for deployment")
    p.add_argument(
        "--without_processors",
        action="store_true",
        help="Tests/externally processed input only; not a standalone deployment export",
    )
    p.add_argument("--vla_only", action="store_true", help="Export with actor execution disabled")
    args = p.parse_args()
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if data.get("kind") == "pi05_rlt_stage1":
        if not args.vla_only:
            raise ValueError("Stage-1 actor is untrained; export with --vla_only or train Stage 2 first")
        policy, _ = load_stage1(args.checkpoint, mode="inference")
    elif data.get("kind") == "pi05_rlt_stage2":
        policy = RLTLearner.load(args.checkpoint).policy
        policy.config.mode = "inference"
    else:
        raise ValueError("Unsupported checkpoint format")
    if args.pi05_path:
        policy.config.pi05_pretrained_path = str(Path(args.pi05_path).expanduser().resolve())
    policy.config.deploy_vla_only = args.vla_only
    policy.eval().requires_grad_(False)
    out = Path(args.output_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Export output must be empty")
    processors = None
    if not args.without_processors:
        backbone = Path(policy.config.pi05_pretrained_path).expanduser()
        if not (backbone / "config.json").is_file():
            raise FileNotFoundError("A local PI05 checkpoint with processors is required")
        policy.config.validate_backbone_config(
            PreTrainedConfig.from_pretrained(backbone, local_files_only=True)
        )
        processors = make_pre_post_processors(policy.config, pretrained_path=backbone)
    policy.save_pretrained(out)
    if processors:
        for processor in processors:
            processor.save_pretrained(out)
    print(
        "Exported RLT components to "
        + str(out)
        + "; frozen PI05 weights remain at "
        + policy.config.pi05_pretrained_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
