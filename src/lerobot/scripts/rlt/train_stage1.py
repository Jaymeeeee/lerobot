# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted from scripts/train_rlt_stage1_pi05.py; see pi05_rlt/NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Train the reference RL-token encoder/causal decoder from cached VLA embeddings."""

import argparse
import json
import math
from pathlib import Path

import draccus
import torch
from torch.utils.data import DataLoader

from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.policies.pi05_rlt.training import (
    atomic_torch_save,
    encode_policy_config,
    load_prefix_cache,
    load_rlt_config,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--precomputed_path", required=True, help="Comma-separated cache directories")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--resume_from")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="One training step, including backward/checkpoint; no VLA or robot",
    )
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.save_every) < 1 or args.num_workers < 0:
        p.error("Invalid steps/batch/workers/save interval")
    torch.manual_seed(args.seed)
    config = load_rlt_config(args.config, mode="rlt_training", device=args.device)
    dataset, metas = load_prefix_cache(args.precomputed_path.split(","))
    if dataset.hidden_dim != config.vlm_hidden_dim:
        raise ValueError("Cache hidden dimension differs from RLT configuration")
    if dataset.seq_len > 2048:
        raise ValueError("Cache sequence exceeds reference decoder positional capacity")
    if metas[0].get("pi05_path"):
        if (
            config.pi05_pretrained_path
            and str(Path(config.pi05_pretrained_path).expanduser().resolve()) != metas[0]["pi05_path"]
        ):
            raise ValueError("RLT backbone path differs from the cache source")
        config.pi05_pretrained_path = metas[0]["pi05_path"]
    if "pi05_config" in metas[0]:
        from lerobot.configs import PreTrainedConfig

        base = draccus.decode(PreTrainedConfig, metas[0]["pi05_config"])
        if not config.input_features:
            config.input_features = base.input_features
        if not config.output_features:
            config.output_features = base.output_features
        for key in ("chunk_size", "max_action_dim", "max_state_dim", "image_resolution"):
            if getattr(config, key) != getattr(base, key):
                raise ValueError(f"RLT/VLA configuration mismatch: {key}")
        config.text_tokenizer_name = base.text_tokenizer_name
        config.normalization_mapping = base.normalization_mapping
        config.validate_backbone_config(base)
    config.validate_features()
    config.rlt_total_steps = args.steps
    provenance = {k: v for k, v in metas[0].items() if k not in ("root", "n_samples")}
    out = Path(args.output_dir).expanduser()
    if out.exists() and any(out.iterdir()) and not args.resume_from:
        raise FileExistsError("Output is not empty; use a new directory or --resume_from")
    out.mkdir(parents=True, exist_ok=True)
    policy = PI05RLTPolicy(config).to(config.device).train()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=config.rlt_lr, weight_decay=config.rlt_weight_decay
    )

    def schedule(step):
        if step < config.rlt_warmup_steps:
            return (step + 1) / max(config.rlt_warmup_steps, 1)
        progress = (step - config.rlt_warmup_steps) / max(args.steps - config.rlt_warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    device = torch.device(config.device)
    enabled = args.use_amp and device.type == "cuda"
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)
    start, best = 0, float("inf")
    if args.resume_from:
        data = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        if data.get("kind") != "pi05_rlt_stage1" or data["provenance"] != provenance:
            raise ValueError("Resume checkpoint and prefix-cache provenance differ")
        if data["policy_config"] != encode_policy_config(config):
            raise ValueError("Resume configuration/schedule differs; use the original configuration")
        policy.load_state_dict(data["policy"], strict=True)
        optimizer.load_state_dict(data["optimizer"])
        scheduler.load_state_dict(data["scheduler"])
        scaler.load_state_dict(data["scaler"])
        torch.set_rng_state(data["rng_state"])
        start, best = int(data["step"]), float(data["best_loss"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)
    end = min(start + 1, args.steps) if args.dry_run else args.steps
    if end <= start:
        raise ValueError("No training steps remain")
    with (out / "metrics.jsonl").open("a", buffering=1) as log:
        for step in range(start, end):
            try:
                embeddings, mask = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                embeddings, mask = next(iterator)
            batch = {
                "vlm_embeddings": embeddings.to(device, dtype=torch.float32),
                "prefix_mask": mask.to(device),
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reconstruction loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            value = float(loss.detach())
            metrics = {
                "step": step + 1,
                "loss": value,
                "grad_norm": float(norm),
                "lr": optimizer.param_groups[0]["lr"],
            }
            log.write(json.dumps(metrics) + "\n")
            if step % 50 == 0 or step + 1 == end:
                print(json.dumps(metrics), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == end:
                improved = value < best
                best = min(value, best)
                data = {
                    "schema_version": 1,
                    "kind": "pi05_rlt_stage1",
                    "image_only": True,
                    "policy_config": encode_policy_config(config),
                    "policy": policy.state_dict(),
                    "encoder_state_dict": policy.rlt_encoder.state_dict(),
                    "decoder_state_dict": policy.rlt_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step + 1,
                    "best_loss": best,
                    "provenance": provenance,
                    "rng_state": torch.get_rng_state(),
                    "resume_exact": False,
                    "resume_note": "DataLoader sampler position is not checkpointed",
                }
                if improved:
                    atomic_torch_save(data, out / "best_checkpoint.pt")
                atomic_torch_save(data, out / "last_checkpoint.pt")
    print("Stage 1 complete: " + str(out / "last_checkpoint.pt"), flush=True)


if __name__ == "__main__":
    main()
