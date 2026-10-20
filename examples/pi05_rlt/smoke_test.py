# SPDX-License-Identifier: Apache-2.0
"""CPU integration smoke test with synthetic data; never connects to a robot."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector
from lerobot.policies.pi05_rlt.learning import RLTLearner
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.policies.pi05_rlt.replay import ReplayBuffer, Transition, TransitionSource
from lerobot.policies.pi05_rlt.training import encode_policy_config


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", "lerobot.scripts.rlt." + module, *map(str, args)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    cfg = PI05RLTConfig(
        device="cpu",
        mode="rlt_training",
        vlm_hidden_dim=8,
        rlt_hidden_dim=8,
        rlt_num_heads=2,
        rlt_encoder_layers=1,
        rlt_decoder_layers=1,
        rlt_dropout=0.0,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        chunk_size=6,
        n_action_steps_rl=3,
        rlt_warmup_steps=0,
    )
    (root / "config.json").write_text(json.dumps(encode_policy_config(cfg), indent=2))
    cache = root / "cache"
    cache.mkdir()
    n = 16
    meta = {
        "schema_version": 2,
        "complete": True,
        "image_only": True,
        "has_mask": True,
        "embedding_stage": "post_transformer",
        "dtype": "float16",
        "n_samples": n,
        "seq_len": 4,
        "vlm_hidden_dim": 8,
        "state_dim": 14,
        "action_dim": 14,
        "has_frame_index": True,
        "checkpoint_signature": "synthetic-smoke-data-only",
    }
    (cache / "meta.json").write_text(json.dumps(meta))
    rng.normal(size=(n, 4, 8)).astype(np.float16).tofile(cache / "prefix_out.mmap")
    np.ones((n, 4), dtype=np.bool_).tofile(cache / "prefix_mask.mmap")
    rng.normal(size=(n, 14)).astype(np.float32).tofile(cache / "proprio.mmap")
    rng.normal(size=(n, 14)).astype(np.float32).tofile(cache / "actions.mmap")
    np.repeat(np.arange(4, dtype=np.int64), 4).tofile(cache / "episode_index.mmap")
    np.tile(np.arange(4, dtype=np.int64), 4).tofile(cache / "frame_index.mmap")
    np.save(root / "labels.npy", np.tile([0, 1], n // 2))
    stage1 = root / "stage1"
    common = (
        "--config",
        root / "config.json",
        "--precomputed_path",
        cache,
        "--output_dir",
        stage1,
        "--steps",
        "3",
        "--batch_size",
        "4",
        "--device",
        "cpu",
    )
    run("train_stage1", *common, "--dry_run")
    run("train_stage1", *common, "--resume_from", stage1 / "last_checkpoint.pt")
    run(
        "encode_cache",
        "--stage1_checkpoint",
        stage1 / "last_checkpoint.pt",
        "--cache",
        cache,
        "--output",
        root / "features.npz",
        "--labels",
        root / "labels.npy",
        "--device",
        "cpu",
    )
    run(
        "train_critical",
        "--features",
        root / "features.npz",
        "--output",
        root / "critical.pt",
        "--epochs",
        "2",
        "--batch_size",
        "4",
    )
    detector = CriticalPhaseDetector(root / "critical.pt", "cpu")
    assert detector.update(np.zeros(8), np.zeros(14)) in (True, False, None)
    replay = ReplayBuffer(16, seed=42)
    with np.load(root / "features.npz", allow_pickle=False) as features:
        for i in range(8):
            action = rng.normal(size=(3, 14)).astype(np.float32)
            replay.add(
                Transition(
                    z_rl=features["z_rl"][i],
                    proprio=features["proprio"][i],
                    ref_chunk=action.copy(),
                    action_chunk=action.copy(),
                    rewards=np.array([0, 0, 1], dtype=np.float32),
                    done=True,
                    next_z_rl=features["z_rl"][i + 1],
                    next_proprio=features["proprio"][i + 1],
                    next_ref_chunk=action.copy(),
                    source=int(TransitionSource.HUMAN),
                    source_chunk=np.full(3, int(TransitionSource.HUMAN), dtype=np.uint8),
                    valid_mask=np.ones(3, dtype=np.bool_),
                    executed_steps=3,
                    collection_phase="warmup",
                    episode_id=i,
                    step_id=0,
                    intervention_flag=True,
                )
            )
    replay.save(root / "replay.npz")
    stage2 = root / "stage2"
    run(
        "train_stage2",
        "--stage1_checkpoint",
        stage1 / "last_checkpoint.pt",
        "--replay_path",
        root / "replay.npz",
        "--output_dir",
        stage2,
        "--steps",
        "2",
        "--batch_size",
        "4",
        "--device",
        "cpu",
    )
    run(
        "train_stage2",
        "--resume_from",
        stage2 / "last_checkpoint.pt",
        "--replay_path",
        root / "replay.npz",
        "--output_dir",
        stage2,
        "--steps",
        "2",
        "--batch_size",
        "4",
        "--device",
        "cpu",
    )
    exported = root / "pretrained_model"
    run(
        "export_policy",
        "--checkpoint",
        stage2 / "last_checkpoint.pt",
        "--output_dir",
        exported,
        "--without_processors",
    )
    expected = RLTLearner.load(stage2 / "last_checkpoint.pt").policy
    actual = PI05RLTPolicy.from_pretrained(exported, local_files_only=True)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key])
    mean, std = actual.actor(torch.zeros(1, 8), torch.zeros(1, 14), torch.zeros(1, 3, 14))
    assert mean.shape == std.shape == (1, 3, 14) and torch.isfinite(mean).all()
    print(f"PASS: Stage 1/resume -> encode/classifier -> Stage 2/resume -> export/reload: {root}")
    print("Synthetic smoke test only: no pretrained PI05, real rewards, or robot rollout validated.")


if __name__ == "__main__":
    main()
