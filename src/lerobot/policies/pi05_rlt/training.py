# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted from train_rlt_stage1_pi05.py; see NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Mmap cache readers and atomic checkpoints shared by RLT training scripts."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import draccus
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from lerobot.configs import PreTrainedConfig

    from .configuration_pi05_rlt import PI05RLTConfig
    from .modeling_pi05_rlt import PI05RLTPolicy


class MMapPrefixDataset(Dataset):
    """Lazily reads prefix embeddings and masks from one or more mmap-backed cache roots.

    Multi-shard support: precomputation can be split by episode across several
    GPUs, each writing its own cache directory; training then concatenates the
    directories into one logical dataset.
    The concatenation order follows the order of ``roots``, and frame order is
    preserved inside each shard (shuffling only happens in the training loop).
    """

    def __init__(
        self, roots: str | Path | list[str | Path], counts: int | list[int], seq_len: int, hidden_dim: int
    ) -> None:
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if isinstance(counts, int):
            counts = [counts]
        if len(roots) != len(counts):
            raise ValueError(f"roots/counts length mismatch: {len(roots)} vs {len(counts)}")
        self.roots = [Path(r) for r in roots]
        self.counts = [int(c) for c in counts]
        self.n_samples = int(sum(self.counts))
        if self.n_samples <= 0:
            raise ValueError("multi-shard cache contains zero samples")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self._offsets: list[int] = []
        acc = 0
        for c in self.counts:
            self._offsets.append(acc)
            acc += c
        self._prefix = None
        self._mask = None

    def _open(self) -> None:
        # Open independently in every DataLoader process; never pickle live mmap handles.
        if self._prefix is None:
            self._prefix, self._mask = [], []
            for root, n in zip(self.roots, self.counts, strict=False):
                self._prefix.append(
                    np.memmap(
                        str(Path(root) / "prefix_out.mmap"),
                        dtype=np.float16,
                        mode="r",
                        shape=(n, self.seq_len, self.hidden_dim),
                    )
                )
                self._mask.append(
                    np.memmap(
                        str(Path(root) / "prefix_mask.mmap"),
                        dtype=np.bool_,
                        mode="r",
                        shape=(n, self.seq_len),
                    )
                )

    def __len__(self) -> int:
        return self.n_samples

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.n_samples
        if not 0 <= index < self.n_samples:
            raise IndexError(index)
        for shard in range(len(self.roots) - 1, -1, -1):
            if index >= self._offsets[shard]:
                return shard, index - self._offsets[shard]
        raise IndexError(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        shard, local = self._locate(index)
        # Copies are batch-sized/lazy and make the read-only mmap safe for torch tensors.
        prefix = torch.from_numpy(np.array(self._prefix[shard][local], copy=True))
        mask = torch.from_numpy(np.array(self._mask[shard][local], copy=True))
        if not mask.any():
            raise ValueError(f"Cache sample {index} has no valid image tokens.")
        if not torch.isfinite(prefix).all():
            raise ValueError(f"Cache sample {index} contains NaN or Inf prefix embeddings.")
        return prefix, mask

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_prefix"] = None
        state["_mask"] = None
        return state


def encode_policy_config(config: PreTrainedConfig) -> dict[str, Any]:
    """JSON primitives only, with the registry type retained for safe torch loading."""
    from lerobot.configs import PreTrainedConfig

    return json.loads(json.dumps(draccus.encode(config, PreTrainedConfig)))


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Write a torch checkpoint atomically in the destination directory."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_prefix_cache(roots: list[str | Path]) -> tuple[MMapPrefixDataset, list[dict]]:
    metas = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        meta = json.loads((root / "meta.json").read_text())
        if (
            not meta.get("complete")
            or not meta.get("image_only")
            or not meta.get("has_mask")
            or meta.get("embedding_stage") != "post_transformer"
            or meta.get("dtype") != "float16"
        ):
            raise ValueError(f"{root}: incomplete/incompatible image-prefix cache")
        n, seq_len, d = (meta.get(k) for k in ("n_samples", "seq_len", "vlm_hidden_dim"))
        if any(type(v) is not int or v <= 0 for v in (n, seq_len, d)):
            raise ValueError("Invalid cache dimensions")
        for filename, size in (("prefix_out.mmap", n * seq_len * d * 2), ("prefix_mask.mmap", n * seq_len)):
            if (root / filename).stat().st_size != size:
                raise ValueError(f"{root / filename}: cache byte length mismatch")
        if metas:
            for key in (
                "seq_len",
                "vlm_hidden_dim",
                "pi05_path",
                "checkpoint_image_features",
                "checkpoint_signature",
            ):
                if meta.get(key) != metas[0].get(key):
                    raise ValueError(f"Cache shards differ in {key}")
        metas.append(dict(meta, root=str(root)))
    if not metas:
        raise ValueError("At least one prefix cache is required")
    return MMapPrefixDataset(
        [m["root"] for m in metas],
        [m["n_samples"] for m in metas],
        metas[0]["seq_len"],
        metas[0]["vlm_hidden_dim"],
    ), metas


def load_rlt_config(path: str | Path, *, mode: str, device: str) -> PI05RLTConfig:
    from lerobot.configs import PreTrainedConfig

    from .configuration_pi05_rlt import PI05RLTConfig

    data = json.loads(Path(path).expanduser().read_text())
    data.update(type="pi05_rlt", mode=mode, device=device)
    config = draccus.decode(PreTrainedConfig, data)
    if not isinstance(config, PI05RLTConfig):
        raise ValueError("Expected pi05_rlt configuration")
    return config


def load_stage1(
    path: str | Path, *, mode: str = "online_rl", device: str = "cpu"
) -> tuple[PI05RLTPolicy, dict[str, Any]]:
    from lerobot.configs import PreTrainedConfig

    from .modeling_pi05_rlt import PI05RLTPolicy

    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("kind") != "pi05_rlt_stage1" or not data.get("image_only"):
        raise ValueError("Expected an image-only pi05_rlt_stage1 checkpoint")
    config = draccus.decode(PreTrainedConfig, data["policy_config"])
    config.mode, config.device = mode, device
    policy = PI05RLTPolicy(config).to(device)
    policy.load_state_dict(data["policy"], strict=True)
    return policy, data
