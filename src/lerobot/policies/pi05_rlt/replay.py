# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted from scripts/rlt_core.py and train_rlt_stage2_pi05.py; see NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-independent action-window transitions and replay for RL-Token."""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


class TransitionSource(enum.IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


def collection_phase_to_id(phase: str) -> int:
    name = str(phase).split(":", 1)[0].lower()
    if name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def _ensure_array(value, *, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


@dataclass
class Transition:
    z_rl: np.ndarray  # (rlt_hidden_dim,)
    proprio: np.ndarray  # (state_dim,)
    ref_chunk: np.ndarray  # (n_action_steps_rl, action_dim)
    action_chunk: np.ndarray  # (n_action_steps_rl, action_dim)
    rewards: np.ndarray  # (n_action_steps_rl,)
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source: int
    source_chunk: np.ndarray  # uint8 (n_action_steps_rl,)
    valid_mask: np.ndarray  # bool (n_action_steps_rl,)
    executed_steps: int
    collection_phase: str
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0

    def validate(self) -> None:
        """Reject misaligned boundaries, padding, sources and nonfinite values."""
        ref = np.asarray(self.ref_chunk)
        if ref.ndim != 2 or 0 in ref.shape:
            raise ValueError("ref_chunk must be a nonempty (horizon, action_dim) array")
        horizon = ref.shape[0]
        for key, shape in (
            ("action_chunk", ref.shape),
            ("next_ref_chunk", ref.shape),
            ("rewards", (horizon,)),
            ("source_chunk", (horizon,)),
            ("valid_mask", (horizon,)),
        ):
            if np.asarray(getattr(self, key)).shape != shape:
                raise ValueError(f"{key} shape must be {shape}")
        for key in ("z_rl", "proprio"):
            value = np.asarray(getattr(self, key))
            if (
                value.ndim != 1
                or value.size == 0
                or np.asarray(getattr(self, "next_" + key)).shape != value.shape
            ):
                raise ValueError(f"Invalid {key}/next_{key} boundary shape")
        for key in (
            "z_rl",
            "next_z_rl",
            "proprio",
            "next_proprio",
            "ref_chunk",
            "next_ref_chunk",
            "action_chunk",
            "rewards",
        ):
            if not np.isfinite(getattr(self, key)).all():
                raise ValueError(f"Nonfinite replay field: {key}")
        if not 0 < self.executed_steps <= horizon:
            raise ValueError("executed_steps must be in [1, horizon]")
        expected = np.arange(horizon) < self.executed_steps
        if not np.array_equal(self.valid_mask, expected):
            raise ValueError("valid_mask must be exactly the executed prefix")
        if not np.isin(self.source_chunk, list(TransitionSource)).all() or self.source not in list(
            TransitionSource
        ):
            raise ValueError("Unknown action source")

    def to_numpy(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "z_rl": _ensure_array(self.z_rl, dtype=np.float16),
            "proprio": _ensure_array(self.proprio, dtype=np.float32),
            "ref_chunk": _ensure_array(self.ref_chunk, dtype=np.float16),
            "action_chunk": _ensure_array(self.action_chunk, dtype=np.float16),
            "rewards": _ensure_array(self.rewards, dtype=np.float32),
            "done": _ensure_array(self.done, dtype=np.bool_),
            "next_z_rl": _ensure_array(self.next_z_rl, dtype=np.float16),
            "next_proprio": _ensure_array(self.next_proprio, dtype=np.float32),
            "next_ref_chunk": _ensure_array(self.next_ref_chunk, dtype=np.float16),
            "source": _ensure_array(self.source, dtype=np.uint8),
            "source_chunk": _ensure_array(self.source_chunk, dtype=np.uint8),
            "valid_mask": _ensure_array(self.valid_mask, dtype=np.bool_),
            "executed_steps": _ensure_array(self.executed_steps, dtype=np.int16),
            "collection_phase_id": _ensure_array(
                collection_phase_to_id(self.collection_phase),
                dtype=np.uint8,
            ),
            "success": _ensure_array(self.success, dtype=np.int8),
            "intervention_flag": _ensure_array(self.intervention_flag, dtype=np.bool_),
            "episode_id": _ensure_array(self.episode_id, dtype=np.int32),
            "step_id": _ensure_array(self.step_id, dtype=np.int32),
        }


@dataclass
class BoundaryContext:
    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor
    collection_phase: str
    episode_id: int
    step_id: int


@dataclass
class ExecutionWindow:
    """Actions actually accepted by hardware under one observation boundary."""

    boundary: BoundaryContext
    actions: list[torch.Tensor] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    intervention_flags: list[bool] = field(default_factory=list)

    def append(self, actual_action: torch.Tensor, source: int, intervention: bool = False) -> None:
        if actual_action.ndim != 1:
            raise ValueError(f"Executed action must be rank 1, got {tuple(actual_action.shape)}")
        self.actions.append(actual_action.detach().clone())
        self.sources.append(int(source))
        self.intervention_flags.append(bool(intervention))

    @property
    def executed_steps(self) -> int:
        return len(self.actions)


def _resolve_chunk_source(sources: list[int], intervention_flags: list[bool]) -> tuple[int, bool]:
    intervention = any(intervention_flags)
    source_set = set(sources)
    has_human = int(TransitionSource.HUMAN) in source_set
    has_policy = any(
        s in source_set
        for s in (
            int(TransitionSource.BASE),
            int(TransitionSource.RL),
            int(TransitionSource.MIXED),
        )
    )
    if int(TransitionSource.MIXED) in source_set or (has_human and has_policy):
        return int(TransitionSource.MIXED), intervention
    if has_human or intervention:
        return int(TransitionSource.HUMAN), intervention
    return int(sources[0]), intervention


@dataclass
class DeferredBoundarySnapshot:
    """Observation captured at takeover; expensive boundary inference is deferred."""

    observation: dict
    state: torch.Tensor
    step_id: int
    raw_actions: torch.Tensor | None = None
    raw_action_index: int = 0


@dataclass
class StepRecord:
    """One executed control step inside the critical-phase replay segment."""

    action: torch.Tensor  # normalized actual action (action_dim,)
    source: int  # TransitionSource value
    intervention: bool
    step_id: int  # absolute episode control-step id
    ref_row: torch.Tensor | None = None  # VLA reference row for this step (normalized); None for HUMAN steps


def build_transition(
    window: ExecutionWindow,
    next_boundary: BoundaryContext,
    *,
    horizon: int,
    action_dim: int,
    reward: float = 0.0,
    terminal: bool = False,
) -> Transition:
    """Materialize a replay row; only real terminal boundaries receive reward/done."""
    if window.executed_steps < 1 or window.executed_steps > horizon:
        raise ValueError(f"Execution window has {window.executed_steps} actions; expected 1..{horizon}")
    if len(window.sources) != window.executed_steps:
        raise ValueError("Execution window source count does not match action count")
    actions = torch.zeros(horizon, action_dim, dtype=torch.float32)
    stacked = torch.stack(window.actions, dim=0).cpu()
    if stacked.shape[1] != action_dim:
        raise ValueError(f"Executed action dim {stacked.shape[1]} != expected {action_dim}")
    actions[: window.executed_steps] = stacked
    refs = window.boundary.ref_chunk.detach().cpu().float().clone()
    if refs.ndim == 3 and refs.shape[0] == 1:
        refs = refs[0]
    if refs.shape != (horizon, action_dim):
        raise ValueError(f"Reference chunk shape {tuple(refs.shape)} != {(horizon, action_dim)}")
    source_chunk = np.full(horizon, int(TransitionSource.BASE), dtype=np.uint8)
    source_chunk[: window.executed_steps] = np.asarray(window.sources, dtype=np.uint8)
    # Algorithm 1: an intervention replaces the VLA reference in replay.
    for index, source in enumerate(window.sources):
        if source == int(TransitionSource.HUMAN):
            refs[index] = actions[index]
    valid_mask = np.zeros(horizon, dtype=np.bool_)
    valid_mask[: window.executed_steps] = True
    rewards = np.zeros(horizon, dtype=np.float32)
    if terminal:
        rewards[window.executed_steps - 1] = float(reward)
    source, intervention = _resolve_chunk_source(window.sources, window.intervention_flags)
    return Transition(
        z_rl=window.boundary.z_rl.detach().cpu().numpy().reshape(-1),
        proprio=window.boundary.proprio.detach().cpu().numpy().reshape(-1),
        ref_chunk=refs.numpy(),
        action_chunk=actions.numpy(),
        rewards=rewards,
        done=bool(terminal),
        next_z_rl=next_boundary.z_rl.detach().cpu().numpy().reshape(-1),
        next_proprio=next_boundary.proprio.detach().cpu().numpy().reshape(-1),
        next_ref_chunk=next_boundary.ref_chunk.detach().cpu().numpy().reshape(horizon, action_dim),
        source=source,
        source_chunk=source_chunk,
        valid_mask=valid_mask,
        executed_steps=window.executed_steps,
        collection_phase=window.boundary.collection_phase,
        intervention_flag=intervention,
        episode_id=window.boundary.episode_id,
        step_id=window.boundary.step_id,
    )


def build_stride2_transitions(
    step_records: list[StepRecord],
    stride_snapshots: dict[int, DeferredBoundarySnapshot],
    terminal_snapshot: DeferredBoundarySnapshot | None,
    *,
    horizon: int,
    action_dim: int,
    stride: int,
    materialize_snapshot,
    terminal_reward: float = 0.0,
) -> list[Transition]:
    """Build stride-2 overlapping replay transitions from a continuous step log.

    This follows the paper's subsampling rule: for stride=2 and chunk length C,
    save transitions <x0, a0:C>, <x2, a2:C+2>, <x4, a4:C+4>, ... The start/end
    observations are taken from ``stride_snapshots``; intermediate actions come
    from ``step_records``. Complete windows are always emitted. If the episode
    ends before a full window, the last partial tail is emitted as a padded
    terminal window. When the critical phase itself is shorter than one horizon
    (fewer executed steps than ``horizon``), every stride start is padded with
    the terminal observation so no executed control step is dropped from replay.
    """
    if stride < 1 or horizon < 1:
        return []
    records = sorted(step_records, key=lambda item: item.step_id)
    if not records:
        return []
    if any(b.step_id != a.step_id + 1 for a, b in zip(records, records[1:], strict=False)):
        raise ValueError("Replay windows cannot cross gaps in the executed step log")

    boundary_cache: dict[int, BoundaryContext] = {}

    def resolve(snapshot: DeferredBoundarySnapshot) -> BoundaryContext:
        key = id(snapshot)
        if key not in boundary_cache:
            boundary_cache[key] = materialize_snapshot(snapshot)
        return boundary_cache[key]

    transitions: list[Transition] = []
    short_episode = len(records) < horizon
    last_possible_start = len(records) - 1
    for start_idx in range(0, len(records), stride):
        if start_idx > last_possible_start:
            break
        start_snapshot = stride_snapshots.get(start_idx)
        if start_snapshot is None:
            continue

        end_idx = start_idx + horizon
        if end_idx < len(records):
            end_snapshot = stride_snapshots.get(end_idx)
            if end_snapshot is None:
                continue
        elif end_idx == len(records) and terminal_snapshot is not None:
            # Complete final window ends at the terminal observation.
            end_snapshot = terminal_snapshot
        elif terminal_snapshot is not None and (short_episode or start_idx + stride > len(records)):
            # Short episode (< horizon steps): pad every stride start with the
            # terminal observation. Otherwise only the last partial tail is
            # emitted as one padded terminal window.
            end_snapshot = terminal_snapshot
        else:
            continue

        window = ExecutionWindow(resolve(start_snapshot))
        for record in records[start_idx:end_idx]:
            window.append(record.action, record.source, record.intervention)
        if window.executed_steps == 0:
            continue
        ends_at_terminal = terminal_snapshot is not None and end_snapshot is terminal_snapshot
        transition = build_transition(
            window,
            resolve(end_snapshot),
            horizon=horizon,
            action_dim=action_dim,
            terminal=ends_at_terminal,
            reward=terminal_reward if ends_at_terminal else 0.0,
        )
        # Per-step reference alignment: a stride window can cross a re-plan
        # boundary (replan every C steps), so its tail rows were executed under
        # the NEXT chunk's VLA reference, not the window-start boundary's.
        # Overwrite each non-HUMAN row with the reference of the chunk that
        # actually covered that step; HUMAN rows keep the executed-action
        # replacement from build_transition (Algorithm 1: intervention replaces
        # the VLA reference).
        for index, record in enumerate(records[start_idx:end_idx]):
            if record.ref_row is not None and record.source != int(TransitionSource.HUMAN):
                transition.ref_chunk[index] = record.ref_row.detach().cpu().numpy()
        transitions.append(transition)
    return transitions


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 0,
        sample_strategy: str = "stratified",
        recent_episode_window: int = 20,
        recent_online_ratio: float = 0.4,
        warmup_demo_ratio: float = 0.3,
        human_intervention_ratio: float = 0.2,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Replay capacity must be a positive integer")
        if sample_strategy not in ("uniform", "stratified"):
            raise ValueError("Unknown replay sample strategy")
        ratios = (recent_online_ratio, warmup_demo_ratio, human_intervention_ratio)
        if any(not np.isfinite(r) or r < 0 for r in ratios) or sum(ratios) > 1:
            raise ValueError("Replay sampling ratios must be nonnegative and sum to at most one")
        self.capacity = capacity
        self._sample_strategy = sample_strategy
        self._recent_episode_window = recent_episode_window
        self._recent_online_ratio = recent_online_ratio
        self._warmup_demo_ratio = warmup_demo_ratio
        self._human_intervention_ratio = human_intervention_ratio
        self._storage: dict[str, np.ndarray] | None = None
        self._position = 0
        self._size = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

    def _initialize_storage(self, record: dict[str, np.ndarray]) -> None:
        self._storage = {}
        for key, value in record.items():
            value = _ensure_array(value)
            self._storage[key] = np.empty((self.capacity, *value.shape), dtype=value.dtype)

    def add(self, transition: Transition) -> None:
        record = transition.to_numpy()
        if any(not np.isfinite(v).all() for v in record.values()):
            raise ValueError("Replay precision conversion overflow")
        with self._lock:
            if self._storage is None:
                self._initialize_storage(record)
            for key, value in record.items():
                expected = self._storage[key].shape[1:]
                if value.shape != expected:
                    raise ValueError(f"{key} shape {value.shape} != expected {expected}")
            for key, value in record.items():
                self._storage[key][self._position] = value
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        with self._lock:
            batch_size = min(batch_size, self._size)
            if batch_size == 0:
                return {}
            if self._sample_strategy == "stratified":
                indices = self._sample_stratified_indices(batch_size)
            else:
                indices = self._sample_uniform_indices(batch_size)
            raw = {k: v[indices].copy() for k, v in self._storage.items()}
        result = {}
        for k, v in raw.items():
            if v.dtype == np.float16:
                result[k] = torch.from_numpy(v.astype(np.float32))
            elif v.dtype == np.bool_:
                tensor = torch.from_numpy(v.astype(np.float32))
                result[k] = tensor.unsqueeze(-1) if k == "done" else tensor
            else:
                result[k] = torch.from_numpy(v.astype(np.float32))
        return result

    def _sample_uniform_indices(self, batch_size: int) -> np.ndarray:
        return self._rng.integers(0, self._size, size=batch_size)

    def _sample_stratified_indices(self, batch_size: int) -> np.ndarray:
        phase = self._storage["collection_phase_id"][: self._size]
        episode_id = self._storage["episode_id"][: self._size]
        source = self._storage["source"][: self._size]
        source_chunk = self._storage["source_chunk"][: self._size]
        intervention = self._storage["intervention_flag"][: self._size]
        all_idx = np.arange(self._size, dtype=np.int64)

        max_ep = int(np.max(episode_id)) if episode_id.size else -1
        recent_start = max_ep - max(self._recent_episode_window, 1) + 1

        recent_pool = np.flatnonzero((phase == COLLECTION_PHASE_ONLINE) & (episode_id >= recent_start))
        warmup_pool = np.flatnonzero(phase == COLLECTION_PHASE_WARMUP)
        human_pool = np.flatnonzero(
            intervention
            | (source == int(TransitionSource.HUMAN))
            | (source == int(TransitionSource.MIXED))
            | np.any(source_chunk == int(TransitionSource.HUMAN), axis=1)
            | np.any(source_chunk == int(TransitionSource.MIXED), axis=1)
        )

        n_recent = int(round(batch_size * self._recent_online_ratio))
        n_warmup = int(round(batch_size * self._warmup_demo_ratio))
        n_human = int(round(batch_size * self._human_intervention_ratio))
        n_uniform = max(batch_size - n_recent - n_warmup - n_human, 0)

        parts = [
            self._pick(recent_pool, n_recent),
            self._pick(warmup_pool, n_warmup),
            self._pick(human_pool, n_human),
            self._pick(all_idx, n_uniform),
        ]
        result = np.concatenate([p for p in parts if p.size > 0])
        if result.size < batch_size:
            result = np.concatenate([result, self._pick(all_idx, batch_size - result.size)])
        self._rng.shuffle(result)
        return result[:batch_size]

    def _pick(self, pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or pool.size == 0:
            return np.empty((0,), dtype=np.int64)
        return self._rng.choice(pool, size=min(count, pool.size), replace=True)

    @property
    def intervention_count(self) -> int:
        if self._storage is None or self._size == 0:
            return 0
        return int(np.sum(self._storage["intervention_flag"][: self._size]))

    @property
    def human_count(self) -> int:
        if self._storage is None or self._size == 0:
            return 0
        source = self._storage["source"][: self._size]
        return int(np.sum((source == int(TransitionSource.HUMAN)) | (source == int(TransitionSource.MIXED))))

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "position": self._position,
            "sample_strategy": self._sample_strategy,
            "intervention_count": self.intervention_count,
        }

    def __len__(self) -> int:
        return self._size

    def save(self, path: str | Path) -> None:
        """Atomically save replay arrays and sampling state without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self._size:
                raise ValueError("Cannot save empty replay")
            ids = np.arange(self._size)
            payload = {k: v[ids] for k, v in self._storage.items()}
            payload["_rng_json"] = np.asarray(json.dumps(self._rng.bit_generator.state))
            payload["_position"] = np.asarray(self._position)
            payload["_capacity"] = np.asarray(self.capacity)
            payload["_sampling_json"] = np.asarray(
                json.dumps(
                    {
                        "sample_strategy": self._sample_strategy,
                        "recent_episode_window": self._recent_episode_window,
                        "recent_online_ratio": self._recent_online_ratio,
                        "warmup_demo_ratio": self._warmup_demo_ratio,
                        "human_intervention_ratio": self._human_intervention_ratio,
                    }
                )
            )
        temporary = path.with_name("." + path.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, capacity: int | None = None, **kwargs) -> ReplayBuffer:
        """Load validated transitions from a local NPZ replay snapshot."""
        with np.load(path, allow_pickle=False) as data:
            n = len(data["z_rl"])
            if any(v.shape[0] != n for k, v in data.items() if not k.startswith("_")):
                raise ValueError("Replay arrays have different row counts")
            saved_capacity = int(data["_capacity"]) if "_capacity" in data else n
            saved_position = int(data["_position"]) if "_position" in data else n % saved_capacity
            if saved_capacity < 1 or not 0 < n <= saved_capacity or not 0 <= saved_position < saved_capacity:
                raise ValueError("Invalid replay ring metadata")
            if n < saved_capacity and saved_position != n:
                raise ValueError("Invalid replay write position")
            sampling = json.loads(str(data["_sampling_json"])) if "_sampling_json" in data else {}
            result = cls(saved_capacity if capacity is None else capacity, **(sampling | kwargs))
            ids = np.arange(n)
            if result.capacity != saved_capacity and n == saved_capacity:
                ids = (ids + saved_position) % n
            for i in ids:
                fields = {k: data[k][i] for k in data.files if not k.startswith("_")}
                phase = int(fields.pop("collection_phase_id"))
                fields["collection_phase"] = {1: "warmup", 2: "online"}.get(phase, "unknown")
                result.add(Transition(**fields))
            if result.capacity == saved_capacity:
                result._position = saved_position
            if "_rng_json" in data:
                result._rng.bit_generator.state = json.loads(str(data["_rng_json"]))
        return result
