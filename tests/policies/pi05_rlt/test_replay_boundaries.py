import numpy as np
import pytest
import torch

from lerobot.policies.pi05_rlt.replay import (
    BoundaryContext,
    DeferredBoundarySnapshot,
    ExecutionWindow,
    ReplayBuffer,
    StepRecord,
    TransitionSource,
    build_stride2_transitions,
    build_transition,
)

from .helpers import transition


def test_single_step_boundary_keeps_action_axis():
    boundary = BoundaryContext(torch.zeros(8), torch.zeros(14), torch.zeros(1, 14), "online", 0, 0)
    window = ExecutionWindow(boundary)
    window.append(torch.ones(14), TransitionSource.HUMAN, intervention=True)
    item = build_transition(window, boundary, horizon=1, action_dim=14, terminal=True, reward=1)
    item.validate()
    assert item.ref_chunk.shape == (1, 14) and item.rewards.tolist() == [1]


def test_stride_windows_preserve_real_boundaries_human_refs_and_terminal_reward():
    records = [
        StepRecord(
            torch.full((14,), float(i)),
            int(TransitionSource.HUMAN if i == 2 else TransitionSource.RL),
            i == 2,
            100 + i,
            torch.full((14,), float(20 + i)),
        )
        for i in range(7)
    ]
    snapshots = {i: DeferredBoundarySnapshot({}, torch.zeros(14), 100 + i) for i in (0, 2, 4, 6)}
    terminal = DeferredBoundarySnapshot({}, torch.zeros(14), 107)

    def materialize(snapshot):
        return BoundaryContext(
            torch.full((8,), float(snapshot.step_id)),
            snapshot.state,
            torch.zeros(4, 14),
            "online",
            7,
            snapshot.step_id,
        )

    rows = build_stride2_transitions(
        records,
        snapshots,
        terminal,
        horizon=4,
        action_dim=14,
        stride=2,
        materialize_snapshot=materialize,
        terminal_reward=1,
    )
    assert [r.step_id for r in rows] == [100, 102, 106]
    assert [r.executed_steps for r in rows] == [4, 4, 1]
    assert [r.done for r in rows] == [False, False, True]
    assert rows[0].next_z_rl[0] == 104 and rows[1].next_z_rl[0] == 106
    np.testing.assert_array_equal(rows[0].ref_chunk[:, 0], [20, 21, 2, 23])
    np.testing.assert_array_equal(rows[-1].rewards, [1, 0, 0, 0])
    for row in rows:
        row.validate()
    with pytest.raises(ValueError, match="gaps"):
        build_stride2_transitions(
            records[:2] + records[3:],
            snapshots,
            terminal,
            horizon=4,
            action_dim=14,
            stride=2,
            materialize_snapshot=materialize,
        )


def test_uniform_replay_sampling_options_and_rng_survive_resume(tmp_path):
    replay = ReplayBuffer(4, sample_strategy="uniform", recent_episode_window=3, seed=73)
    for index in range(4):
        replay.add(transition(index))
    replay.save(tmp_path / "replay.npz")
    restored = ReplayBuffer.load(tmp_path / "replay.npz")
    assert restored.stats()["sample_strategy"] == "uniform"
    assert restored._recent_episode_window == 3
    expected, actual = replay.sample(4), restored.sample(4)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])
