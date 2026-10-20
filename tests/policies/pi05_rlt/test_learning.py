import copy

import numpy as np
import pytest
import torch

from lerobot.policies.pi05_rlt import PI05RLTPolicy
from lerobot.policies.pi05_rlt.learning import (
    RLTLearner,
    RLTLearnerConfig,
    actor_loss,
    critic_loss,
    discounted_chunk_return,
)
from lerobot.policies.pi05_rlt.replay import BoundaryContext, ExecutionWindow, ReplayBuffer, build_transition

from .helpers import config, transition, zero


def test_discount_uses_executed_steps_and_ignores_padded_rewards():
    reward = torch.tensor([[1.0, 2.0, 999.0], [0.0, 0.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    returns, bootstrap = discounted_chunk_return(reward, mask, 0.9)
    torch.testing.assert_close(returns, torch.tensor([[2.8], [0.81]]))
    torch.testing.assert_close(bootstrap, torch.tensor([[0.81], [0.729]]))
    with pytest.raises(ValueError, match="contiguous"):
        discounted_chunk_return(reward, torch.tensor([[1, 0, 1], [1, 1, 1]]), 0.9)


def test_critic_uses_target_actor_gamma_c_and_terminal_mask(monkeypatch):
    policy = PI05RLTPolicy(config(discount=0.9))
    zero(policy.actor)
    zero(policy.actor_target)
    zero(policy.critic)
    zero(policy.critic_target)
    with torch.no_grad():
        policy.actor_target.net[-1].bias.fill_(0.4)
        policy.critic_target.q1[-1].bias.fill_(0.7)
        policy.critic_target.q2[-1].bias.fill_(0.7)
    replay = ReplayBuffer(2)
    replay.add(transition(0, partial=True))
    replay.add(transition(1))
    batch = replay.sample(2)
    batch["done"].zero_()
    monkeypatch.setattr("lerobot.policies.pi05_rlt.learning.sample_fixed_std_action", lambda m, s: m)
    expected_return, discount = discounted_chunk_return(batch["rewards"], batch["valid_mask"], 0.9)
    expected = expected_return + discount * 0.7
    losses = critic_loss(policy, batch, target_clip=None)
    torch.testing.assert_close(losses["td_target_mean"], expected.mean())
    torch.testing.assert_close(losses["critic_loss"], 2 * expected.square().mean())
    batch["done"].fill_(1)
    terminal = critic_loss(policy, batch, target_clip=None)
    torch.testing.assert_close(terminal["td_target_mean"], expected_return.mean())
    losses["critic_loss"].backward()
    assert all(p.grad is None for p in policy.actor_target.parameters())
    assert all(p.grad is None for p in policy.critic_target.parameters())


def test_human_bc_reference_dropout_padding_and_critic_freezing(monkeypatch):
    policy = PI05RLTPolicy(config())
    zero(policy.actor)
    zero(policy.critic)
    replay = ReplayBuffer(1)
    replay.add(transition())
    batch = replay.sample(1)
    batch["ref_chunk"].fill_(1)
    batch["action_chunk"].fill_(3)
    batch["action_chunk"][:, 2].fill_(10000)
    batch["source_chunk"] = torch.tensor([[2, 0, 3]])
    batch["valid_mask"] = torch.tensor([[1.0, 1.0, 0.0]])
    monkeypatch.setattr("lerobot.policies.pi05_rlt.learning.sample_fixed_std_action", lambda m, s: m)
    losses = actor_loss(policy, batch, reference_dropout=1, bc_weight=10)
    assert losses["reference_dropout_fraction"] == 1
    torch.testing.assert_close(losses["bc_loss"], torch.tensor(5.0))
    torch.testing.assert_close(losses["actor_loss"], torch.tensor(50.0))
    losses["actor_loss"].backward()
    assert any(p.grad is not None for p in policy.actor.parameters())
    assert all(p.grad is None for p in policy.critic.parameters())


def test_learner_updates_heads_only_delays_actor_and_reloads_exactly(tmp_path):
    torch.manual_seed(2)
    policy = PI05RLTPolicy(config())
    before = copy.deepcopy(policy.rlt_encoder.state_dict())
    learner = RLTLearner(policy)
    replay = ReplayBuffer(8, seed=3)
    for i in range(6):
        replay.add(transition(i, partial=i % 2 == 0))
    batch = replay.sample(4)
    first = learner.update(batch)
    second = learner.update(batch)
    assert first["actor_updated"] == 1 and second["actor_updated"] == 0
    assert learner.updates == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in first.values())
    assert not policy.rlt_encoder.training
    for name, value in before.items():
        torch.testing.assert_close(value, policy.rlt_encoder.state_dict()[name], rtol=0, atol=0)
    path = tmp_path / "learner.pt"
    learner.save(path)
    expected = learner.update(batch)
    expected_weights = copy.deepcopy(policy.state_dict())
    restored = RLTLearner.load(path)
    actual = restored.update(batch)
    assert actual == expected
    for name, value in expected_weights.items():
        torch.testing.assert_close(value, restored.policy.state_dict()[name], rtol=0, atol=0)


def test_invalid_replay_fails_before_parameter_update():
    policy = PI05RLTPolicy(config())
    learner = RLTLearner(policy)
    replay = ReplayBuffer(1)
    replay.add(transition())
    batch = replay.sample(1)
    batch["proprio"][0, 0] = float("nan")
    before = copy.deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match="proprio"):
        learner.update(batch)
    assert learner.updates == 0
    for k, v in before.items():
        torch.testing.assert_close(v, policy.state_dict()[k])


def test_learner_utd_budget_and_batch_warmup():
    learner = RLTLearner(PI05RLTPolicy(config()), RLTLearnerConfig(utd_ratio=3))
    replay = ReplayBuffer(4)
    assert learner.learn(replay, 2) == []
    replay.add(transition())
    replay.add(transition(1))
    assert len(learner.learn(replay, 2, new_transitions=2)) == 6


def test_partial_human_window_only_marks_real_terminal_reward():
    start = BoundaryContext(torch.zeros(1, 8), torch.zeros(14), torch.ones(1, 3, 14), "online", 4, 2)
    end = BoundaryContext(torch.ones(1, 8), torch.ones(14), torch.ones(1, 3, 14) * 2, "online", 4, 4)
    window = ExecutionWindow(start)
    window.append(torch.full((14,), 3.0), 2, True)
    window.append(torch.full((14,), 4.0), 0, False)
    row = build_transition(window, end, horizon=3, action_dim=14, reward=1, terminal=True)
    row.validate()
    np.testing.assert_array_equal(row.valid_mask, [True, True, False])
    np.testing.assert_array_equal(row.rewards, [0, 1, 0])
    np.testing.assert_array_equal(row.ref_chunk[0], row.action_chunk[0])
    assert row.source == 3 and row.intervention_flag
    nonterminal = build_transition(window, end, horizon=3, action_dim=14, reward=1, terminal=False)
    assert not nonterminal.done and not nonterminal.rewards.any()


def test_replay_npz_ring_rng_roundtrip_and_failed_add_is_atomic(tmp_path):
    replay = ReplayBuffer(3, seed=3)
    for i in range(5):
        replay.add(transition(i))
    path = tmp_path / "replay.npz"
    replay.save(path)
    expected = replay.sample(3)
    restored = ReplayBuffer.load(path)
    actual = restored.sample(3)
    for k in expected:
        torch.testing.assert_close(expected[k], actual[k])
    assert restored.stats()["position"] == replay.stats()["position"]
    snapshot = copy.deepcopy(restored._storage)
    with pytest.raises(ValueError, match="shape"):
        restored.add(transition(9, action_dim=6))
    for k in snapshot:
        np.testing.assert_array_equal(snapshot[k], restored._storage[k])
