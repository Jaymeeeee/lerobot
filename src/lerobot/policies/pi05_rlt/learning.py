# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted from train_rlt_stage2_pi05.py; see NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Reference Stage-2 learner, independent of robot transport or ROS."""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

if TYPE_CHECKING:
    from .modeling_pi05_rlt import PI05RLTPolicy
    from .replay import ReplayBuffer


def discounted_chunk_return(rewards: Tensor, valid_mask: Tensor, discount: float) -> tuple[Tensor, Tensor]:
    """Return sum(gamma**i*r_i), gamma**executed_steps for a contiguous prefix."""
    if rewards.ndim != 2 or rewards.shape != valid_mask.shape or not torch.isfinite(rewards).all():
        raise ValueError("Rewards/mask must be finite [batch, horizon] tensors")
    mask = valid_mask.bool()
    steps = mask.sum(-1, keepdim=True)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < steps
    if not (steps > 0).all() or not torch.equal(mask, expected):
        raise ValueError("valid_mask must mark a nonempty contiguous executed prefix")
    weights = torch.as_tensor(discount, device=rewards.device).pow(
        torch.arange(rewards.shape[1], device=rewards.device)
    )
    returns = (rewards * mask * weights).sum(-1, keepdim=True)
    return returns, torch.as_tensor(discount, device=rewards.device).pow(steps)


def sample_fixed_std_action(mean: Tensor, std: Tensor) -> Tensor:
    return mean + std * torch.randn_like(mean)


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
        target_parameter.lerp_(parameter, tau)


@contextmanager
def frozen_parameters(module: nn.Module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(flag)


def critic_loss(
    policy: PI05RLTPolicy, batch: dict[str, Tensor], target_clip: float | None = 2.0
) -> dict[str, Tensor]:
    with torch.no_grad():
        mean, std = policy.actor_target(batch["next_z_rl"], batch["next_proprio"], batch["next_ref_chunk"])
        next_q = policy.critic_target.q_min(
            batch["next_z_rl"], batch["next_proprio"], sample_fixed_std_action(mean, std)
        )
        reward, discount = discounted_chunk_return(
            batch["rewards"], batch["valid_mask"], policy.config.discount
        )
        target = reward + discount * (1 - batch["done"].reshape(-1, 1).float()) * next_q
        if target_clip is not None:
            target = target.clamp(-target_clip, target_clip)
    q1, q2 = policy.critic(batch["z_rl"], batch["proprio"], batch["action_chunk"])
    return {
        "critic_loss": functional.mse_loss(q1, target) + functional.mse_loss(q2, target),
        "q1_mean": q1.mean(),
        "q2_mean": q2.mean(),
        "td_target_mean": target.mean(),
    }


def actor_loss(
    policy: PI05RLTPolicy,
    batch: dict[str, Tensor],
    reference_dropout: float = 0.5,
    bc_weight: float | None = None,
    q_weight: float = 1.0,
) -> dict[str, Tensor]:
    refs = batch["ref_chunk"]
    drop = torch.rand(refs.shape[0], device=refs.device) < reference_dropout
    inputs = torch.where(drop[:, None, None], torch.zeros_like(refs), refs)
    mean, std = policy.actor(batch["z_rl"].detach(), batch["proprio"], inputs)
    sample = sample_fixed_std_action(mean, std)
    with frozen_parameters(policy.critic):
        q = policy.critic.q_min(batch["z_rl"].detach(), batch["proprio"], sample).mean()
    source = batch["source_chunk"]
    human = (source == 2) | (source == 3)
    target = torch.where(human[..., None], batch["action_chunk"], refs).detach()
    mask = batch["valid_mask"].float()
    penalty = (((sample - target) ** 2).mean(-1) * mask).sum() / mask.sum().clamp_min(1)
    weight = policy.config.bc_weight if bc_weight is None else bc_weight
    return {
        "actor_loss": weight * penalty - q_weight * q,
        "q_loss": -q,
        "bc_loss": penalty,
        "reference_dropout_fraction": drop.float().mean(),
    }


@dataclass
class RLTLearnerConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    reference_dropout: float = 0.5
    critic_per_actor: int = 2
    utd_ratio: int = 5
    critic_grad_clip: float = 10.0
    td_target_clip: float | None = 2.0

    def __post_init__(self) -> None:
        for name in ("actor_lr", "critic_lr", "critic_grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.reference_dropout) or not 0 <= self.reference_dropout <= 1:
            raise ValueError("reference_dropout must be in [0,1]")
        for name in ("critic_per_actor", "utd_ratio"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.td_target_clip is not None and (
            not math.isfinite(self.td_target_clip) or self.td_target_clip <= 0
        ):
            raise ValueError("td_target_clip must be positive or None")


class RLTLearner:
    """Twin critics + delayed full-output Gaussian actor and Polyak targets.

    Call `update` on normalized replay batches, or `learn` after adding online
    transitions. Human/mixed rows regularize toward executed actions; other
    rows regularize toward the frozen VLA reference. No robot is opened here.
    """

    def __init__(self, policy: PI05RLTPolicy, config: RLTLearnerConfig | None = None) -> None:
        if policy.config.mode != "online_rl":
            raise ValueError("Stage-2 learner requires policy mode=online_rl")
        self.policy = policy
        self.config = config or RLTLearnerConfig()
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=self.config.critic_lr)
        self.updates = 0
        self.lock = threading.RLock()

    def _batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        p = self.policy
        b = {
            k: v.detach().to(device=next(p.parameters()).device, dtype=torch.float32)
            for k, v in batch.items()
        }
        n = b["z_rl"].shape[0]
        z, s, c, a = (
            p.config.rlt_hidden_dim,
            p.config.state_dim,
            p.config.n_action_steps_rl,
            p.config.action_dim,
        )
        shapes = {
            "z_rl": (n, z),
            "next_z_rl": (n, z),
            "proprio": (n, s),
            "next_proprio": (n, s),
            "ref_chunk": (n, c, a),
            "next_ref_chunk": (n, c, a),
            "action_chunk": (n, c, a),
            "rewards": (n, c),
            "valid_mask": (n, c),
            "source_chunk": (n, c),
        }
        for key, shape in shapes.items():
            if key not in b or b[key].shape != shape or not torch.isfinite(b[key]).all():
                raise ValueError(f"Invalid replay field {key}; expected {shape}")
        if b["done"].numel() != n or not torch.isfinite(b["done"]).all():
            raise ValueError("Invalid replay done flags")
        if not ((b["done"] == 0) | (b["done"] == 1)).all():
            raise ValueError("Replay done flags must be binary")
        if not ((b["valid_mask"] == 0) | (b["valid_mask"] == 1)).all():
            raise ValueError("Replay valid_mask must be binary")
        if not (
            (b["source_chunk"] >= 0)
            & (b["source_chunk"] <= 3)
            & (b["source_chunk"] == b["source_chunk"].round())
        ).all():
            raise ValueError("Invalid replay action sources")
        discounted_chunk_return(b["rewards"], b["valid_mask"], p.config.discount)
        return b

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        with self.lock:
            b = self._batch(batch)
            p = self.policy
            p.train()
            losses = critic_loss(p, b, self.config.td_target_clip)
            self.critic_optimizer.zero_grad(set_to_none=True)
            losses["critic_loss"].backward()
            nn.utils.clip_grad_norm_(
                p.critic.parameters(), self.config.critic_grad_clip, error_if_nonfinite=True
            )
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            actor_updated = self.updates % self.config.critic_per_actor == 0
            if actor_updated:
                losses.update(actor_loss(p, b, self.config.reference_dropout))
                if not torch.isfinite(losses["actor_loss"]):
                    raise FloatingPointError("Nonfinite actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                losses["actor_loss"].backward()
                nn.utils.clip_grad_norm_(p.actor.parameters(), float("inf"), error_if_nonfinite=True)
                self.actor_optimizer.step()
                polyak_update(p.actor, p.actor_target, p.config.target_tau)
                polyak_update(p.critic, p.critic_target, p.config.target_tau)
            self.updates += 1
            return dict(
                update=self.updates,
                actor_updated=float(actor_updated),
                **{k: float(v.detach()) for k, v in losses.items()},
            )

    def learn(
        self, replay: ReplayBuffer, batch_size: int, new_transitions: int = 1
    ) -> list[dict[str, float]]:
        if new_transitions < 0 or batch_size < 1:
            raise ValueError("Invalid transition/batch count")
        if len(replay) < batch_size:
            return []
        return [
            self.update(replay.sample(batch_size)) for _ in range(new_transitions * self.config.utd_ratio)
        ]

    def save(self, path: str | Path, **metadata) -> None:
        from .training import atomic_torch_save, encode_policy_config

        with self.lock:
            atomic_torch_save(
                {
                    "schema_version": 1,
                    "kind": "pi05_rlt_stage2",
                    "policy_config": encode_policy_config(self.policy.config),
                    "policy": self.policy.state_dict(),
                    "learner_config": asdict(self.config),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "updates": self.updates,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if next(self.policy.parameters()).is_cuda
                    else [],
                    "metadata": metadata,
                },
                Path(path),
            )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> RLTLearner:
        from lerobot.configs import PreTrainedConfig

        from .modeling_pi05_rlt import PI05RLTPolicy

        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("kind") != "pi05_rlt_stage2" or data.get("schema_version") != 1:
            raise ValueError("Unsupported Stage-2 checkpoint")
        config = draccus.decode(PreTrainedConfig, data["policy_config"])
        config.device = device
        policy = PI05RLTPolicy(config).to(device)
        policy.load_state_dict(data["policy"], strict=True)
        learner = cls(policy, RLTLearnerConfig(**data["learner_config"]))
        learner.actor_optimizer.load_state_dict(data["actor_optimizer"])
        learner.critic_optimizer.load_state_dict(data["critic_optimizer"])
        learner.updates = int(data["updates"])
        torch.set_rng_state(data["rng_state"])
        if torch.device(device).type == "cuda" and data["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(data["cuda_rng_state"])
        return learner
