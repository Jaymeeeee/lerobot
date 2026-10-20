# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted for AgileX_ws; changes documented in NOTICE and README.md.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the reference π0.5 RL-Token architecture.

RLT (Physical Intelligence) trains a compact RL token ``z_rl`` on top of
frozen π0.5 embeddings (Stage 1), then a lightweight full-output Gaussian
Actor + Twin Critic via online RL (Stage 2). This is the canonical
configuration of the RL-Token-Pi05 pipeline on SO-101:

- ``z_rl`` dim 2048 (matches the frozen VLA token dim, per the paper)
- configurable state/action dimensions (default: dual Piper's 14 dimensions)
- RL chunk = ``n_action_steps_rl = 10`` steps, subsampled with
  ``action_stride = 2`` for overlapping replay collection; references remain contiguous
- paper full-output MLP actor/critic with fixed std ``policy_fixed_std``
  (no residual mode)

.. note::
   The earlier reference-projection actor/critic fields (``actor_hidden_dims``
   list, ``ref_dropout``, ``actor_residual_scale``) were removed: the paper
   pipeline always trains the full-output MLP heads and the SmolVLA-oriented
   variant is not maintained anymore.
"""

import math
from dataclasses import dataclass, field

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("pi05_rlt")
@dataclass
class PI05RLTConfig(PreTrainedConfig):
    # ── π0.5 base model ──────────────────────────────────────────────────
    pi05_pretrained_path: str = ""  # Path to pretrained π0.5 checkpoint
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: torch.dtype | None = torch.float32

    # These mirror PI05Config defaults for the frozen π0.5
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"
    num_inference_steps: int = 10
    n_obs_steps: int = 1

    # ── RLT encoder-decoder (trained contract: keep dims aligned with the
    #    Stage 1 checkpoints produced by train_rlt_stage1_pi05.py) ────────
    rlt_architecture: str = "paper_v1"
    vlm_hidden_dim: int = 2048  # Gemma 2B hidden size (VLA token dim)
    expert_hidden_dim: int = 1024  # Gemma 300M hidden size
    rlt_hidden_dim: int = 2048  # z_rl dimension (match VLA dim, per paper)
    rlt_encoder_layers: int = 4
    rlt_decoder_layers: int = 4
    rlt_num_heads: int = 8
    rlt_dropout: float = 0.1

    # ── Paper full-output actor MLP ──────────────────────────────────────
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2

    # ── Twin critic MLP ──────────────────────────────────────────────────
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2

    # Fixed exploration std of the Gaussian actor (paper: small fixed std)
    policy_fixed_std: float = 0.05

    # ── Configurable action space (dual Piper defaults) ─────────────────
    action_dim: int = 14  # left J1–J6, left gripper, right J1–J6, right gripper
    state_dim: int = 14
    n_action_steps_rl: int = 10  # RL chunk executed per policy call
    action_stride: int = 2  # overlapping replay window start stride
    # Optional canonical feature names for the trained action
    # order. Declared so export_rlt_pretrained can persist it into config.json
    # and rollout boundaries can validate/reorder before touching hardware.
    action_feature_names: list[str] | None = None
    # Export-contract fields (written by scripts/export_rlt_pretrained.py).
    # The local modeling is hard-wired to the paper contract (full-output MLP
    # actor, no residual head), so these are descriptive round-trip fields:
    # they must exist so a saved config.json loads back via from_pretrained.
    actor_critic_style: str = "paper_mlp"
    actor_residual_scale: float = 0.0

    # ── RL hyperparameters used by the policy helpers (Stage 2) ──────────
    discount: float = 0.99
    bc_weight: float = 10.0  # reference Stage-2 script's online_bc_weight
    target_tau: float = 0.005

    # ── Deployment behavior (lerobot record/rollout) ─────────────────────
    # All fields default OFF so plain loading keeps the legacy behavior
    # (always actor-refined chunk on each window). When enabled, the policy
    # replicates the Stage-2 online-RL deployment semantics at window
    # granularity: auto-critical detection on chunk boundaries (VLA <-> actor
    # switching) and a sticky manual actor gate.
    deploy_auto_critical: bool = False
    deploy_gating: bool = False  # manual actor/critical gates, without requiring a classifier
    critical_classifier_path: str = ""  # legacy .pt checkpoint (CriticalPhaseDetector format)
    auto_critical_threshold_on: float = 0.75
    auto_critical_threshold_off: float = 0.45
    auto_critical_smooth_steps: int = 8
    auto_critical_min_on_chunks: int = 3
    auto_critical_start_delay_steps: int = 0  # control steps before detection arms
    auto_critical_once: bool = True  # only one auto ON per episode
    auto_critical_override_cooldown: float = 3.0  # s after a manual 'c'
    actor_critical_delay_steps: int = 0  # sustained critical steps before actor takes over
    deploy_vla_only: bool = False  # never let the actor take over

    # ── Training mode ────────────────────────────────────────────────────
    mode: str = "rlt_training"

    # ── Stage 1 optimizer/schedule (used by get_optimizer_preset) ────────
    rlt_lr: float = 1e-4
    rlt_weight_decay: float = 1e-5
    rlt_warmup_steps: int = 200
    rlt_total_steps: int = 5000

    # ── Normalization ────────────────────────────────────────────────────
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Paths for loading stage checkpoints
    rlt_encoder_path: str = ""
    rlt_decoder_path: str = ""
    actor_path: str = ""
    critic_path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "vlm_hidden_dim",
            "expert_hidden_dim",
            "rlt_hidden_dim",
            "rlt_num_heads",
            "rlt_encoder_layers",
            "rlt_decoder_layers",
            "action_dim",
            "state_dim",
            "max_action_dim",
            "max_state_dim",
            "chunk_size",
            "num_inference_steps",
            "n_action_steps_rl",
            "action_stride",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "actor_num_layers",
            "critic_num_layers",
            "rlt_total_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("policy_fixed_std", "rlt_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("discount", "target_tau"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.rlt_dropout < 1 or not math.isfinite(self.rlt_dropout):
            raise ValueError("rlt_dropout must be in [0, 1)")
        if not math.isfinite(self.bc_weight) or self.bc_weight < 0:
            raise ValueError("bc_weight must be finite and nonnegative")
        if self.n_action_steps_rl > self.chunk_size:
            raise ValueError("RL action horizon cannot exceed the frozen VLA chunk_size")
        if self.n_obs_steps != 1:
            raise ValueError("Reference RLT requires n_obs_steps=1")
        if self.actor_critic_style != "paper_mlp" or self.actor_residual_scale != 0:
            raise ValueError("Reference RLT uses full-output paper_mlp heads, without residual scaling")
        if self.action_feature_names is not None and (
            len(self.action_feature_names) != self.action_dim
            or len(set(self.action_feature_names)) != self.action_dim
        ):
            raise ValueError("action_feature_names must contain action_dim unique names")
        if self.mode not in ("rlt_training", "online_rl", "inference"):
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: rlt_training, online_rl, inference"
            )
        if self.rlt_architecture != "paper_v1":
            raise ValueError(f"Unsupported rlt_architecture '{self.rlt_architecture}'. Expected 'paper_v1'.")
        if self.rlt_hidden_dim % self.rlt_num_heads != 0:
            raise ValueError("rlt_hidden_dim must be divisible by rlt_num_heads")
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) cannot exceed max_action_dim ({self.max_action_dim})"
            )
        if self.state_dim > self.max_state_dim:
            raise ValueError(
                f"state_dim ({self.state_dim}) cannot exceed max_state_dim ({self.max_state_dim})"
            )
        if self.action_dim <= 0 or self.state_dim <= 0:
            raise ValueError("action_dim and state_dim must be positive")
        if self.n_action_steps_rl < 1:
            raise ValueError("n_action_steps_rl must be at least 1")
        if self.action_stride < 1:
            raise ValueError("action_stride must be at least 1")
        if self.actor_num_layers < 1 or self.critic_num_layers < 1:
            raise ValueError("actor_num_layers and critic_num_layers must be at least 1")
        if self.actor_hidden_dim <= 0 or self.critic_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim and critic_hidden_dim must be positive")
        if self.policy_fixed_std <= 0:
            raise ValueError("policy_fixed_std must be positive")
        if self.deploy_auto_critical:
            if not self.critical_classifier_path:
                raise ValueError(
                    "deploy_auto_critical requires critical_classifier_path "
                    "(a trained CriticalPhaseDetector .pt checkpoint)"
                )
            if not (0.0 <= self.auto_critical_threshold_off < self.auto_critical_threshold_on <= 1.0):
                raise ValueError(
                    "auto_critical thresholds must satisfy 0 <= off < on <= 1 "
                    f"(got off={self.auto_critical_threshold_off}, on={self.auto_critical_threshold_on})"
                )
            if self.auto_critical_smooth_steps < 1 or self.auto_critical_min_on_chunks < 0:
                raise ValueError("auto_critical_smooth_steps/min_on_chunks out of range")
            if self.auto_critical_override_cooldown < 0:
                raise ValueError("auto_critical_override_cooldown must be >= 0")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.rlt_lr,
            weight_decay=self.rlt_weight_decay,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.rlt_lr,
            decay_lr=0.0,
            num_warmup_steps=self.rlt_warmup_steps,
            num_decay_steps=self.rlt_total_steps,
        )

    def validate_features(self) -> None:
        from lerobot.utils.constants import ACTION, OBS_STATE

        for key, features, expected in (
            (OBS_STATE, self.input_features, self.state_dim),
            (ACTION, self.output_features, self.action_dim),
        ):
            if features and key in features and features[key].shape != (expected,):
                raise ValueError(f"{key} shape must be {(expected,)}, got {features[key].shape}")

    def validate_backbone_config(self, base: PreTrainedConfig) -> None:
        """Reject a backbone whose action, camera, or prompt contract differs."""
        if base.type != "pi05":
            raise ValueError("Frozen backbone must be a PI05 configuration")
        for key in (
            "chunk_size",
            "max_action_dim",
            "max_state_dim",
            "image_resolution",
            "paligemma_variant",
            "action_expert_variant",
            "tokenizer_max_length",
        ):
            if getattr(base, key) != getattr(self, key):
                raise ValueError(f"Frozen VLA {key} does not match RLT configuration")
        if any(
            getattr(base, key, False)
            for key in ("use_relative_actions", "use_visual_memory", "use_proprioceptive_memory")
        ):
            raise ValueError("RLT requires an absolute-action PI05 without observation memory")
        for key, features, expected in (
            ("observation.state", base.input_features, self.state_dim),
            ("action", base.output_features, self.action_dim),
        ):
            if not features or key not in features or features[key].shape != (expected,):
                raise ValueError(f"Frozen VLA {key} must have shape {(expected,)}")
        if self.input_features and list(base.image_features) != list(self.image_features):
            raise ValueError("Frozen VLA camera features/order differ from RLT configuration")
        if base.normalization_mapping != self.normalization_mapping:
            raise ValueError("Frozen VLA normalization differs from RLT configuration")
        names = getattr(base, "action_feature_names", None)
        if names and self.action_feature_names and names != self.action_feature_names:
            raise ValueError("Frozen VLA action order differs from RLT configuration")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.n_action_steps_rl))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def n_action_steps(self) -> int:
        return self.n_action_steps_rl
