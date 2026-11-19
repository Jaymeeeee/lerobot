#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action-sequence diffusion Transformer with AdaLN-Zero conditioning.

The model follows the DiT conditioning design (https://arxiv.org/abs/2212.09748).
Each token represents one action timestep. Observation features are supplied by
the existing Diffusion Policy encoders, so visual processing and noise scheduling
remain independent of this denoising backbone.
"""

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from .configuration_diffusion import DiffusionConfig


def _sinusoidal_embedding(positions: Tensor, dim: int) -> Tensor:
    """Encode scalar positions in float32, including odd embedding dimensions."""
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half_dim, device=positions.device, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )
    phases = positions.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((phases.sin(), phases.cos()), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(positions).float().unsqueeze(-1)), dim=-1)
    return embedding


def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionDiTBlock(nn.Module):
    """Bidirectional self-attention and MLP, each with adaptive normalization."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # Shift, scale, and residual gate for each of the two sublayers.
        self.conditioning = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = self.conditioning(condition).chunk(
            6, dim=-1
        )
        normalized = _modulate(self.attention_norm(tokens), attn_shift, attn_scale)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attn_gate.unsqueeze(1) * attended
        normalized = _modulate(self.mlp_norm(tokens), mlp_shift, mlp_scale)
        return tokens + mlp_gate.unsqueeze(1) * self.mlp(normalized)


class DiffusionConditionalDiT(nn.Module):
    """Conditionally denoise an action trajectory with shape ``(batch, time, action_dim)``.

    The input and output sequence lengths match and may range from one to the
    configured horizon. Attention is noncausal: all noisy action tokens jointly
    predict either noise or clean actions, as chosen by the external scheduler.
    No variance prediction or observation tokenization is performed here.
    """

    def __init__(self, config: "DiffusionConfig", global_cond_dim: int) -> None:
        super().__init__()
        action_feature = config.action_feature
        if action_feature is None:
            raise ValueError("Diffusion DiT requires an action output feature.")
        self.config = config
        self.action_dim = action_feature.shape[0]
        self.global_cond_dim = global_cond_dim
        hidden_dim = config.dit_hidden_dim

        self.action_projection = nn.Linear(self.action_dim, hidden_dim)
        self.register_buffer(
            "position_embedding",
            _sinusoidal_embedding(torch.arange(config.horizon), hidden_dim).unsqueeze(0),
        )
        self.timestep_encoder = nn.Sequential(
            nn.Linear(config.diffusion_step_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.observation_encoder = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiffusionDiTBlock(hidden_dim, config.dit_num_heads, config.dit_mlp_ratio, config.dit_dropout)
                for _ in range(config.dit_num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_conditioning = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Each Transformer block starts as the identity. The final prediction
        # starts at zero, following AdaLN-Zero rather than a random denoiser.
        for block in self.blocks:
            nn.init.zeros_(block.conditioning[-1].weight)
            nn.init.zeros_(block.conditioning[-1].bias)
        nn.init.zeros_(self.final_conditioning[-1].weight)
        nn.init.zeros_(self.final_conditioning[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        """Predict a trajectory from noisy actions, diffusion time, and observations.

        ``timestep`` accepts a scalar, a one-element vector, or one value per
        batch item. ``global_cond`` has shape ``(batch, global_cond_dim)`` and
        contains the existing policy's flattened observation history.
        """
        if x.ndim != 3 or x.shape[-1] != self.action_dim:
            raise ValueError(f"Expected actions of shape (B, T, {self.action_dim}), got {tuple(x.shape)}.")
        batch_size, sequence_length, _ = x.shape
        if not 1 <= sequence_length <= self.config.horizon:
            raise ValueError(
                f"Action sequence length must be between 1 and {self.config.horizon}, got {sequence_length}."
            )
        timesteps = torch.as_tensor(timestep, device=x.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.ndim == 1 and timesteps.shape[0] == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(f"Expected scalar timestep or shape ({batch_size},), got {tuple(timesteps.shape)}.")

        tokens = self.action_projection(x)
        tokens = tokens + self.position_embedding[:, :sequence_length].to(dtype=tokens.dtype)
        time_embedding = _sinusoidal_embedding(timesteps, self.config.diffusion_step_embed_dim)
        condition = self.timestep_encoder(time_embedding.to(dtype=self.timestep_encoder[0].weight.dtype))
        if global_cond is not None:
            if global_cond.shape != (batch_size, self.global_cond_dim):
                raise ValueError(
                    f"Expected global_cond of shape ({batch_size}, {self.global_cond_dim}), "
                    f"got {tuple(global_cond.shape)}."
                )
            condition = condition + self.observation_encoder(global_cond)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)

        shift, scale = self.final_conditioning(condition).chunk(2, dim=-1)
        return self.output_projection(_modulate(self.final_norm(tokens), shift, scale))
