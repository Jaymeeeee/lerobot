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
"""CPU checks for conditional Transformer diffusion and legacy U-Net compatibility."""

import json

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionPolicy
from lerobot.policies.diffusion.modeling_dit import DiffusionConditionalDiT
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

pytest.importorskip("diffusers")


def make_config(**overrides):
    options = {
        "device": "cpu",
        "denoiser_type": "dit",
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        "n_obs_steps": 2,
        "horizon": 7,
        "n_action_steps": 4,
        "drop_n_last_frames": 0,
        "dit_hidden_dim": 32,
        "dit_num_layers": 2,
        "dit_num_heads": 4,
        "dit_mlp_ratio": 2.0,
        "dit_dropout": 0.0,
        "diffusion_step_embed_dim": 16,
        "down_dims": (16, 32),
        "n_groups": 4,
        "num_train_timesteps": 8,
        "num_inference_steps": 4,
        "pretrained_backbone_weights": None,
        "push_to_hub": False,
    }
    options.update(overrides)
    return DiffusionConfig(**options)


def make_batch(config, batch_size=2):
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(batch_size, config.n_obs_steps, 2),
        ACTION: torch.randn(batch_size, config.horizon, 2).clamp(-1, 1),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_dit_training_reaches_attention_and_observation_inputs(gradient_checkpointing, prediction_type):
    torch.manual_seed(17)
    config = make_config(gradient_checkpointing=gradient_checkpointing, prediction_type=prediction_type)
    policy = DiffusionPolicy(config).train()
    assert isinstance(policy.diffusion.denoiser, DiffusionConditionalDiT)
    batch = make_batch(config)
    batch[OBS_STATE].requires_grad_()
    batch[OBS_ENV_STATE].requires_grad_()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # AdaLN-Zero starts with closed residual gates and a zero output projection;
    # allow both to learn before requiring gradients in attention and conditioning.
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        batch[OBS_STATE].grad = None
        batch[OBS_ENV_STATE].grad = None
        loss, metrics = policy(batch)
        assert metrics is None
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    attention_parameters = [
        parameter
        for module in policy.diffusion.denoiser.modules()
        if isinstance(module, nn.MultiheadAttention)
        for parameter in module.parameters()
    ]
    assert attention_parameters, "The denoiser must contain multihead self-attention."
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attention_parameters
    )
    assert any(parameter.grad.abs().sum() > 0 for parameter in attention_parameters)
    for key in (OBS_STATE, OBS_ENV_STATE):
        assert torch.isfinite(batch[key].grad).all()
        assert batch[key].grad.abs().sum() > 0

    # Every output action may use future noisy action tokens; diffusion denoising
    # is bidirectional, rather than autoregressive next-action prediction.
    policy.eval()
    trajectory = torch.randn(2, config.horizon, 2, requires_grad=True)
    condition = policy.diffusion._prepare_global_conditioning(batch).detach()
    output = policy.diffusion.denoiser(trajectory, torch.tensor([2, 5]), global_cond=condition)
    output[:, 0].sum().backward()
    assert trajectory.grad[:, -1].abs().sum() > 0
    changed_condition = policy.diffusion.denoiser(trajectory.detach(), torch.tensor([2, 5]), condition + 1)
    changed_step = policy.diffusion.denoiser(trajectory.detach(), torch.tensor([3, 6]), condition)
    assert not torch.equal(output, changed_condition)
    assert not torch.equal(output, changed_step)


@pytest.mark.parametrize("scheduler", ["DDPM", "DDIM"])
@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_dit_sampling_uses_transformer_and_preserves_action_queue(scheduler, prediction_type):
    torch.manual_seed(9)
    config = make_config(noise_scheduler_type=scheduler, prediction_type=prediction_type)
    policy = DiffusionPolicy(config).eval()
    batch = make_batch(config)
    observations = {key: batch[key] for key in (OBS_STATE, OBS_ENV_STATE)}
    noise = torch.randn(2, config.horizon, 2)
    calls = []
    handle = policy.diffusion.denoiser.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output.shape)
    )
    try:
        torch.manual_seed(42)
        expected = policy.predict_action_chunk(observations, noise=noise)
        assert expected.shape == (2, config.n_action_steps, 2)
        assert torch.isfinite(expected).all()
        assert calls == [torch.Size((2, config.horizon, 2))] * config.num_inference_steps

        policy.reset()
        calls.clear()
        # Queue initialization repeats the first observation for the full history.
        current = {key: value[:, -1] for key, value in observations.items()}
        repeated = {
            key: value.unsqueeze(1).expand(-1, config.n_obs_steps, -1) for key, value in current.items()
        }
        torch.manual_seed(42)
        expected = policy.predict_action_chunk(repeated, noise=noise)
        calls.clear()
        torch.manual_seed(42)
        actual = torch.stack(
            [policy.select_action(dict(current), noise=noise) for _ in range(config.n_action_steps)], dim=1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(calls) == config.num_inference_steps
    finally:
        handle.remove()


def test_dit_checkpoint_round_trip_preserves_config_and_predictions(tmp_path):
    torch.manual_seed(13)
    config = make_config(noise_scheduler_type="DDIM")
    policy = DiffusionPolicy(config)
    batch = make_batch(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss, _ = policy(batch)
    loss.backward()
    optimizer.step()
    policy.eval()
    noise = torch.randn(2, config.horizon, 2)
    expected = policy.predict_action_chunk(batch, noise=noise)
    policy.save_pretrained(tmp_path)

    restored_config = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    assert restored_config.denoiser_type == "dit"
    assert restored_config.dit_hidden_dim == config.dit_hidden_dim
    assert restored_config.dit_num_layers == config.dit_num_layers
    restored = DiffusionPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True)
    assert isinstance(restored.diffusion.denoiser, DiffusionConditionalDiT)
    torch.testing.assert_close(restored.predict_action_chunk(batch, noise=noise), expected, rtol=0, atol=0)


def test_legacy_checkpoint_without_dit_fields_loads_same_unet(tmp_path):
    torch.manual_seed(31)
    config = make_config(denoiser_type="unet", horizon=8, noise_scheduler_type="DDIM")
    policy = DiffusionPolicy(config).eval()
    policy.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved_config = json.loads(path.read_text())
    saved_config = {
        key: value
        for key, value in saved_config.items()
        if key != "denoiser_type" and not key.startswith("dit_")
    }
    path.write_text(json.dumps(saved_config))

    restored = DiffusionPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True)
    assert restored.config.denoiser_type == "unet"
    assert isinstance(restored.diffusion.denoiser, DiffusionConditionalUnet1d)
    assert restored.diffusion.denoiser is restored.diffusion.unet
    assert all(key.startswith("diffusion.unet.") for key in restored.state_dict())
    assert restored.state_dict().keys() == policy.state_dict().keys()
    batch = make_batch(config)
    noise = torch.randn(2, config.horizon, 2)
    torch.testing.assert_close(
        restored.predict_action_chunk(batch, noise=noise),
        policy.predict_action_chunk(batch, noise=noise),
        rtol=0,
        atol=0,
    )


def test_dit_accepts_non_downsampled_horizon_while_unet_rejects_it():
    assert make_config(horizon=7).horizon == 7
    with pytest.raises(ValueError, match="horizon"):
        make_config(denoiser_type="unet", horizon=7)


@pytest.mark.parametrize("timestep", [3, torch.tensor(3), torch.tensor([3]), torch.tensor([3, 3])])
def test_dit_scalar_and_batch_timesteps_match(timestep):
    torch.manual_seed(11)
    config = make_config()
    model = DiffusionConditionalDiT(config, global_cond_dim=10).eval()
    x, condition = torch.randn(2, config.horizon, 2), torch.randn(2, 10)
    # Test after one output-layer update so zero initialization cannot hide a
    # broken timestep broadcast or an incorrect positional embedding.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(x, torch.tensor([3, 3]), condition).sub(1).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(model(x, timestep, condition), model(x, torch.tensor([3, 3]), condition))


def test_dit_bfloat16_autocast_has_finite_training_gradients():
    torch.manual_seed(8)
    policy = DiffusionPolicy(make_config(gradient_checkpointing=True)).train()
    batch = make_batch(policy.config)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in policy.parameters()
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
    )


def test_dit_rgbd_training_reaches_both_camera_depth_inputs():
    torch.manual_seed(23)
    rgb_keys = ["observation.images.head", "observation.images.wrist"]
    depth_keys = ["observation.images.head_depth", "observation.images.wrist_depth"]
    inputs = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))}
    inputs.update({key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for key in rgb_keys})
    inputs.update({key: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 64, 64)) for key in depth_keys})
    config = make_config(
        input_features=inputs,
        depth_camera_map=dict(zip(rgb_keys, depth_keys, strict=True)),
        n_obs_steps=1,
        use_separate_rgb_encoder_per_camera=False,
        spatial_softmax_num_keypoints=4,
    )
    policy = DiffusionPolicy(config).train()
    batch = make_batch(config)
    del batch[OBS_ENV_STATE]
    batch.update({key: torch.randn(2, 1, 3, 64, 64, requires_grad=True) for key in rgb_keys})
    batch.update({key: torch.rand(2, 1, 1, 64, 64, requires_grad=True) for key in depth_keys})
    # Updating the small denoiser is enough to open AdaLN-Zero's gates; retain
    # gradients through the real ResNet and paired RGB-D inputs throughout.
    optimizer = torch.optim.Adam(policy.diffusion.denoiser.parameters(), lr=1e-3)
    for _ in range(4):
        policy.zero_grad(set_to_none=True)
        for key in [*rgb_keys, *depth_keys]:
            batch[key].grad = None
        loss, _ = policy(batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for key in [*rgb_keys, *depth_keys]:
        assert torch.isfinite(batch[key].grad).all()
        assert batch[key].grad.abs().sum() > 0
    stem = next(module for module in policy.diffusion.rgb_encoder.modules() if isinstance(module, nn.Conv2d))
    assert stem.in_channels == 4
    assert torch.isfinite(stem.weight.grad).all()
    assert stem.weight.grad[:, 3].abs().sum() > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"denoiser_type": "unknown"},
        {"dit_hidden_dim": 30},
        {"dit_hidden_dim": 0},
        {"dit_num_heads": 0},
        {"dit_num_layers": 0},
        {"dit_mlp_ratio": 0.0},
        {"dit_dropout": -0.1},
        {"dit_dropout": 1.0},
    ],
)
def test_invalid_dit_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)
