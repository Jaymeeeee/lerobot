"""Opt-in paired RGB and metric-depth inputs for ACT and Diffusion.

The saved processors normalize RGB and depth separately. Models concatenate the
normalized channels before spatial transforms and use one four-channel backbone
per original camera (ACT retains its shared backbone).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.configs.types import FeatureType, PolicyFeature

if TYPE_CHECKING:
    from torch import Tensor, nn
    from lerobot.configs.policies import PreTrainedConfig


def rgb_image_features(
    input_features: dict[str, PolicyFeature] | None, depth_camera_map: dict[str, str]
) -> dict[str, PolicyFeature]:
    depth_keys = set(depth_camera_map.values())
    return {
        key: feature for key, feature in (input_features or {}).items()
        if feature.type is FeatureType.VISUAL and key not in depth_keys
    }


def validate_depth_camera_map(config: PreTrainedConfig) -> None:
    mapping = config.depth_camera_map
    if not mapping:
        return
    if config.depth_unit != "m":
        raise ValueError("RGB-D policy inputs must use metric depth in metres (depth_unit='m').")
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("depth_camera_map must map RGB feature names to depth feature names.")
    if len(set(mapping.values())) != len(mapping) or set(mapping) & set(mapping.values()):
        raise ValueError("Each RGB camera requires its own distinct depth feature.")
    if set(mapping) != set(config.image_features):
        raise ValueError("depth_camera_map must cover every RGB camera, without adding cameras.")
    for rgb_key, depth_key in mapping.items():
        if depth_key not in config.input_features:
            raise ValueError(f"Missing depth input feature: {depth_key}")
        rgb, depth = config.input_features[rgb_key], config.input_features[depth_key]
        if depth.type is not FeatureType.VISUAL or len(depth.shape) != 3 or depth.shape[0] != 1:
            raise ValueError(f"{depth_key} must be a one-channel VISUAL feature.")
        if len(rgb.shape) != 3 or rgb.shape[0] != 3 or tuple(rgb.shape[1:]) != tuple(depth.shape[1:]):
            raise ValueError(f"RGB and depth must be aligned 3xHxW and 1xHxW: {rgb_key}, {depth_key}")


def get_camera_tensors(config: PreTrainedConfig, batch: dict[str, Tensor]) -> list[Tensor]:
    import torch

    images = []
    for key in config.image_features:
        rgb = batch[key]
        if not config.depth_camera_map:
            images.append(rgb)
            continue
        depth_key = config.depth_camera_map[key]
        if depth_key not in batch:
            raise KeyError(f"RGB-D observation is missing {depth_key}")
        depth = batch[depth_key]
        if (rgb.ndim < 3 or depth.ndim != rgb.ndim or rgb.shape[-3] != 3 or depth.shape[-3] != 1
                or rgb.shape[:-3] != depth.shape[:-3] or rgb.shape[-2:] != depth.shape[-2:]):
            raise ValueError(f"RGB/depth batch or spatial dimensions do not align: {key}={tuple(rgb.shape)}, "
                             f"{depth_key}={tuple(depth.shape)}")
        if not rgb.is_floating_point() or not depth.is_floating_point():
            raise TypeError("RGB-D model inputs must be separately normalized floating-point tensors.")
        images.append(torch.cat((rgb, depth), dim=-3))
    return images


def make_rgbd_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """Preserve RGB weights and initialize depth with their channel mean.

    Preserve the RNG stream so the added input channel does not shift the
    initialization of the remaining unchanged architecture.
    """
    import torch
    from torch import nn

    if conv.in_channels != 3 or conv.groups != 1:
        raise ValueError("RGB-D conversion requires a standard three-channel input convolution.")
    devices = [conv.weight.device.index] if conv.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        expanded = nn.Conv2d(
            4, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
            bias=conv.bias is not None, padding_mode=conv.padding_mode,
            device=conv.weight.device, dtype=conv.weight.dtype,
        )
    with torch.no_grad():
        expanded.weight[:, :3].copy_(conv.weight)
        expanded.weight[:, 3:4].copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            expanded.bias.copy_(conv.bias)
    expanded.weight.requires_grad_(conv.weight.requires_grad)
    if conv.bias is not None:
        expanded.bias.requires_grad_(conv.bias.requires_grad)
    expanded.train(conv.training)
    return expanded
