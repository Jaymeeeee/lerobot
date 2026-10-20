import json
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05_rlt.critical import CriticalPhaseClassifier, CriticalPhaseDetector
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.policies.pi05_rlt.training import encode_policy_config, load_prefix_cache
from lerobot.policies.pi05_rlt.vla_compat import extract_embeddings
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .helpers import config, zero


class Backbone(nn.Module):
    """Tiny public PI05 API stub; does not load pretrained weights."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.calls = 0

    def embed_prefix(self, *args):
        raise NotImplementedError

    def embed_suffix(self, *args):
        raise NotImplementedError

    def sample_actions(self, images, image_masks, tokens, masks, num_steps):
        self.calls += 1
        return torch.arange(6, device=tokens.device).float()[None, :, None].expand(len(tokens), 6, 32)

    def extract_embeddings(self, images, image_masks, tokens, masks, actions, **kwargs):
        size = len(tokens)
        return torch.ones(size, 4, 8, device=tokens.device), None, torch.ones(size, 4, dtype=torch.bool)


def observation():
    return {
        "observation.state": torch.zeros(1, 14),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }


def test_factory_config_safe_checkpoint_and_pretrained_roundtrip(tmp_path):
    cfg = config()
    encoded = encode_policy_config(cfg)
    assert encoded["type"] == "pi05_rlt"
    assert encoded["dtype"] == "float32"
    assert draccus.decode(PreTrainedConfig, encoded) == cfg
    assert get_policy_class("pi05_rlt") is PI05RLTPolicy
    policy = PI05RLTPolicy(cfg)
    policy.load_frozen_vla(Backbone())
    policy.save_pretrained(tmp_path)
    restored = PI05RLTPolicy.from_pretrained(tmp_path, local_files_only=True)
    assert restored.config == cfg
    assert not restored._vla_loaded
    assert not any("anchor" in key or "frozen_vla" in key for key in restored.state_dict())
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_backbone_frozen_excluded_and_follows_moves():
    policy = PI05RLTPolicy(config())
    base = Backbone()
    policy.load_frozen_vla(base)
    policy.train()
    assert not base.training and not base.anchor.requires_grad
    assert not policy.rlt_encoder.training and not policy.actor_target.training
    assert all(p.requires_grad for p in policy.actor.parameters())
    policy.to(dtype=torch.float64)
    assert base.anchor.dtype == torch.float64


def test_backbone_contract_rejects_wrong_dimensions_cameras_and_relative_actions():
    cfg = config(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
            "observation.images.cam_h": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
    )
    base = PI05Config(
        device="cpu",
        chunk_size=6,
        n_action_steps=3,
        input_features=cfg.input_features.copy(),
        output_features=cfg.output_features.copy(),
    )
    cfg.validate_backbone_config(base)
    base.output_features["action"] = PolicyFeature(FeatureType.ACTION, (6,))
    with pytest.raises(ValueError, match="action"):
        cfg.validate_backbone_config(base)
    base.output_features = cfg.output_features.copy()
    base.input_features.pop("observation.images.cam_h")
    with pytest.raises(ValueError, match="camera"):
        cfg.validate_backbone_config(base)
    base.input_features = cfg.input_features.copy()
    base.use_relative_actions = True
    with pytest.raises(ValueError, match="absolute-action"):
        cfg.validate_backbone_config(base)


def test_prediction_chunk_queue_gates_and_contiguous_reference():
    policy = PI05RLTPolicy(config(mode="inference", deploy_gating=True, actor_critical_delay_steps=0))
    base = Backbone()
    policy.load_frozen_vla(base)
    zero(policy.actor)
    spec = policy.chunk_inference_spec()
    assert spec.prediction_steps == 3
    batch = observation()
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (1, 3, 14)
    torch.testing.assert_close(chunk[0, :, 0], torch.arange(3).float())
    policy.select_action(batch)
    assert len(policy._queues["action"]) == 2
    policy.set_actor_enabled(True)
    policy.set_critical_manual(True)
    assert not policy._queues["action"]
    torch.testing.assert_close(policy.select_action(batch), torch.zeros(1, 14))
    policy.reset()
    assert policy.actor_enabled and not policy.critical_active
    policy.config.deploy_vla_only = True
    policy.set_critical_manual(True)
    torch.testing.assert_close(policy.predict_action_chunk(batch), chunk)


def test_stage1_processor_preserves_embedding_keys():
    cfg = config(mode="rlt_training")
    pre, _ = make_pre_post_processors(cfg)
    batch = {"vlm_embeddings": torch.randn(2, 4, 8), "prefix_mask": torch.ones(2, 4, dtype=torch.bool)}
    actual = pre(batch)
    assert set(batch) <= set(actual)
    for key in batch:
        torch.testing.assert_close(actual[key], batch[key])


def test_inference_processor_delegates_pi05_contract(monkeypatch):
    import lerobot.policies.pi05_rlt.processor_pi05_rlt as processors

    cfg = config()
    cfg.input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    captured = {}

    def maker(base, stats):
        captured.update(base=base, stats=stats)
        return "pre", "post"

    monkeypatch.setattr(processors, "make_pi05_pre_post_processors", maker)
    assert processors.make_pi05_rlt_pre_post_processors(cfg) == ("pre", "post")
    base = captured["base"]
    assert base.n_action_steps == 3 and base.chunk_size == 6
    assert base.input_features == cfg.input_features
    assert base.text_tokenizer_name == cfg.text_tokenizer_name


def test_camera_validation_and_uint8_normalization():
    cfg = config(
        input_features={"observation.images.cam_h": PolicyFeature(FeatureType.VISUAL, (3, 224, 224))}
    )
    policy = PI05RLTPolicy(cfg)
    with pytest.raises(ValueError, match="camera feature"):
        policy._prepare_images({})
    images, masks = policy._prepare_images(
        {"observation.images.cam_h": torch.full((1, 3, 224, 224), 255, dtype=torch.uint8)}
    )
    assert images[0].min() == 1 and masks[0].all()


def test_current_pi05_api_extracts_post_transformer_image_positions():
    class Expert:
        paligemma = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            self_attn=SimpleNamespace(q_proj=SimpleNamespace(weight=torch.zeros(1)))
                        )
                    ]
                )
            )
        )

        def forward(self, **kwargs):
            # Contextualization stands in for the real PaliGemma transformer.
            assert kwargs["attention_mask"].shape == (1, 1, 8, 8)
            assert kwargs["position_ids"].shape == (1, 8)
            return [value + 7 for value in kwargs["inputs_embeds"]], None

    class CurrentBackbone:
        paligemma_with_expert = Expert()

        def embed_prefix(self, *args):
            return (
                torch.ones(1, 5, 8),
                torch.tensor([[True, True, False, True, True]]),
                torch.zeros(1, 5, dtype=torch.bool),
            )

        def embed_suffix(self, actions, time):
            return (
                torch.zeros(1, 3, 8),
                torch.ones(1, 3, dtype=torch.bool),
                torch.ones(1, 3, dtype=torch.bool),
                None,
            )

    prefix, suffix, mask = extract_embeddings(
        CurrentBackbone(),
        [],
        [],
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, 32),
        chunk_size=3,
        max_action_dim=32,
        noise=torch.zeros(1, 3, 32),
        time=torch.ones(1),
        image_only=True,
    )
    assert prefix.shape == (1, 3, 8) and suffix.shape == (1, 3, 8)
    assert prefix.min() == 8
    assert mask.tolist() == [[True, True, False]]


def write_cache(root, n=8):
    root.mkdir()
    meta = {
        "complete": True,
        "image_only": True,
        "has_mask": True,
        "embedding_stage": "post_transformer",
        "dtype": "float16",
        "n_samples": n,
        "seq_len": 4,
        "vlm_hidden_dim": 8,
    }
    (root / "meta.json").write_text(json.dumps(meta))
    np.ones((n, 4, 8), dtype=np.float16).tofile(root / "prefix_out.mmap")
    np.ones((n, 4), dtype=np.bool_).tofile(root / "prefix_mask.mmap")
    return meta


def test_cache_shards_masks_and_truncation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_cache(a)
    write_cache(b, n=3)
    dataset, _ = load_prefix_cache([a, b])
    assert len(dataset) == 11
    assert dataset[8][0].shape == (4, 8)
    assert dataset[-1][1].all()
    (b / "prefix_mask.mmap").write_bytes(b"short")
    with pytest.raises(ValueError, match="byte length"):
        load_prefix_cache([a, b])


def test_critical_detector_resets_hold_timer_and_validates_dimensions(tmp_path):
    model = CriticalPhaseClassifier(z_dim=8, proprio_dim=14, hidden_dim=8)
    zero(model)
    model.net[-1].bias.data[1] = 5
    path = tmp_path / "critical.pt"
    torch.save(
        {
            "z_dim": 8,
            "proprio_dim": 14,
            "hidden_dim": 8,
            "num_layers": 2,
            "classifier_state_dict": model.state_dict(),
            "proprio_mean": torch.zeros(14),
            "proprio_std": torch.ones(14),
        },
        path,
    )
    detector = CriticalPhaseDetector(path, "cpu", smooth_steps=2, min_on_chunks=3)
    assert detector.update(np.zeros(8), np.zeros(14)) is True
    assert detector.update(np.zeros(8), np.zeros(14)) is True
    detector.classifier.net[-1].bias.data[1] = -5
    assert detector.update(np.zeros(8), np.zeros(14)) is None
    assert detector.update(np.zeros(8), np.zeros(14)) is False
    detector.reset()
    assert detector.last_probability is None and detector._on_since_update is None
    with pytest.raises(ValueError, match="dimension"):
        detector.update(np.zeros(8), np.zeros(6))
