import numpy as np
import torch

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.replay import Transition, TransitionSource


def config(**kwargs):
    values = {
        "device": "cpu",
        "mode": "online_rl",
        "vlm_hidden_dim": 8,
        "rlt_hidden_dim": 8,
        "rlt_num_heads": 2,
        "rlt_encoder_layers": 1,
        "rlt_decoder_layers": 1,
        "rlt_dropout": 0.0,
        "actor_hidden_dim": 16,
        "critic_hidden_dim": 16,
        "chunk_size": 6,
        "n_action_steps_rl": 3,
    }
    return PI05RLTConfig(**(values | kwargs))


def transition(index=0, partial=False, action_dim=14):
    rng = np.random.default_rng(index)
    valid = np.array([True, True, not partial])
    actions = rng.normal(size=(3, action_dim)).astype("float32")
    actions[~valid] = 0
    return Transition(
        z_rl=np.full(8, index, dtype="float32"),
        proprio=np.zeros(14, dtype="float32"),
        ref_chunk=np.ones((3, action_dim), dtype="float32"),
        action_chunk=actions,
        rewards=np.array([0, 1, 0] if partial else [0, 0, 1], dtype="float32"),
        done=True,
        next_z_rl=np.full(8, index + 1, dtype="float32"),
        next_proprio=np.ones(14, dtype="float32"),
        next_ref_chunk=np.ones((3, action_dim), dtype="float32"),
        source=int(TransitionSource.MIXED),
        source_chunk=np.array([0, 2, 0], dtype="uint8"),
        valid_mask=valid,
        executed_steps=int(valid.sum()),
        collection_phase="warmup" if index % 2 else "online",
        episode_id=index,
        step_id=0,
        intervention_flag=True,
    )


def zero(module):
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
