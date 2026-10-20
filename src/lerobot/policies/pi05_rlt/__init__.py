# Copyright 2026 The RL-Token-Pi05 authors.
# Adapted for AgileX_ws; changes documented in NOTICE and README.md.
# SPDX-License-Identifier: Apache-2.0

from .configuration_pi05_rlt import PI05RLTConfig

__all__ = ["PI05RLTConfig", "PI05RLTPolicy", "make_pi05_rlt_pre_post_processors"]


def __getattr__(name: str):
    if name == "PI05RLTPolicy":
        from .modeling_pi05_rlt import PI05RLTPolicy

        return PI05RLTPolicy
    if name == "make_pi05_rlt_pre_post_processors":
        from .processor_pi05_rlt import make_pi05_rlt_pre_post_processors

        return make_pi05_rlt_pre_post_processors
    raise AttributeError(name)
