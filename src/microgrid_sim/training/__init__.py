# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

"""Training-side utilities for safe / offline RL experiments."""

from .bc_warmstart import (
    BCWarmstartReport,
    OfflineTransitionBatch,
    apply_bc_warmstart,
    behavior_clone_sac_actor,
    evaluate_actor_behavior_cloning_mse,
    load_offline_dataset,
    offline_dataset_to_transition_batch,
    seed_replay_buffer_from_offline_dataset,
)
from .shield import BatteryActionShieldConfig, BatteryActionShieldDecision, shield_battery_action

__all__ = [
    "BCWarmstartReport",
    "BatteryActionShieldConfig",
    "BatteryActionShieldDecision",
    "OfflineTransitionBatch",
    "apply_bc_warmstart",
    "behavior_clone_sac_actor",
    "evaluate_actor_behavior_cloning_mse",
    "load_offline_dataset",
    "offline_dataset_to_transition_batch",
    "seed_replay_buffer_from_offline_dataset",
    "shield_battery_action",
]
