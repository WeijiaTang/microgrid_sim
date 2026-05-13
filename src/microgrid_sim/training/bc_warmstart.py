# Ref: docs/spec/task.md (Task-ID: SPEC-SAFE-OFFLINE-RL-001)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces

from .offline_dataset import decode_observation_json
from microgrid_sim.rl_utils import SAFE_REPLAY_META_FIELDS, _ensure_safe_replay_metadata_arrays


REQUIRED_OFFLINE_DATASET_COLUMNS: tuple[str, ...] = (
    "obs",
    "next_obs",
    "action",
    "reward",
    "done",
)


@dataclass(frozen=True)
class OfflineTransitionBatch:
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


@dataclass(frozen=True)
class BCWarmstartReport:
    dataset_rows: int
    replay_seeded_transitions: int
    actor_gradient_steps: int
    actor_batch_size: int
    initial_actor_mse: float
    final_actor_mse: float


@dataclass(frozen=True)
class ReplayDistillationReport:
    replay_rows: int
    actor_gradient_steps: int
    actor_batch_size: int
    initial_actor_mse: float
    final_actor_mse: float
    intervention_rows: int = 0
    inventory_teacher_rows: int = 0
    mean_sample_weight: float = 1.0


def _require_dataset_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OFFLINE_DATASET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Offline dataset is missing required columns: {missing}")


def _normalize_filter_values(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalized or None


def _decode_action_value(value: object) -> np.ndarray:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return np.asarray(json.loads(stripped), dtype=np.float32).reshape(-1)
    return np.asarray([float(value)], dtype=np.float32)


def _as_action_matrix(actions: Iterable[object]) -> np.ndarray:
    decoded = [_decode_action_value(value) for value in actions]
    if not decoded:
        return np.zeros((0, 0), dtype=np.float32)
    action_dim = max(int(array.size) for array in decoded)
    matrix = np.zeros((len(decoded), action_dim), dtype=np.float32)
    for index, array in enumerate(decoded):
        matrix[index, : int(array.size)] = array.reshape(-1)
    return matrix


def load_offline_dataset(
    dataset_source: str | Path | pd.DataFrame,
    *,
    controller_sources: Iterable[str] | None = None,
    cases: Iterable[str] | None = None,
    regimes: Iterable[str] | None = None,
    battery_models: Iterable[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if isinstance(dataset_source, pd.DataFrame):
        frame = dataset_source.copy()
    else:
        frame = pd.read_csv(Path(dataset_source))
    _require_dataset_columns(frame)

    filters = {
        "controller_source": _normalize_filter_values(controller_sources),
        "case": _normalize_filter_values(cases),
        "regime": _normalize_filter_values(regimes),
        "battery_model": _normalize_filter_values(battery_models),
    }
    for column, values in filters.items():
        if values is None or column not in frame.columns:
            continue
        frame = frame.loc[frame[column].astype(str).str.strip().str.lower().isin(values)]
    if limit is not None and int(limit) > 0:
        frame = frame.iloc[: int(limit)]
    return frame.reset_index(drop=True)


def offline_dataset_to_transition_batch(dataset: pd.DataFrame) -> OfflineTransitionBatch:
    _require_dataset_columns(dataset)
    if dataset.empty:
        return OfflineTransitionBatch(
            observations=np.zeros((0, 0), dtype=np.float32),
            next_observations=np.zeros((0, 0), dtype=np.float32),
            actions=np.zeros((0, 0), dtype=np.float32),
            rewards=np.zeros((0,), dtype=np.float32),
            dones=np.zeros((0,), dtype=np.float32),
        )

    observations = np.stack([decode_observation_json(value) for value in dataset["obs"]], axis=0).astype(np.float32)
    next_observations = np.stack([decode_observation_json(value) for value in dataset["next_obs"]], axis=0).astype(np.float32)
    actions = _as_action_matrix(dataset["action"])
    rewards = dataset["reward"].astype(np.float32).to_numpy(copy=True)
    dones = dataset["done"].astype(np.float32).to_numpy(copy=True)
    return OfflineTransitionBatch(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )


def _scale_actions_to_policy_space(agent, actions: np.ndarray) -> np.ndarray:
    action_space = getattr(agent, "action_space", None)
    policy = getattr(agent, "policy", None)
    if not isinstance(action_space, spaces.Box) or policy is None or not hasattr(policy, "scale_action"):
        return np.asarray(actions, dtype=np.float32)
    scaled = [np.asarray(policy.scale_action(action), dtype=np.float32).reshape(-1) for action in np.asarray(actions, dtype=np.float32)]
    return np.asarray(scaled, dtype=np.float32)


def _align_observations_to_agent_space(agent, observations: np.ndarray) -> np.ndarray:
    observation_space = getattr(agent, "observation_space", None)
    if not isinstance(observation_space, spaces.Box):
        return np.asarray(observations, dtype=np.float32)
    target_shape = tuple(int(value) for value in getattr(observation_space, "shape", ()) or ())
    if not target_shape:
        return np.asarray(observations, dtype=np.float32)

    matrix = np.asarray(observations, dtype=np.float32)
    row_count = int(matrix.shape[0]) if matrix.ndim > 0 else 0
    flat = matrix.reshape(row_count, -1) if row_count > 0 else np.zeros((0, int(np.prod(target_shape))), dtype=np.float32)
    target_dim = int(np.prod(target_shape))
    current_dim = int(flat.shape[1]) if flat.ndim == 2 else 0
    if current_dim < target_dim:
        flat = np.pad(flat, ((0, 0), (0, target_dim - current_dim)), mode="constant", constant_values=0.0)
    elif current_dim > target_dim:
        flat = flat[:, :target_dim]
    return flat.reshape((row_count,) + target_shape).astype(np.float32)


def _behavior_clone_sac_actor_from_aligned_arrays(
    agent,
    *,
    observations: np.ndarray,
    target_actions: np.ndarray,
    gradient_steps: int,
    batch_size: int = 256,
    learning_rate: float | None = None,
    shuffle_seed: int = 0,
    sample_weights: np.ndarray | None = None,
    sampling_probabilities: np.ndarray | None = None,
) -> dict[str, float | int]:
    if observations.size == 0 or target_actions.size == 0:
        raise ValueError("Cannot run actor distillation on empty observation/action arrays.")

    device = getattr(agent, "device", "cpu")
    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
    target_tensor = th.as_tensor(target_actions, dtype=th.float32, device=device)
    weight_tensor = None
    if sample_weights is not None:
        normalized_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if normalized_weights.shape[0] != observations.shape[0]:
            raise ValueError("sample_weights must align with observations.")
        normalized_weights = np.clip(normalized_weights, 1e-6, None)
        weight_tensor = th.as_tensor(normalized_weights, dtype=th.float32, device=device)
    actor = getattr(agent, "actor", None)
    if actor is None or not hasattr(actor, "optimizer"):
        raise ValueError("Agent does not expose a SAC-compatible actor optimizer for behavior cloning.")

    def _mse() -> float:
        actor.eval()
        with th.no_grad():
            predictions = actor(obs_tensor, deterministic=True)
            squared_error = th.mean((predictions - target_tensor) ** 2, dim=1)
            if weight_tensor is not None:
                loss = th.sum(weight_tensor * squared_error) / th.sum(weight_tensor)
            else:
                loss = th.mean(squared_error)
        return float(loss.detach().cpu().item())

    if int(gradient_steps) <= 0:
        initial_mse = _mse()
        return {
            "actor_gradient_steps": 0,
            "actor_batch_size": int(max(batch_size, 1)),
            "initial_actor_mse": float(initial_mse),
            "final_actor_mse": float(initial_mse),
        }

    optimizer = actor.optimizer
    old_learning_rates = [group["lr"] for group in optimizer.param_groups]
    if learning_rate is not None and float(learning_rate) > 0.0:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)

    actor.train()
    initial_mse = _mse()
    rng = np.random.default_rng(int(shuffle_seed))
    total_rows = int(obs_tensor.shape[0])
    effective_batch_size = min(max(int(batch_size), 1), total_rows)
    sampling_probs = None
    if sampling_probabilities is not None:
        sampling_probs = np.asarray(sampling_probabilities, dtype=np.float64).reshape(-1)
        if sampling_probs.shape[0] != total_rows:
            raise ValueError("sampling_probabilities must align with observations.")
        sampling_probs = np.clip(sampling_probs, 0.0, None)
        total_prob = float(np.sum(sampling_probs))
        if total_prob > 0.0:
            sampling_probs = sampling_probs / total_prob
        else:
            sampling_probs = None
    for _ in range(int(gradient_steps)):
        if sampling_probs is None:
            indices = rng.integers(0, total_rows, size=effective_batch_size)
        else:
            indices = rng.choice(total_rows, size=effective_batch_size, replace=True, p=sampling_probs)
        obs_batch = obs_tensor[indices]
        action_batch = target_tensor[indices]
        batch_weights = weight_tensor[indices] if weight_tensor is not None else None
        predictions = actor(obs_batch, deterministic=True)
        squared_error = th.mean((predictions - action_batch) ** 2, dim=1)
        if batch_weights is not None:
            loss = th.sum(batch_weights * squared_error) / th.sum(batch_weights)
        else:
            loss = th.mean(squared_error)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_mse = _mse()
    for group, old_lr in zip(optimizer.param_groups, old_learning_rates):
        group["lr"] = old_lr
    return {
        "actor_gradient_steps": int(gradient_steps),
        "actor_batch_size": int(effective_batch_size),
        "initial_actor_mse": float(initial_mse),
        "final_actor_mse": float(final_mse),
    }


def seed_replay_buffer_from_offline_dataset(
    agent,
    dataset_source: str | Path | pd.DataFrame,
    *,
    max_transitions: int | None = None,
    controller_sources: Iterable[str] | None = None,
    cases: Iterable[str] | None = None,
    regimes: Iterable[str] | None = None,
    battery_models: Iterable[str] | None = None,
) -> int:
    replay_buffer = getattr(agent, "replay_buffer", None)
    if replay_buffer is None:
        raise ValueError("Agent does not expose a replay_buffer for offline warmstart seeding.")

    dataset = load_offline_dataset(
        dataset_source,
        controller_sources=controller_sources,
        cases=cases,
        regimes=regimes,
        battery_models=battery_models,
        limit=max_transitions,
    )
    batch = offline_dataset_to_transition_batch(dataset)
    if batch.observations.size == 0:
        return 0

    observations = _align_observations_to_agent_space(agent, batch.observations)
    next_observations = _align_observations_to_agent_space(agent, batch.next_observations)
    scaled_actions = _scale_actions_to_policy_space(agent, batch.actions)
    for index in range(int(observations.shape[0])):
        insert_pos = int(getattr(replay_buffer, "pos", 0))
        info = {
            "offline_dataset": True,
            "dataset_row_index": int(index),
        }
        if "controller_source" in dataset.columns:
            info["controller_source"] = str(dataset.iloc[index]["controller_source"])
        if "case" in dataset.columns:
            info["case"] = str(dataset.iloc[index]["case"])
        if "regime" in dataset.columns:
            info["regime"] = str(dataset.iloc[index]["regime"])
        if "battery_model" in dataset.columns:
            info["battery_model"] = str(dataset.iloc[index]["battery_model"])
        replay_buffer.add(
            observations[index : index + 1],
            next_observations[index : index + 1],
            scaled_actions[index : index + 1],
            batch.rewards[index : index + 1],
            batch.dones[index : index + 1],
            [info],
        )
        _ensure_safe_replay_metadata_arrays(replay_buffer, row_count=1)
        for field in SAFE_REPLAY_META_FIELDS:
            target = getattr(replay_buffer, f"_safe_replay_meta_{field}", None)
            if isinstance(target, np.ndarray) and target.shape[0] > insert_pos:
                target[insert_pos, :] = 0.0
        offline_meta = getattr(replay_buffer, "_safe_replay_meta_offline_dataset", None)
        if isinstance(offline_meta, np.ndarray) and offline_meta.shape[0] > insert_pos:
            offline_meta[insert_pos, :] = 1.0
    return int(observations.shape[0])


def evaluate_actor_behavior_cloning_mse(agent, dataset_source: str | Path | pd.DataFrame) -> float:
    dataset = load_offline_dataset(dataset_source)
    batch = offline_dataset_to_transition_batch(dataset)
    if batch.observations.size == 0:
        return 0.0
    targets = _scale_actions_to_policy_space(agent, batch.actions)
    observations = _align_observations_to_agent_space(agent, batch.observations)
    device = getattr(agent, "device", "cpu")
    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
    target_tensor = th.as_tensor(targets, dtype=th.float32, device=device)
    actor = getattr(agent, "actor", None)
    if actor is None:
        raise ValueError("Agent does not expose an actor for BC warmstart evaluation.")
    actor.eval()
    with th.no_grad():
        predictions = actor(obs_tensor, deterministic=True)
        loss = th.mean((predictions - target_tensor) ** 2)
    return float(loss.detach().cpu().item())


def behavior_clone_sac_actor(
    agent,
    dataset_source: str | Path | pd.DataFrame,
    *,
    gradient_steps: int,
    batch_size: int = 256,
    learning_rate: float | None = None,
    shuffle_seed: int = 0,
) -> dict[str, float | int]:
    dataset = load_offline_dataset(dataset_source)
    batch = offline_dataset_to_transition_batch(dataset)
    targets = _scale_actions_to_policy_space(agent, batch.actions)
    aligned_observations = _align_observations_to_agent_space(agent, batch.observations)
    return _behavior_clone_sac_actor_from_aligned_arrays(
        agent,
        observations=aligned_observations,
        target_actions=targets,
        gradient_steps=int(gradient_steps),
        batch_size=int(batch_size),
        learning_rate=learning_rate,
        shuffle_seed=int(shuffle_seed),
    )


def _safe_replay_meta_attr(field: str) -> str:
    return f"_safe_replay_meta_{field}"


def _extract_replay_buffer_observation_action_arrays(
    agent, *, max_samples: int | None = None, exclude_offline_dataset: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    replay_buffer = getattr(agent, "replay_buffer", None)
    if replay_buffer is None or not hasattr(replay_buffer, "observations") or not hasattr(replay_buffer, "actions"):
        raise ValueError("Agent does not expose a compatible replay_buffer for replay distillation.")
    if not hasattr(replay_buffer, "size"):
        raise ValueError("Replay buffer does not expose a size() method.")
    size = int(replay_buffer.size())
    if size <= 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), {
            field: np.zeros((0,), dtype=np.float32) for field in SAFE_REPLAY_META_FIELDS
        }

    observations = np.asarray(replay_buffer.observations[:size], dtype=np.float32)
    actions = np.asarray(replay_buffer.actions[:size], dtype=np.float32)
    flat_obs = observations.reshape(size * int(observations.shape[1]), -1)
    flat_actions = actions.reshape(size * int(actions.shape[1]), -1)
    flat_meta: dict[str, np.ndarray] = {}
    for field in SAFE_REPLAY_META_FIELDS:
        raw_meta = getattr(replay_buffer, _safe_replay_meta_attr(field), None)
        if raw_meta is None:
            flat_meta[field] = np.zeros((flat_obs.shape[0],), dtype=np.float32)
            continue
        meta_array = np.asarray(raw_meta[:size], dtype=np.float32).reshape(size * int(raw_meta.shape[1]))
        flat_meta[field] = meta_array.astype(np.float32, copy=False)
    if exclude_offline_dataset and flat_obs.shape[0] > 0:
        offline_mask = np.asarray(flat_meta.get("offline_dataset", np.zeros((flat_obs.shape[0],), dtype=np.float32))) <= 0.5
        flat_obs = flat_obs[offline_mask]
        flat_actions = flat_actions[offline_mask]
        for field, values in list(flat_meta.items()):
            flat_meta[field] = values[offline_mask]
    if max_samples is not None and int(max_samples) > 0 and flat_obs.shape[0] > int(max_samples):
        flat_obs = flat_obs[-int(max_samples) :]
        flat_actions = flat_actions[-int(max_samples) :]
        for field, values in list(flat_meta.items()):
            flat_meta[field] = values[-int(max_samples) :]
    return flat_obs.astype(np.float32, copy=False), flat_actions.astype(np.float32, copy=False), flat_meta


def _prepare_safe_distillation_targets(
    *,
    actions: np.ndarray,
    metadata: dict[str, np.ndarray],
    intervention_priority_coef: float,
    boundary_priority_coef: float,
    terminal_priority_coef: float,
    reserve_priority_coef: float,
    teacher_priority_coef: float,
    peak_value_priority_coef: float,
    valley_value_priority_coef: float,
    delta_priority_coef: float,
    terminal_deviation_priority_coef: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    targets = np.asarray(actions, dtype=np.float32).copy()
    shield_applied = np.asarray(metadata.get("shield_applied", np.zeros((actions.shape[0],), dtype=np.float32))) > 0.5
    boundary_active = np.maximum(
        np.asarray(metadata.get("shield_boundary_active", np.zeros((actions.shape[0],), dtype=np.float32))),
        np.asarray(metadata.get("inventory_teacher_boundary_active", np.zeros((actions.shape[0],), dtype=np.float32))),
    )
    terminal_active = np.maximum(
        np.asarray(metadata.get("shield_terminal_active", np.zeros((actions.shape[0],), dtype=np.float32))),
        np.asarray(metadata.get("inventory_teacher_terminal_active", np.zeros((actions.shape[0],), dtype=np.float32))),
    )
    reserve_active = np.asarray(metadata.get("shield_reserve_active", np.zeros((actions.shape[0],), dtype=np.float32)))
    reserve_active = np.maximum(
        reserve_active,
        np.asarray(metadata.get("inventory_teacher_reserve_active", np.zeros((actions.shape[0],), dtype=np.float32))),
    )
    teacher_active = np.asarray(metadata.get("inventory_teacher_active", np.zeros((actions.shape[0],), dtype=np.float32))) > 0.5
    teacher_boundary_active = (
        np.asarray(metadata.get("inventory_teacher_boundary_active", np.zeros((actions.shape[0],), dtype=np.float32))) > 0.5
    )
    teacher_terminal_active = (
        np.asarray(metadata.get("inventory_teacher_terminal_active", np.zeros((actions.shape[0],), dtype=np.float32))) > 0.5
    )
    teacher_reserve_active = (
        np.asarray(metadata.get("inventory_teacher_reserve_active", np.zeros((actions.shape[0],), dtype=np.float32))) > 0.5
    )
    teacher_peak_value_active = (
        np.asarray(metadata.get("inventory_teacher_peak_value_active", np.zeros((actions.shape[0],), dtype=np.float32)))
        > 0.5
    )
    teacher_valley_value_active = (
        np.asarray(metadata.get("inventory_teacher_valley_value_active", np.zeros((actions.shape[0],), dtype=np.float32)))
        > 0.5
    )
    teacher_material_active = (
        teacher_boundary_active
        | teacher_terminal_active
        | teacher_reserve_active
        | teacher_peak_value_active
        | teacher_valley_value_active
    )
    teacher_structural_active = teacher_boundary_active | teacher_terminal_active | teacher_reserve_active
    teacher_peak_value_only_active = teacher_peak_value_active & ~teacher_structural_active
    teacher_valley_value_only_active = teacher_valley_value_active & ~teacher_structural_active
    teacher_action = np.asarray(metadata.get("inventory_teacher_action", np.zeros((actions.shape[0],), dtype=np.float32)))
    teacher_weight = np.asarray(metadata.get("inventory_teacher_weight", np.zeros((actions.shape[0],), dtype=np.float32)))
    shield_delta_abs = np.asarray(metadata.get("shield_delta_abs", np.zeros((actions.shape[0],), dtype=np.float32)))
    terminal_deviation = np.asarray(metadata.get("terminal_soc_deviation", np.zeros((actions.shape[0],), dtype=np.float32)))

    teacher_override_mask = teacher_material_active & ~shield_applied
    if targets.ndim == 2 and targets.shape[1] >= 1 and np.any(teacher_override_mask):
        targets[teacher_override_mask, 0] = np.clip(teacher_action[teacher_override_mask], -1.0, 1.0)

    sample_weights = np.ones((targets.shape[0],), dtype=np.float32)
    sample_weights += float(max(intervention_priority_coef, 0.0)) * shield_applied.astype(np.float32)
    sample_weights += float(max(boundary_priority_coef, 0.0)) * boundary_active.astype(np.float32)
    sample_weights += float(max(terminal_priority_coef, 0.0)) * terminal_active.astype(np.float32)
    sample_weights += float(max(reserve_priority_coef, 0.0)) * reserve_active.astype(np.float32)
    sample_weights += (
        float(max(teacher_priority_coef, 0.0))
        * teacher_structural_active.astype(np.float32)
        * np.clip(teacher_weight, 0.0, 2.0)
    )
    sample_weights += (
        float(max(peak_value_priority_coef, 0.0))
        * teacher_peak_value_only_active.astype(np.float32)
        * np.clip(teacher_weight, 0.0, 2.0)
    )
    sample_weights += (
        float(max(valley_value_priority_coef, 0.0))
        * teacher_valley_value_only_active.astype(np.float32)
        * np.clip(teacher_weight, 0.0, 2.0)
    )
    sample_weights += float(max(delta_priority_coef, 0.0)) * np.clip(shield_delta_abs, 0.0, 1.0)
    sample_weights += float(max(terminal_deviation_priority_coef, 0.0)) * np.clip(terminal_deviation, 0.0, 1.0)
    sampling_probabilities = sample_weights / max(float(np.sum(sample_weights)), 1e-9)
    diagnostics = {
        "intervention_rows": int(np.sum(shield_applied.astype(np.int32))),
        "inventory_teacher_rows": int(np.sum(teacher_material_active.astype(np.int32))),
        "mean_sample_weight": float(np.mean(sample_weights)) if sample_weights.size else 1.0,
    }
    return targets, sample_weights.astype(np.float32, copy=False), sampling_probabilities.astype(np.float32, copy=False), diagnostics


def distill_sac_actor_from_replay_buffer(
    agent,
    *,
    gradient_steps: int,
    batch_size: int = 256,
    learning_rate: float | None = None,
    max_samples: int | None = None,
    shuffle_seed: int = 0,
    intervention_priority_coef: float = 4.0,
    boundary_priority_coef: float = 2.0,
    terminal_priority_coef: float = 2.0,
    reserve_priority_coef: float = 1.0,
    teacher_priority_coef: float = 2.0,
    peak_value_priority_coef: float = 0.75,
    valley_value_priority_coef: float = 0.5,
    delta_priority_coef: float = 2.0,
    terminal_deviation_priority_coef: float = 1.0,
) -> ReplayDistillationReport:
    observations, actions, metadata = _extract_replay_buffer_observation_action_arrays(
        agent,
        max_samples=max_samples,
        exclude_offline_dataset=True,
    )
    if observations.size == 0 or actions.size == 0:
        return ReplayDistillationReport(
            replay_rows=0,
            actor_gradient_steps=0,
            actor_batch_size=int(max(batch_size, 1)),
            initial_actor_mse=0.0,
            final_actor_mse=0.0,
        )
    target_actions, sample_weights, sampling_probabilities, diagnostics = _prepare_safe_distillation_targets(
        actions=actions,
        metadata=metadata,
        intervention_priority_coef=float(intervention_priority_coef),
        boundary_priority_coef=float(boundary_priority_coef),
        terminal_priority_coef=float(terminal_priority_coef),
        reserve_priority_coef=float(reserve_priority_coef),
        teacher_priority_coef=float(teacher_priority_coef),
        peak_value_priority_coef=float(peak_value_priority_coef),
        valley_value_priority_coef=float(valley_value_priority_coef),
        delta_priority_coef=float(delta_priority_coef),
        terminal_deviation_priority_coef=float(terminal_deviation_priority_coef),
    )
    report = _behavior_clone_sac_actor_from_aligned_arrays(
        agent,
        observations=observations,
        target_actions=target_actions,
        gradient_steps=int(gradient_steps),
        batch_size=int(batch_size),
        learning_rate=learning_rate,
        shuffle_seed=int(shuffle_seed),
        sample_weights=sample_weights,
        sampling_probabilities=sampling_probabilities,
    )
    return ReplayDistillationReport(
        replay_rows=int(observations.shape[0]),
        actor_gradient_steps=int(report["actor_gradient_steps"]),
        actor_batch_size=int(report["actor_batch_size"]),
        initial_actor_mse=float(report["initial_actor_mse"]),
        final_actor_mse=float(report["final_actor_mse"]),
        intervention_rows=int(diagnostics["intervention_rows"]),
        inventory_teacher_rows=int(diagnostics["inventory_teacher_rows"]),
        mean_sample_weight=float(diagnostics["mean_sample_weight"]),
    )


def apply_bc_warmstart(
    agent,
    dataset_source: str | Path | pd.DataFrame,
    *,
    replay_seed_limit: int | None = None,
    actor_prefit_gradient_steps: int = 0,
    actor_prefit_batch_size: int = 256,
    actor_prefit_learning_rate: float | None = None,
    controller_sources: Iterable[str] | None = None,
    cases: Iterable[str] | None = None,
    regimes: Iterable[str] | None = None,
    battery_models: Iterable[str] | None = None,
    shuffle_seed: int = 0,
) -> BCWarmstartReport:
    dataset = load_offline_dataset(
        dataset_source,
        controller_sources=controller_sources,
        cases=cases,
        regimes=regimes,
        battery_models=battery_models,
    )
    replay_seeded = seed_replay_buffer_from_offline_dataset(
        agent,
        dataset,
        max_transitions=replay_seed_limit,
    )
    actor_report = behavior_clone_sac_actor(
        agent,
        dataset,
        gradient_steps=int(actor_prefit_gradient_steps),
        batch_size=int(actor_prefit_batch_size),
        learning_rate=actor_prefit_learning_rate,
        shuffle_seed=int(shuffle_seed),
    )
    return BCWarmstartReport(
        dataset_rows=int(len(dataset)),
        replay_seeded_transitions=int(replay_seeded),
        actor_gradient_steps=int(actor_report["actor_gradient_steps"]),
        actor_batch_size=int(actor_report["actor_batch_size"]),
        initial_actor_mse=float(actor_report["initial_actor_mse"]),
        final_actor_mse=float(actor_report["final_actor_mse"]),
    )
