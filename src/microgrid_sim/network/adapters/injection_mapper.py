"""Write load, PV, and battery injections into a pandapower network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InjectionState:
    load_indices: tuple[int, ...]
    load_weights: np.ndarray
    load_q_to_p_ratios: np.ndarray
    pv_indices: tuple[int, ...]
    pv_weights: np.ndarray
    storage_index: int


def _resolve_distribution_weights(explicit_weights, fallback_weights: np.ndarray, *, label: str) -> np.ndarray:
    if explicit_weights is None:
        weights = np.asarray(fallback_weights, dtype=float)
    else:
        weights = np.asarray(explicit_weights, dtype=float)
        if weights.shape != fallback_weights.shape:
            raise ValueError(f"{label} weights must have shape {fallback_weights.shape}, got {weights.shape}")
    if weights.size == 0:
        raise ValueError(f"{label} weights must not be empty")
    if np.any(weights < 0.0):
        raise ValueError(f"{label} weights must be non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.ones_like(fallback_weights, dtype=float)
        total = float(np.sum(weights))
    return np.asarray(weights / total, dtype=float)


def initialize_injection_state(net) -> InjectionState:
    metadata = getattr(net, "user_metadata", {}) or {}
    load_indices = tuple(int(idx) for idx in net.load.index.tolist())
    base_load_weights = np.asarray(net.load["p_mw"].to_numpy(dtype=float), dtype=float)
    load_q_mvar = np.asarray(net.load["q_mvar"].to_numpy(dtype=float), dtype=float) if "q_mvar" in net.load else np.zeros_like(base_load_weights)
    if base_load_weights.size == 0:
        raise ValueError("Network must include at least one load element")
    q_to_p = np.divide(
        load_q_mvar,
        np.maximum(base_load_weights, 1e-9),
        out=np.zeros_like(load_q_mvar, dtype=float),
        where=np.abs(base_load_weights) > 1e-9,
    )
    load_weights = _resolve_distribution_weights(
        metadata.get("load_distribution_weights"),
        base_load_weights,
        label="load distribution",
    )

    pv_rows = net.sgen.index[net.sgen["name"].astype(str).str.startswith("pv_injection_")].tolist()
    pv_indices = tuple(int(idx) for idx in pv_rows)
    if not pv_indices:
        raise ValueError("Network must include controllable PV injection slots named 'pv_injection_*'")
    pv_weights = _resolve_distribution_weights(
        metadata.get("pv_distribution_weights"),
        np.ones(len(pv_indices), dtype=float),
        label="PV distribution",
    )

    if "storage_index" not in metadata:
        raise ValueError("Network metadata must expose 'storage_index'")

    return InjectionState(
        load_indices=load_indices,
        load_weights=load_weights,
        load_q_to_p_ratios=q_to_p,
        pv_indices=pv_indices,
        pv_weights=pv_weights,
        storage_index=int(metadata["storage_index"]),
    )


def apply_power_injections(net, state: InjectionState, load_w: float, pv_w: float, battery_power_w: float) -> None:
    load_mw = max(float(load_w), 0.0) / 1_000_000.0
    pv_mw = max(float(pv_w), 0.0) / 1_000_000.0
    battery_mw = -float(battery_power_w) / 1_000_000.0

    for idx, weight, q_to_p in zip(state.load_indices, state.load_weights, state.load_q_to_p_ratios):
        load_p = load_mw * float(weight)
        net.load.at[idx, "p_mw"] = load_p
        if "q_mvar" in net.load:
            net.load.at[idx, "q_mvar"] = load_p * float(q_to_p)

    for idx, weight in zip(state.pv_indices, state.pv_weights):
        net.sgen.at[idx, "p_mw"] = pv_mw * float(weight)

    net.storage.at[state.storage_index, "p_mw"] = battery_mw
