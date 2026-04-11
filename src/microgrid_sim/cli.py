"""Package-native CLI for the network-first microgrid platform."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from .cases import CIGREEuropeanLVConfig, IEEE33Config
from .envs.network_microgrid import NetworkMicrogridEnv


def _build_config(case: str, days: int, seed: int, battery_model: str, reward_profile: str):
    if case == "ieee33":
        return IEEE33Config(simulation_days=days, seed=seed, battery_model=battery_model, reward_profile=reward_profile)
    return CIGREEuropeanLVConfig(simulation_days=days, seed=seed, battery_model=battery_model, reward_profile=reward_profile)


def _build_env(case: str, battery_model: str, days: int, seed: int, reward_profile: str):
    config = _build_config(case=case, days=days, seed=seed, battery_model=battery_model, reward_profile=reward_profile)
    return NetworkMicrogridEnv(config)


def cmd_smoke(args) -> int:
    env = _build_env(case=args.case, battery_model=args.model, days=args.days, seed=args.seed, reward_profile=args.reward_profile)
    if args.monitor:
        monitor_path = Path(args.monitor)
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_path))

    obs, info = env.reset()
    print("reset_ok", obs.shape, sorted(info.keys())[:8])
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(
            "step",
            step,
            "reward",
            float(reward),
            "vmin",
            float(info["min_bus_voltage_pu"]),
            "line_max",
            float(info["max_line_loading_pct"]),
        )
        if terminated or truncated:
            break
    env.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network-first microgrid CLI")
    subparsers = parser.add_subparsers(dest="command")

    smoke = subparsers.add_parser("smoke", help="Run a short environment smoke test")
    smoke.add_argument("--case", choices=["cigre", "ieee33"], default="cigre")
    smoke.add_argument("--model", choices=["none", "thevenin", "thevenin_loss_only", "simple"], default="thevenin")
    smoke.add_argument("--reward-profile", choices=["network", "paper_aligned", "paper_balanced"], default="network")
    smoke.add_argument("--days", type=int, default=1)
    smoke.add_argument("--steps", type=int, default=4)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--monitor", type=str, default="")
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
