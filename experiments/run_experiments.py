#!/usr/bin/env python3
"""
Enhanced CLI for Microgrid DRL Experiments.

Supports:
- Training SAC/PPO/TD3 on Thevenin/Simple battery models
- Evaluation of trained models
- Individual or combined experiments
- Loading pre-trained models

Usage:
    # Train SAC on Thevenin model
    python run_experiments.py train --agent sac --model thevenin --steps 300000

    # Train TD3 on Thevenin model
    python run_experiments.py train --agent td3 --model thevenin --steps 300000

    # Evaluate existing models
    python run_experiments.py eval --load-model ./models/sac_thevenin.zip --model thevenin

    # Run full comparison (loads existing models if available)
    python run_experiments.py compare --days 365 --output ./results_full

    # Quick test
    python run_experiments.py compare --fast
"""

import os
import sys
import shutil
import importlib
import inspect
from dataclasses import replace
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from typing import Optional, List

from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from microgrid_sim import (
    CIGREConfig,
    CIGREMicrogridEnv,
    MicrogridEnvThevenin, 
    MicrogridEnvSimple, 
    MicrogridConfig,
    run_milp_baseline,
    run_rule_based_baseline,
)
from microgrid_sim.paths import DEFAULT_DATASET_FILES, resolve_data_dir

TRAIN_LOG_DIR = Path(__file__).parent.parent / 'results' / 'training_logs'
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_INFO_KEYS = ('cumulative_cost', 'soc', 'soh', 'price')
PROJECT_ROOT = Path(__file__).parent.parent

SCRIPT_CLI_COMMANDS = {
    'cigre-gap-run': {
        'module': '_cmd_run_cigre_gap_experiment',
        'entry': 'main',
        'help': 'Run focused MG-CIGRE PBM-vs-EBM experiment',
    },
    'residential-d4-run': {
        'module': '_cmd_run_residential_d4_experiment',
        'entry': 'main',
        'help': 'Run residential PBM-vs-EBM experiment (D4 line)',
    },
    'residential-d9-grid': {
        'module': '_cmd_run_residential_d9_grid_search',
        'entry': 'main',
        'help': 'Run residential D9 parameter grid search',
    },
}


def resolve_real_data_dir() -> Optional[str]:
    """Resolve the best available real-data directory."""
    data_dir = resolve_data_dir(DEFAULT_DATASET_FILES)
    return str(data_dir) if data_dir else None


def resolve_cigre_data_dir() -> Optional[str]:
    """Resolve the best available MG-CIGRE data directory."""
    return resolve_real_data_dir()


def _scale_battery_params(base_params, power_scale: float, energy_scale: float):
    scaled_power = max(float(power_scale), 1e-6)
    scaled_energy = max(float(energy_scale), 1e-6)
    nominal_energy_kwh = base_params.nominal_energy_wh / 1000.0 * scaled_energy
    return replace(
        base_params,
        nominal_energy_kwh=nominal_energy_kwh,
        p_charge_max=base_params.p_charge_max * scaled_power,
        p_discharge_max=base_params.p_discharge_max * scaled_power,
    )


def parse_float_list(text: str) -> List[float]:
    vals = []
    for tok in str(text).split(','):
        t = tok.strip().strip("'").strip('"')
        if not t:
            continue
        vals.append(float(t))
    return vals


def parse_int_list(text: str) -> List[int]:
    vals = []
    for tok in str(text).split(','):
        t = tok.strip().strip("'").strip('"')
        if not t:
            continue
        vals.append(int(t))
    return vals


def eta_to_tag(eta: float) -> str:
    x = int(round(float(eta) * 100))
    if x == 90:
        return '09'
    if x == 95:
        return '095'
    if x == 100:
        return '100'
    return f"{x:03d}"


def get_device():
    """Detect best compute device."""
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  Using CPU")
    return device


def create_env(
    env_type: str,
    simulation_days: int = 30,
    seed: int = 42,
    monitor_log_path: Optional[Path] = None,
    simple_eta: Optional[float] = None,
):
    """Create environment by type."""
    def _make_env():
        config = MicrogridConfig(
            simulation_days=simulation_days,
            seed=seed,
            data_dir=resolve_real_data_dir(),
        )
        if env_type == 'thevenin':
            env = MicrogridEnvThevenin(config)
        else:
            env = MicrogridEnvSimple(config)
            if simple_eta is not None:
                env.battery.params.eta_charge = float(simple_eta)
                env.battery.params.eta_discharge = float(simple_eta)
        log_file = None
        if monitor_log_path is not None:
            monitor_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = str(monitor_log_path)
        return Monitor(env, filename=log_file, info_keywords=MONITOR_INFO_KEYS)
    return DummyVecEnv([_make_env])


def train_agent(
    agent_type: str,
    env_type: str,
    total_timesteps: int,
    simulation_days: int = 30,
    seed: int = 42,
    save_path: Optional[str] = None,
    use_gpu: bool = True,
    log_name: Optional[str] = None,
    simple_eta: Optional[float] = None,
):
    """
    Train DRL agent.
    
    Args:
        agent_type: 'sac' or 'ppo'
        env_type: 'thevenin' or 'simple'
        total_timesteps: Training steps
        simulation_days: Days per episode
        seed: Random seed
        save_path: Path to save model
        use_gpu: Use GPU acceleration
        log_name: Optional prefix for monitor/progress logs
    """
    print(f"\n{'='*60}")
    print(f"Training {agent_type.upper()} on {env_type.upper()} model")
    print(f"{'='*60}")
    
    device = get_device() if use_gpu else "cpu"
    log_prefix = log_name or f"{agent_type}_{env_type}"
    monitor_file = TRAIN_LOG_DIR / f"{log_prefix}_monitor.csv"
    env = create_env(
        env_type,
        simulation_days=simulation_days,
        seed=seed,
        monitor_log_path=monitor_file,
        simple_eta=simple_eta,
    )

    obs_dim = int(np.prod(getattr(env.observation_space, "shape", (0,)) or (0,)))
    action_dim = int(np.prod(getattr(env.action_space, "shape", (0,)) or (0,)))
    replay_buffer_size = int(min(1_000_000, max(50_000, int(total_timesteps))))
    print(f"  Replay buffer: {replay_buffer_size:,} (obs_dim={obs_dim}, action_dim={action_dim})")
    
    log_dir = TRAIN_LOG_DIR / log_prefix
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Agent configuration
    policy_kwargs = dict(net_arch=dict(pi=[256, 128, 64], qf=[256, 128, 64]))
    
    if agent_type == 'sac':
        agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=replay_buffer_size,
            learning_starts=1000,
            batch_size=384,
            tau=0.003,
            gamma=0.985,
            ent_coef='auto',
            target_entropy=-float(max(action_dim, 1)),
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
        )
    elif agent_type == 'ppo':
        # PPO configuration
        policy_kwargs_ppo = dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.985,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs_ppo,
            tensorboard_log=None,
        )
    elif agent_type == 'td3':
        # TD3 configuration
        agent = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=replay_buffer_size,
            learning_starts=1000,
            batch_size=384,
            tau=0.003,
            gamma=0.985,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
        )
    elif agent_type == 'ddpg':
        # DDPG configuration
        agent = DDPG(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=replay_buffer_size,
            learning_starts=1000,
            batch_size=384,
            tau=0.003,
            gamma=0.985,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.set_logger(configure(str(log_dir), ["csv"]))
    print(f"Starting training for {total_timesteps:,} timesteps...")
    agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    if save_path:
        try:
            agent.save(save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save model ({e})")
    
    progress_csv = log_dir / 'progress.csv'
    residual_csv = TRAIN_LOG_DIR / f"{log_prefix}_residual.csv"
    if progress_csv.exists():
        shutil.copy(progress_csv, residual_csv)
    
    return agent


def evaluate_agent(
    agent,
    env_type: str = 'thevenin',
    simulation_days: int = 365,
    seed: int = 42,
    agent_name: str = "Agent",
    simple_eta: Optional[float] = None,
) -> dict:
    """Evaluate trained agent."""
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} on {env_type.upper()} environment")
    print(f"Simulation: {simulation_days} days ({simulation_days * 24} hours)")
    print(f"{'='*60}")
    
    config = MicrogridConfig(
        simulation_days=simulation_days,
        seed=seed,
        data_dir=resolve_real_data_dir(),
    )
    
    if env_type == 'thevenin':
        env = MicrogridEnvThevenin(config)
    else:
        env = MicrogridEnvSimple(config)
        if simple_eta is not None:
            env.battery.params.eta_charge = float(simple_eta)
            env.battery.params.eta_discharge = float(simple_eta)
    
    obs, info = env.reset()
    total_steps = simulation_days * 24
    
    # Tracking
    steps, soc_hist, soh_hist, cost_hist = [], [], [], []
    pv_hist, load_hist, grid_hist, batt_hist, price_hist = [], [], [], [], []
    current_hist, voltage_hist, eff_hist, loss_hist = [], [], [], []
    r_int_hist, v_ocv_hist = [], []
    
    for step in tqdm(range(total_steps), desc=f"Evaluating {agent_name}"):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        steps.append(step)
        soc_hist.append(info['soc'])
        soh_hist.append(info['soh'])
        cost_hist.append(info['cumulative_cost'])
        pv_hist.append(info.get('pv_power', 0))
        load_hist.append(info.get('load_power', 0))
        # Note: step() returns 'p_grid' and 'p_actual' in info
        grid_hist.append(info.get('p_grid', info.get('grid_power', 0)))
        batt_hist.append(info.get('p_actual', info.get('battery_power', 0)))
        price_hist.append(info.get('price', 0))

        battery_info = info.get('battery_info', {})
        current_hist.append(battery_info.get('current', 0.0))
        voltage_hist.append(battery_info.get('voltage', 0.0))
        eff_hist.append(battery_info.get('efficiency', 0.0))
        loss_hist.append(battery_info.get('power_loss', 0.0))
        r_int_hist.append(battery_info.get('r_int', 0.0))
        v_ocv_hist.append(battery_info.get('v_ocv', 0.0))
        
        if terminated or truncated:
            break
    
    total_cost = cost_hist[-1] if cost_hist else 0
    final_soh = soh_hist[-1] if soh_hist else 1.0
    
    print(f"\n{agent_name} Results:")
    print(f"  Total Cost: ${total_cost:,.2f}")
    print(f"  Final SOH: {final_soh:.4f} ({(1-final_soh)*100:.2f}% degradation)")
    print(f"  SOC Range: {min(soc_hist)*100:.2f}% - {max(soc_hist)*100:.2f}%")
    
    return {
        'name': agent_name,
        'total_cost': total_cost,
        'final_soh': final_soh,
        'soh_degradation': (1.0 - final_soh) * 100,
        'steps': steps,
        'soc': soc_hist,
        'soh': soh_hist,
        'cost': cost_hist,
        'pv': pv_hist,
        'load': load_hist,
        'grid': grid_hist,
        'battery_power': batt_hist,
        'price': price_hist,
        'current': current_hist,
        'voltage': voltage_hist,
        'efficiency': eff_hist,
        'power_loss': loss_hist,
        'r_int': r_int_hist,
        'v_ocv': v_ocv_hist,
    }


def evaluate_cigre_agent(
    agent,
    simulation_days: int = 365,
    seed: int = 42,
    agent_name: str = "Agent",
    data_dir: Optional[str] = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} on MG-CIGRE (PBM/Thevenin) environment")
    print(f"Simulation: {simulation_days} days ({simulation_days * 24} hours)")
    print(f"{'='*60}")

    config = CIGREConfig(
        simulation_days=simulation_days,
        seed=seed,
    )
    config.data_dir = str(data_dir) if data_dir else resolve_cigre_data_dir()
    env = CIGREMicrogridEnv(config, battery_model='thevenin')

    obs, info = env.reset()
    total_steps = simulation_days * 24

    steps, soc_hist, soh_hist, cost_hist = [], [], [], []

    for step in tqdm(range(total_steps), desc=f"Evaluating {agent_name}"):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(step)
        soc_hist.append(info.get('soc', 0.0))
        soh_hist.append(info.get('soh', 1.0))
        cost_hist.append(info.get('cumulative_cost', 0.0))
        if terminated or truncated:
            break

    total_cost = cost_hist[-1] if cost_hist else 0.0
    final_soh = soh_hist[-1] if soh_hist else 1.0

    print(f"\n{agent_name} Results:")
    print(f"  Total Cost: ${total_cost:,.2f}")
    print(f"  Final SOH: {final_soh:.4f} ({(1-final_soh)*100:.2f}% degradation)")
    if soc_hist:
        print(f"  SOC Range: {min(soc_hist)*100:.2f}% - {max(soc_hist)*100:.2f}%")

    return {
        'name': agent_name,
        'total_cost': float(total_cost),
        'final_soh': float(final_soh),
        'soh_degradation': float((1.0 - final_soh) * 100),
        'steps': steps,
        'soc': soc_hist,
        'soh': soh_hist,
        'cost': cost_hist,
    }


def run_no_control_baseline(simulation_days: int = 365, seed: int = 42) -> dict:
    """Run no-control baseline."""
    print(f"\n{'='*60}")
    print("Running No-Control Baseline")
    print(f"{'='*60}")
    
    config = MicrogridConfig(simulation_days=simulation_days, seed=seed)
    env = MicrogridEnvThevenin(config)
    
    obs, info = env.reset()
    total_steps = simulation_days * 24
    steps, soc_hist, soh_hist, cost_hist = [], [], [], []
    pv_hist, load_hist, grid_hist, batt_hist, price_hist = [], [], [], [], []
    current_hist, voltage_hist, eff_hist, loss_hist = [], [], [], []
    r_int_hist, v_ocv_hist = [], []
    
    for step in tqdm(range(total_steps), desc="No Control"):
        action = np.array([0.0])  # No battery action
        obs, reward, terminated, truncated, info = env.step(action)

        steps.append(step)
        soc_hist.append(info['soc'])
        soh_hist.append(info['soh'])
        cost_hist.append(info['cumulative_cost'])
        pv_hist.append(info.get('pv_power', 0.0))
        load_hist.append(info.get('load_power', 0.0))
        grid_hist.append(info.get('p_grid', info.get('grid_power', 0.0)))
        batt_hist.append(info.get('p_actual', info.get('battery_power', 0.0)))
        price_hist.append(info.get('price', 0.0))

        battery_info = info.get('battery_info', {})
        current_hist.append(battery_info.get('current', 0.0))
        voltage_hist.append(battery_info.get('voltage', 0.0))
        eff_hist.append(battery_info.get('efficiency', 0.0))
        loss_hist.append(battery_info.get('power_loss', 0.0))
        r_int_hist.append(battery_info.get('r_int', 0.0))
        v_ocv_hist.append(battery_info.get('v_ocv', 0.0))

        if terminated or truncated:
            break
    
    total_cost = cost_hist[-1] if cost_hist else 0
    final_soh = soh_hist[-1] if soh_hist else 1.0
    print(f"No-Control Total Cost: ${total_cost:,.2f}")
    
    return {
        'name': 'No Control',
        'total_cost': total_cost,
        'final_soh': final_soh,
        'soh_degradation': (1.0 - final_soh) * 100,
        'steps': steps,
        'soc': soc_hist,
        'soh': soh_hist,
        'cost': cost_hist,
        'pv': pv_hist,
        'load': load_hist,
        'grid': grid_hist,
        'battery_power': batt_hist,
        'price': price_hist,
        'current': current_hist,
        'voltage': voltage_hist,
        'efficiency': eff_hist,
        'power_loss': loss_hist,
        'r_int': r_int_hist,
        'v_ocv': v_ocv_hist,
    }


def save_results(results: dict, output_dir: str, prefix: str = ""):
    """Save detailed results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Time series data
    if 'steps' in results:
        n_steps = len(results['steps'])
        
        # Build dataframe with available data, padding missing arrays
        data = {
            'Hour': results['steps'],
            'SOC': results['soc'],
            'SOH': results['soh'],
            'Cumulative_Cost': results['cost'],
        }
        
        # Add optional fields if they exist and have correct length
        for key, col_name in [
            ('pv', 'PV_Power'),
            ('load', 'Load_Power'),
            ('grid', 'Grid_Power'),
            ('battery_power', 'Battery_Power'),
            ('price', 'Price'),
            ('current', 'Battery_Current'),
            ('voltage', 'Battery_Voltage'),
            ('efficiency', 'Battery_Efficiency'),
            ('power_loss', 'Battery_Power_Loss'),
            ('r_int', 'Battery_Rint'),
            ('v_ocv', 'Battery_OCV'),
        ]:
            if key in results and len(results[key]) == n_steps:
                data[col_name] = results[key]
        
        df = pd.DataFrame(data)
        safe_name = str(results.get('name', 'results')).replace(' ', '_').lower()
        safe_name = safe_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = f"{prefix}_{safe_name}_timeseries.csv"
        df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Saved: {filename}")


def export_timeseries_exact(results: dict, out_path: Path):
    """Export exact filename with standard columns for downstream eta tools."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        'Hour': results.get('steps', []),
        'SOC': results.get('soc', []),
        'Cumulative_Cost': results.get('cost', []),
    })
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def cmd_milp_both(args):
    """Run LP/MPC-style baseline for MG-RES and MG-CIGRE and export a summary."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # --- MG-RES ---
    config_res = MicrogridConfig(
        simulation_days=args.res_days,
        seed=args.seed,
        data_dir=resolve_real_data_dir(),
    )
    if bool(getattr(args, 'paper_protocol', False)):
        config_res.data_year = int(args.res_year)
        config_res.tou_price_spread_multiplier = 1.75
        config_res.monthly_demand_charge_per_kw = 16.0
        config_res.monthly_demand_charge_threshold_w = 2_000.0
    env_res = MicrogridEnvThevenin(config_res)
    res = run_milp_baseline(
        env_res,
        simulation_days=args.res_days,
        name=f"LP Oracle (MG-RES)",
        chunk_days=args.chunk_days,
        efficiency_model=args.efficiency_model,
    )
    rows.append({
        'Microgrid': 'MG-RES',
        'Days': int(args.res_days),
        'Hours': int(args.res_days) * 24,
        'Method': res.get('name', 'LP Oracle'),
        'Total_Cost': float(res.get('total_cost', float('nan'))),
        'Final_SOH': float(res.get('final_soh', float('nan'))),
        'SOH_Degradation_%': float(res.get('soh_degradation', float('nan'))),
        'Efficiency_Model': str(args.efficiency_model),
        'Chunk_Days': int(args.chunk_days),
        'Seed': int(args.seed),
        'Data_Year': int(config_res.data_year) if config_res.data_year is not None else '',
        'Paper_Protocol': bool(getattr(args, 'paper_protocol', False)),
    })
    env_res.close()

    # --- MG-CIGRE ---
    config_cigre = CIGREConfig(
        simulation_days=args.cigre_days,
        seed=args.seed,
    )
    config_cigre.data_dir = args.cigre_data_dir if getattr(args, 'cigre_data_dir', None) else resolve_cigre_data_dir()
    if bool(getattr(args, 'paper_protocol', False)):
        config_cigre.data_year = int(args.cigre_year)
        config_cigre.battery_params = _scale_battery_params(
            config_cigre.battery_params,
            power_scale=1.3,
            energy_scale=1.1,
        )
    env_cigre = CIGREMicrogridEnv(config_cigre, battery_model='thevenin')
    cigre = run_milp_baseline(
        env_cigre,
        simulation_days=args.cigre_days,
        name=f"LP Oracle (MG-CIGRE)",
        chunk_days=args.chunk_days,
        efficiency_model=args.efficiency_model,
    )
    rows.append({
        'Microgrid': 'MG-CIGRE',
        'Days': int(args.cigre_days),
        'Hours': int(args.cigre_days) * 24,
        'Method': cigre.get('name', 'LP Oracle'),
        'Total_Cost': float(cigre.get('total_cost', float('nan'))),
        'Final_SOH': float(cigre.get('final_soh', float('nan'))),
        'SOH_Degradation_%': float(cigre.get('soh_degradation', float('nan'))),
        'Efficiency_Model': str(args.efficiency_model),
        'Chunk_Days': int(args.chunk_days),
        'Seed': int(args.seed),
        'Data_Year': int(config_cigre.data_year) if config_cigre.data_year is not None else '',
        'Paper_Protocol': bool(getattr(args, 'paper_protocol', False)),
    })
    env_cigre.close()

    summary_df = pd.DataFrame(rows)
    out_csv = output_dir / 'milp_both_summary.csv'
    summary_df.to_csv(out_csv, index=False)

    print("\n=== LP/MPC Baseline (Both Microgrids) Summary ===")
    print(summary_df.to_string(index=False))
    print(f"Saved: {out_csv}")

    return {
        'summary': summary_df,
        'out_csv': str(out_csv),
    }


def cmd_train(args):
    """Train command handler."""
    save_path = args.output or f"./models/{args.agent}_{args.model}"
    
    agent = train_agent(
        agent_type=args.agent,
        env_type=args.model,
        total_timesteps=args.steps,
        simulation_days=args.days,
        seed=args.seed,
        save_path=save_path,
        use_gpu=not args.cpu,
        simple_eta=args.eta,
    )
    
    print(f"\nTraining complete!")
    return agent


def cmd_eval(args):
    """Evaluate command handler."""
    # Load model
    if args.agent == 'sac':
        agent = SAC.load(args.load_model)
    elif args.agent == 'ppo':
        agent = PPO.load(args.load_model)
    elif args.agent == 'td3':
        agent = TD3.load(args.load_model)
    elif args.agent == 'ddpg':
        agent = DDPG.load(args.load_model)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    results = evaluate_agent(
        agent,
        env_type=args.model,
        simulation_days=args.days,
        seed=args.seed,
        agent_name=f"{args.agent.upper()} ({args.model})",
        simple_eta=args.eta,
    )
    
    if args.output:
        save_results(results, args.output, prefix=args.agent)
    
    return results


def cmd_cigre_eval(args):
    if args.agent != 'sac':
        raise ValueError("Currently only --agent sac is supported for cigre-eval")

    agent = SAC.load(args.load_model)
    data_dir = args.data_dir if getattr(args, 'data_dir', None) else resolve_cigre_data_dir()
    results = evaluate_cigre_agent(
        agent,
        simulation_days=args.days,
        seed=args.seed,
        agent_name=f"{args.agent.upper()} (cigre/pbm-eval)",
        data_dir=data_dir,
    )
    if args.output:
        save_results(results, args.output, prefix=args.agent)
    return results


def cmd_eta_compare(args):
    """Train + evaluate PBM/EBM under different EBM efficiencies and compare."""
    print("=" * 70)
    print(" Eta Compare (PBM vs EBM@eta) ")
    print(f" Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    etas = parse_float_list(args.etas)
    seeds = parse_int_list(args.seeds)
    output_dir = Path(args.output)
    model_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) PBM baseline per seed
    pbm_cost_by_seed = {}
    for seed in seeds:
        pbm_model_path = model_dir / f"sac_thevenin_seed{seed}.zip"
        if pbm_model_path.exists() and (not args.retrain_pbm):
            print(f"Loading PBM model: {pbm_model_path}")
            pbm_agent = SAC.load(str(pbm_model_path))
        else:
            pbm_agent = train_agent(
                agent_type='sac',
                env_type='thevenin',
                total_timesteps=args.steps,
                simulation_days=args.train_days,
                seed=seed,
                save_path=str(pbm_model_path),
                use_gpu=not args.cpu,
                log_name=f"sac_thevenin_seed{seed}",
            )

        pbm_res = evaluate_agent(
            pbm_agent,
            env_type='thevenin',
            simulation_days=args.eval_days,
            seed=seed,
            agent_name=f"PBM_seed{seed}",
        )
        pbm_cost_by_seed[seed] = pbm_res['total_cost']
        export_timeseries_exact(pbm_res, output_dir / f"pbm_seed{seed}_timeseries.csv")

    # 2) EBM@eta per seed (evaluate on PBM env)
    rows = []
    for eta in etas:
        eta_tag = eta_to_tag(eta)
        for seed in seeds:
            ebm_model_path = model_dir / f"sac_simple_eta{eta_tag}_seed{seed}.zip"
            if ebm_model_path.exists() and (not args.retrain):
                print(f"Loading EBM model: {ebm_model_path}")
                ebm_agent = SAC.load(str(ebm_model_path))
            else:
                ebm_agent = train_agent(
                    agent_type='sac',
                    env_type='simple',
                    total_timesteps=args.steps,
                    simulation_days=args.train_days,
                    seed=seed,
                    save_path=str(ebm_model_path),
                    use_gpu=not args.cpu,
                    log_name=f"sac_simple_eta{eta_tag}_seed{seed}",
                    simple_eta=eta,
                )

            ebm_res = evaluate_agent(
                ebm_agent,
                env_type='thevenin',
                simulation_days=args.eval_days,
                seed=seed,
                agent_name=f"EBM{eta_tag}_seed{seed}",
            )
            export_timeseries_exact(ebm_res, output_dir / f"ebm{eta_tag}_seed{seed}_timeseries.csv")

            pbm_cost = float(pbm_cost_by_seed[seed])
            ebm_cost = float(ebm_res['total_cost'])
            gap_pct = (ebm_cost - pbm_cost) / pbm_cost * 100.0 if abs(pbm_cost) > 1e-9 else np.nan
            rows.append({
                'eta': float(eta),
                'eta_tag': eta_tag,
                'seed': int(seed),
                'pbm_cost': pbm_cost,
                'ebm_cost': ebm_cost,
                'gap_pct': gap_pct,
            })

    raw_df = pd.DataFrame(rows)
    summary_df = raw_df.groupby('eta_tag', as_index=False).agg(
        N=('seed', 'count'),
        MeanPBM=('pbm_cost', 'mean'),
        MeanEBM=('ebm_cost', 'mean'),
        MeanGapPct=('gap_pct', 'mean'),
    ).sort_values('eta_tag')

    spread_pct = np.nan
    if len(summary_df) >= 2:
        min_ebm = float(summary_df['MeanEBM'].min())
        max_ebm = float(summary_df['MeanEBM'].max())
        if abs(min_ebm) > 1e-9:
            spread_pct = (max_ebm - min_ebm) / min_ebm * 100.0

    result = 'SUCCESS' if (np.isfinite(spread_pct) and abs(spread_pct) > args.success_threshold_pct) else 'NOT_ENOUGH_OR_LT_THRESHOLD'

    raw_path = output_dir / 'eta_compare_raw.csv'
    summary_path = output_dir / 'eta_compare_summary.csv'
    verdict_path = output_dir / 'eta_compare_verdict.txt'
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with verdict_path.open('w', encoding='utf-8') as f:
        f.write(f"CrossEtaSpreadPct={spread_pct}\n")
        f.write(f"ThresholdPct={args.success_threshold_pct}\n")
        f.write(f"Result={result}\n")

    print("\n=== ETA SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"CrossEtaSpreadPct={spread_pct}")
    print(f"ThresholdPct={args.success_threshold_pct}")
    print(f"Result={result}")
    print(f"RAW_PATH={raw_path}")
    print(f"SUMMARY_PATH={summary_path}")
    print(f"VERDICT_PATH={verdict_path}")

    return {
        'raw': raw_df,
        'summary': summary_df,
        'spread_pct': spread_pct,
        'result': result,
    }


def cmd_compare(args):
    """Full comparison command handler."""
    print("="*70)
    print(" Microgrid DRL Comparison Experiment")
    print(f" Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configuration
    if args.fast:
        train_steps = 5_000
        train_days = 2
        eval_days = 7
        print(" [FAST MODE] Reduced training for quick smoke test")
    else:
        train_steps = args.steps
        train_days = 30
        eval_days = args.days
    
    SEED = args.seed
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    results = {}
    
    # === SAC on Thevenin (Proposed) ===
    sac_thev_path = './models/sac_thevenin.zip'
    if os.path.exists(sac_thev_path) and not args.retrain:
        print(f"\nLoading existing model: {sac_thev_path}")
        agent_sac_thev = SAC.load(sac_thev_path)
    else:
        agent_sac_thev = train_agent('sac', 'thevenin', train_steps, train_days, SEED, sac_thev_path)
    
    results['sac_thevenin'] = evaluate_agent(
        agent_sac_thev, 'thevenin', eval_days, SEED, 
        "Proposed (SAC + Thevenin)"
    )
    
    # === SAC on Simple (Baseline - Model Mismatch) ===
    # Use fewer training steps to simulate "quick deployment" scenario
    # This amplifies the model mismatch effect
    sac_simple_path = './models/sac_simple.zip'
    simple_steps = train_steps // 10  # 10x fewer steps than proposed method
    if os.path.exists(sac_simple_path) and not args.retrain:
        print(f"\nLoading existing model: {sac_simple_path}")
        agent_sac_simple = SAC.load(sac_simple_path)
    else:
        print(f"\n[SAC+Simple] Training with {simple_steps} steps (1/10 of proposed)")
        agent_sac_simple = train_agent('sac', 'simple', simple_steps, train_days, SEED, sac_simple_path)
    
    results['sac_simple'] = evaluate_agent(
        agent_sac_simple, 'thevenin', eval_days, SEED,  # Evaluate on Thevenin!
        "SAC + Simple (Model Mismatch)"
    )
    
    # === PPO on Thevenin (Algorithm Comparison) ===
    ppo_thev_path = './models/ppo_thevenin.zip'
    if os.path.exists(ppo_thev_path) and not args.retrain:
        print(f"\nLoading existing model: {ppo_thev_path}")
        agent_ppo_thev = PPO.load(ppo_thev_path)
    else:
        agent_ppo_thev = train_agent('ppo', 'thevenin', train_steps, train_days, SEED, ppo_thev_path)

    results['ppo_thevenin'] = evaluate_agent(
        agent_ppo_thev, 'thevenin', eval_days, SEED,
        "PPO + Thevenin"
    )

    # === TD3 on Thevenin (Algorithm Comparison) ===
    td3_thev_path = './models/td3_thevenin.zip'
    if os.path.exists(td3_thev_path) and not args.retrain:
        print(f"\nLoading existing model: {td3_thev_path}")
        agent_td3_thev = TD3.load(td3_thev_path)
    else:
        agent_td3_thev = train_agent('td3', 'thevenin', train_steps, train_days, SEED, td3_thev_path)

    results['td3_thevenin'] = evaluate_agent(
        agent_td3_thev, 'thevenin', eval_days, SEED,
        "TD3 + Thevenin"
    )

    # === PPO on Simple (Weaker Baseline) ===
    ppo_simple_path = './models/ppo_simple.zip'
    ppo_steps = train_steps // 3  # Fewer steps for PPO baseline
    if os.path.exists(ppo_simple_path) and not args.retrain:
        print(f"\nLoading existing model: {ppo_simple_path}")
        agent_ppo_simple = PPO.load(ppo_simple_path)
    else:
        agent_ppo_simple = train_agent('ppo', 'simple', ppo_steps, train_days, SEED, ppo_simple_path)

    results['ppo_simple'] = evaluate_agent(
        agent_ppo_simple, 'thevenin', eval_days, SEED,
        "PPO + Simple (Baseline)"
    )

    # === TD3 on Simple (Model Mismatch) ===
    td3_simple_path = './models/td3_simple.zip'
    td3_simple_steps = train_steps // 3
    if os.path.exists(td3_simple_path) and not args.retrain:
        print(f"\nLoading existing model: {td3_simple_path}")
        agent_td3_simple = TD3.load(td3_simple_path)
    else:
        agent_td3_simple = train_agent('td3', 'simple', td3_simple_steps, train_days, SEED, td3_simple_path)

    results['td3_simple'] = evaluate_agent(
        agent_td3_simple, 'thevenin', eval_days, SEED,
        "TD3 + Simple"
    )

    # === Traditional Baselines ===
    results['no_control'] = run_no_control_baseline(eval_days, SEED)
    
    # Rule-based
    from microgrid_sim import MicrogridConfig, MicrogridEnvThevenin
    config = MicrogridConfig(simulation_days=eval_days, seed=SEED)
    env_rb = MicrogridEnvThevenin(config)
    results['rule_based'] = run_rule_based_baseline(env_rb, eval_days, "Rule-Based")
    
    # === MILP baseline (rolling-horizon perfect-foresight over the lookahead window) ===
    from microgrid_sim.baselines import run_milp_baseline
    
    # Note: this is a rolling-horizon benchmark (MPC-style). It is not a single-shot
    # full-year perfect-foresight optimum unless the horizon spans the full episode.
    env_milp = MicrogridEnvThevenin(MicrogridConfig(simulation_days=eval_days, seed=SEED))
    results['milp'] = run_milp_baseline(
        env_milp,
        simulation_days=eval_days,
        name="MILP Oracle",
        efficiency_model="simple",  # Best performing configuration
    )
    
    # === Results Summary ===
    print("\n" + "="*70)
    print(" FINAL RESULTS COMPARISON")
    print("="*70)
    
    cost_proposed = results['sac_thevenin']['total_cost']
    
    print(f"\n{'Method':<40} {'Cost':>12} {'vs Proposed':>12}")
    print("-"*70)
    
    for key, label in [
        ('milp', 'MILP (Rolling Horizon, 24h)'),
        ('sac_thevenin', 'Proposed (SAC + Thevenin)'),
        ('ppo_thevenin', 'PPO + Thevenin'),
        ('td3_thevenin', 'TD3 + Thevenin'),
        ('sac_simple', 'SAC + Simple (Model Mismatch)'),
        ('ppo_simple', 'PPO + Simple'),
        ('td3_simple', 'TD3 + Simple'),
        ('rule_based', 'Rule-Based'),
        ('no_control', 'No Control'),
    ]:
        if key in results:
            cost = results[key]['total_cost']
            pct = (cost - cost_proposed) / cost_proposed * 100
            print(f"{label:<40} ${cost:>10,.0f} {pct:>+10.1f}%")
    
    print("-"*70)
    
    # Save all results
    for key, res in results.items():
        save_results(res, OUTPUT_DIR, prefix="compare")
    
    # Summary CSV
    summary = []
    for key in ['milp', 'sac_thevenin', 'ppo_thevenin', 'td3_thevenin', 'sac_simple', 'ppo_simple', 'td3_simple', 'rule_based', 'no_control']:
        if key in results:
            r = results[key]
            summary.append({
                'Method': r['name'],
                'Total_Cost': r['total_cost'],
                'Final_SOH': r.get('final_soh', 'N/A'),
                'SOH_Degradation_%': r.get('soh_degradation', 'N/A'),
            })
    
    pd.DataFrame(summary).to_csv(f'{OUTPUT_DIR}/comparison_summary.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print(f" Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return results


def run_embedded_cli_command(command_name: str, passthrough_args: List[str]) -> None:
    """Execute an embedded command module through the unified CLI."""
    if command_name not in SCRIPT_CLI_COMMANDS:
        raise ValueError(f"Unknown script CLI command: {command_name}")

    spec = SCRIPT_CLI_COMMANDS[command_name]
    module = importlib.import_module(spec['module'])
    entrypoint = getattr(module, spec['entry'])
    if not callable(entrypoint):
        raise TypeError(f"Entrypoint {spec['entry']} in {spec['module']} is not callable")

    signature = inspect.signature(entrypoint)
    if len(signature.parameters) == 0:
        if passthrough_args:
            raise ValueError(
                f"Command '{command_name}' does not accept extra arguments: {passthrough_args}"
            )
        entrypoint()
        return
    entrypoint(passthrough_args)


def main():
    parser = argparse.ArgumentParser(
        description='Microgrid DRL Experiments CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison test
  python run_experiments.py compare --fast
  
  # Full 1-year comparison
  python run_experiments.py compare --days 365 --steps 300000
  
  # Train only SAC on Thevenin
  python run_experiments.py train --agent sac --model thevenin --steps 300000
  
  # Evaluate existing model
  python run_experiments.py eval --agent sac --model thevenin --load-model ./models/sac_thevenin.zip

  # Embedded focused CIGRE CLI
  python run_experiments.py cigre-gap-run --steps 880000 --output-dir ./results/cigre_d4_880k --models-dir ./models/cigre_d4_880k

  # Embedded focused residential CLI
  python run_experiments.py residential-d4-run --steps 300000 --output-dir ./results/residential_lowload_300k --models-dir ./models/residential_lowload_300k
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # === Train command ===
    train_parser = subparsers.add_parser('train', help='Train a DRL agent')
    train_parser.add_argument('--agent', choices=['sac', 'ppo', 'td3', 'ddpg'], default='sac', help='Agent type')
    train_parser.add_argument('--model', choices=['thevenin', 'simple'], default='thevenin', help='Battery model')
    train_parser.add_argument('--steps', type=int, default=300000, help='Training timesteps')
    train_parser.add_argument('--days', type=int, default=30, help='Days per episode')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--output', type=str, help='Model save path')
    train_parser.add_argument('--cpu', action='store_true', help='Force CPU')
    train_parser.add_argument('--eta', type=float, default=None, help='Simple model eta (charge/discharge), only used when --model simple')

    # === Eval command ===
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--agent', choices=['sac', 'ppo', 'td3', 'ddpg'], default='sac', help='Agent type')
    eval_parser.add_argument('--model', choices=['thevenin', 'simple'], default='thevenin', help='Eval environment')
    eval_parser.add_argument('--load-model', required=True, help='Model file to load')
    eval_parser.add_argument('--days', type=int, default=365, help='Evaluation days')
    eval_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    eval_parser.add_argument('--output', type=str, default='./results', help='Output directory')
    eval_parser.add_argument('--eta', type=float, default=None, help='Simple model eta (charge/discharge), only used when --model simple')

    cigre_eval_parser = subparsers.add_parser('cigre-eval', help='Evaluate trained model on MG-CIGRE (PBM/Thevenin evaluation)')
    cigre_eval_parser.add_argument('--agent', choices=['sac'], default='sac', help='Agent type (currently only sac)')
    cigre_eval_parser.add_argument('--load-model', type=str, required=True, help='Path to saved model (.zip)')
    cigre_eval_parser.add_argument('--days', type=int, default=365, help='Evaluation days (e.g., 365 for annual/8760 h)')
    cigre_eval_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    cigre_eval_parser.add_argument('--output', type=str, default=None, help='Output directory for timeseries/results')
    cigre_eval_parser.add_argument('--data-dir', type=str, default=None, help='Override MG-CIGRE data_dir (default: auto-detect current project data directory)')

    # === MILP (both microgrids) ===
    milp_both_parser = subparsers.add_parser('milp-both', help='Run LP/MPC baseline for MG-RES and MG-CIGRE')
    milp_both_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    milp_both_parser.add_argument('--res-days', type=int, default=30, help='MG-RES simulation days (e.g., 30 for 720 h)')
    milp_both_parser.add_argument('--cigre-days', type=int, default=365, help='MG-CIGRE simulation days (e.g., 365 for 8760 h)')
    milp_both_parser.add_argument('--chunk-days', type=int, default=7, help='Rolling optimization chunk size in days; use 0 for full-horizon solve')
    milp_both_parser.add_argument('--efficiency-model', choices=['simple', 'realistic'], default='simple', help='MILP internal efficiency model')
    milp_both_parser.add_argument('--paper-protocol', action='store_true', help='Use the paper-aligned annual RES/CIGRE evaluation settings')
    milp_both_parser.add_argument('--res-year', type=int, default=2024, help='Residential data year used when --paper-protocol is enabled')
    milp_both_parser.add_argument('--cigre-year', type=int, default=2024, help='CIGRE data year used when --paper-protocol is enabled')
    milp_both_parser.add_argument('--output', type=str, default='./results_milp', help='Output directory')
    milp_both_parser.add_argument('--cigre-data-dir', type=str, default=None, help='Override MG-CIGRE data_dir (default: auto-detect current project data directory)')
    milp_both_parser.add_argument('--cpu', action='store_true', help='(unused) kept for interface consistency')
    milp_both_parser.set_defaults(func=cmd_milp_both)
    
    # === Compare command ===
    compare_parser = subparsers.add_parser('compare', help='Run full comparison experiment')
    compare_parser.add_argument('--fast', action='store_true', help='Quick test mode')
    compare_parser.add_argument('--steps', type=int, default=300000, help='Training steps')
    compare_parser.add_argument('--days', type=int, default=365, help='Evaluation days')
    compare_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    compare_parser.add_argument('--output', type=str, default='./results', help='Output directory')
    compare_parser.add_argument('--retrain', action='store_true', help='Retrain even if models exist')

    # === Eta compare command ===
    eta_parser = subparsers.add_parser('eta-compare', help='Train/eval across multiple EBM efficiencies (PBM eval)')
    eta_parser.add_argument('--etas', type=str, default='0.9,0.95,1.0', help='Comma-separated EBM efficiencies')
    eta_parser.add_argument('--seeds', type=str, default='42,43,44', help='Comma-separated seeds')
    eta_parser.add_argument('--steps', type=int, default=300000, help='Training timesteps per model')
    eta_parser.add_argument('--train-days', type=int, default=365, help='Training episode days')
    eta_parser.add_argument('--eval-days', type=int, default=365, help='Evaluation days (PBM env)')
    eta_parser.add_argument('--output', type=str, default='./results/cigre/severity_eta', help='Output directory')
    eta_parser.add_argument('--models-dir', type=str, default='./models', help='Directory to save/load trained models')
    eta_parser.add_argument('--success-threshold-pct', type=float, default=1.0, help='Cross-eta spread threshold for SUCCESS')
    eta_parser.add_argument('--retrain', action='store_true', help='Retrain EBM models even if model files exist')
    eta_parser.add_argument('--retrain-pbm', action='store_true', help='Retrain PBM baseline models even if model files exist')
    eta_parser.add_argument('--cpu', action='store_true', help='Force CPU')

    # === Embedded experiment CLI commands ===
    for command_name, spec in SCRIPT_CLI_COMMANDS.items():
        subparsers.add_parser(
            command_name,
            help=f"{spec['help']} (embedded command)",
            add_help=False,
        )
    
    args, passthrough_args = parser.parse_known_args()
    
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'milp-both':
        cmd_milp_both(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'eta-compare':
        cmd_eta_compare(args)
    elif args.command == 'cigre-eval':
        cmd_cigre_eval(args)
    elif args.command in SCRIPT_CLI_COMMANDS:
        run_embedded_cli_command(args.command, passthrough_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
