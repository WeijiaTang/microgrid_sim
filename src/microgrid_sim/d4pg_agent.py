"""Lightweight DI-engine D4PG wrapper with an SB3-like interface."""

from __future__ import annotations

import copy
from collections import deque, namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
from easydict import EasyDict
from gymnasium import spaces

try:
    from ding.model.template import QACDIST
    from ding.policy import D4PGPolicy
    from ding.worker import AdvancedReplayBuffer
except ImportError as exc:                                                                  
    QACDIST = None                            
    D4PGPolicy = None                            
    AdvancedReplayBuffer = None                            
    _D4PG_IMPORT_ERROR = exc
else:
    _D4PG_IMPORT_ERROR = None


TransitionTimestep = namedtuple("TransitionTimestep", ["obs", "reward", "done", "info"])


def _ensure_d4pg_available() -> None:
    if _D4PG_IMPORT_ERROR is not None:
        raise ImportError(
            "D4PG support requires the optional DI-engine dependency. "
            "Install it with `uv pip install --python .venv\\Scripts\\python.exe DI-engine==0.5.3`."
        ) from _D4PG_IMPORT_ERROR


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_easydict_dict(data: dict[str, Any]) -> EasyDict:
    return EasyDict(copy.deepcopy(data))


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, EasyDict):
        return {key: _to_plain_data(val) for key, val in value.items()}
    if isinstance(value, dict):
        return {key: _to_plain_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return copy.deepcopy(value)


def _unwrap_training_env(env) -> Any:
    if hasattr(env, "envs") and getattr(env, "envs", None):
        return env.envs[0]
    return env


def _env_reset(env, seed: int | None = None):
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=int(seed))
        except TypeError:
            result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _env_step(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        return result
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info
    raise RuntimeError(f"Unexpected env.step return format: {type(result)} -> {result!r}")


class D4PGAgent:
    """A thin DI-engine wrapper exposing the subset of the SB3 interface used by the repo."""

    def __init__(
        self,
        env=None,
        total_steps: int = 0,
        seed: int = 0,
        device: str = "cpu",
        agent_hyperparams: dict | None = None,
        config: dict[str, Any] | None = None,
        train_iter: int = 0,
        env_step: int = 0,
    ) -> None:
        _ensure_d4pg_available()
        self.seed = int(seed)
        self.device = _resolve_device(device)
        self.total_steps = int(total_steps)
        self._agent_hyperparams = copy.deepcopy(agent_hyperparams or {})
        self._env = None
        self._current_obs: np.ndarray | None = None
        self._needs_reset = True
        self._train_iter = int(train_iter)
        self._env_step = int(env_step)
        self._steps_since_update = 0
        self._episode_window: deque[dict[str, Any]] = deque()

        if config is None:
            if env is None:
                raise ValueError("D4PGAgent requires an environment when no saved config is provided.")
            self._config = self._build_config(env=env, total_steps=total_steps, agent_hyperparams=self._agent_hyperparams)
        else:
            self._config = _to_easydict_dict(config)

        model = QACDIST(**self._config.model)
        self._policy = D4PGPolicy(self._config, model=model, enable_field=["learn", "collect", "eval"])
        buffer_cfg = AdvancedReplayBuffer.default_config()
        buffer_cfg.replay_buffer_size = int(self._config.other.replay_buffer.replay_buffer_size)
        self._buffer = AdvancedReplayBuffer(
            buffer_cfg,
            tb_logger=None,
            exp_name="microgrid_sim_d4pg",
            instance_name=f"d4pg_buffer_seed{self.seed}_{id(self):x}",
        )

        self._warmup_steps = int(self._config.learn.warmup_steps)
        self._collect_n_sample = int(self._config.collect.n_sample)
        self._update_per_collect = int(self._config.learn.update_per_collect)
        self._batch_size = int(self._config.learn.batch_size)
        self._nstep = int(self._config.nstep)
        self._action_shape = tuple(int(v) for v in np.atleast_1d(self._config.model.action_shape))
        if len(self._action_shape) == 1 and self._action_shape[0] <= 1:
            self._action_shape = (int(self._config.model.action_shape),)

        if env is not None:
            self.set_env(env)

    @classmethod
    def load(cls, model_path: str, env=None, device: str = "auto") -> "D4PGAgent":
        _ensure_d4pg_available()
        map_location = _resolve_device(device)
        payload = torch.load(model_path, map_location=map_location, weights_only=False)
        agent = cls(
            env=env,
            total_steps=int(payload.get("total_steps", 0)),
            seed=int(payload["seed"]),
            device=map_location,
            agent_hyperparams=payload.get("agent_hyperparams", {}),
            config=payload["config"],
            train_iter=int(payload.get("train_iter", 0)),
            env_step=int(payload.get("env_step", 0)),
        )
        agent._policy.learn_mode.load_state_dict(payload["learn_state"])
        agent._policy.collect_mode.load_state_dict(payload["collect_state"])
        agent._policy.eval_mode.load_state_dict(payload["eval_state"])
        return agent

    def save(self, model_path: str) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": int(self.seed),
            "total_steps": int(self.total_steps),
            "train_iter": int(self._train_iter),
            "env_step": int(self._env_step),
            "agent_hyperparams": _to_plain_data(self._agent_hyperparams),
            "config": _to_plain_data(self._config),
            "learn_state": self._policy.learn_mode.state_dict(),
            "collect_state": self._policy.collect_mode.state_dict(),
            "eval_state": self._policy.eval_mode.state_dict(),
        }
        torch.save(payload, path)

    def set_env(self, env) -> None:
        raw_env = _unwrap_training_env(env)
        action_space = getattr(raw_env, "action_space", None)
        observation_space = getattr(raw_env, "observation_space", None)
        if not isinstance(action_space, spaces.Box):
            raise TypeError("D4PGAgent requires a continuous Box action space.")
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("D4PGAgent requires a Box observation space.")
        self._env = raw_env
        self._needs_reset = True
        self._current_obs = None
        self._episode_window.clear()
        self._steps_since_update = 0

    def learn(self, total_timesteps: int, progress_bar: bool = False, reset_num_timesteps: bool = True):
        if self._env is None:
            raise RuntimeError("Call set_env before training D4PGAgent.")
        steps_to_run = max(int(total_timesteps), 0)
        if steps_to_run == 0:
            return self
        if reset_num_timesteps:
            self._train_iter = 0
            self._env_step = 0
            self._steps_since_update = 0
            self._current_obs = None
            self._needs_reset = True
            self._episode_window.clear()
            self._buffer.clear()

        target_env_step = self._env_step + steps_to_run
        next_log_step = self._env_step + max(steps_to_run // 10, 1)

        while self._env_step < target_env_step:
            if self._needs_reset or self._current_obs is None:
                obs, _ = _env_reset(self._env, seed=self.seed)
                self._current_obs = self._flatten_obs(obs)
                self._needs_reset = False
                self._episode_window.clear()

            use_random_action = self._env_step < self._warmup_steps
            action_tensor, action_np = self._select_action(self._current_obs, use_random_action=use_random_action)
            next_obs, reward, terminated, truncated, info = _env_step(self._env, action_np)
            done = bool(terminated or truncated)
            next_obs_flat = self._flatten_obs(next_obs)
            transition = self._policy.collect_mode.process_transition(
                torch.as_tensor(self._current_obs, dtype=torch.float32),
                {"action": action_tensor},
                TransitionTimestep(
                    obs=torch.as_tensor(next_obs_flat, dtype=torch.float32),
                    reward=torch.tensor([float(reward)], dtype=torch.float32),
                    done=torch.tensor(done),
                    info=info,
                ),
            )
            transition["collect_iter"] = int(self._train_iter)
            self._episode_window.append(transition)
            self._emit_ready_samples(done=done)

            self._current_obs = next_obs_flat
            self._env_step += 1
            self._steps_since_update += 1

            if self._env_step >= self._warmup_steps and self._steps_since_update >= self._collect_n_sample:
                self._run_update_block()
                self._steps_since_update = 0

            if done:
                self._needs_reset = True
                self._current_obs = None

            if progress_bar and self._env_step >= next_log_step:
                print(
                    f"D4PG progress: env_step={self._env_step:,}/{target_env_step:,}, "
                    f"train_iter={self._train_iter:,}, buffer={self._buffer.count():,}"
                )
                next_log_step += max(steps_to_run // 10, 1)

        return self

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        obs = self._flatten_obs(observation)
        output = self._policy.eval_mode.forward({0: obs})[0]
        action_tensor = output["action"]
        if isinstance(action_tensor, torch.Tensor):
            action_np = action_tensor.detach().cpu().numpy()
        else:
            action_np = np.asarray(action_tensor, dtype=np.float32)
        action_np = action_np.astype(np.float32).reshape(self._action_shape)
        return action_np, None

    def _build_config(self, env, total_steps: int, agent_hyperparams: dict[str, Any]) -> EasyDict:
        raw_env = _unwrap_training_env(env)
        observation_space = getattr(raw_env, "observation_space", None)
        action_space = getattr(raw_env, "action_space", None)
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("D4PGAgent requires Box observations.")
        if not isinstance(action_space, spaces.Box):
            raise TypeError("D4PGAgent requires Box actions.")

        hidden_sizes = [int(size) for size in agent_hyperparams.get("net_arch", (256, 128, 64)) if int(size) > 0]
        hidden_sizes = hidden_sizes or [256, 128, 64]
        hidden_size = int(hidden_sizes[0])
        layer_num = max(len(hidden_sizes), 1)
        obs_shape = int(np.prod(observation_space.shape))
        action_shape = int(np.prod(action_space.shape))
        replay_buffer_size = int(agent_hyperparams.get("d4pg_replay_buffer_size", min(1_000_000, max(50_000, int(total_steps)))))
        learning_rate = float(agent_hyperparams.get("learning_rate", 3e-4))
        batch_size = max(int(agent_hyperparams.get("d4pg_batch_size", 256)), 1)
        warmup_steps = max(int(agent_hyperparams.get("d4pg_learning_starts", min(max(int(total_steps) // 10, batch_size), 1_000))), batch_size)
        collect_n_sample = max(int(agent_hyperparams.get("d4pg_collect_n_sample", 32)), 1)
        update_per_collect = max(int(agent_hyperparams.get("d4pg_update_per_collect", 2)), 1)
        gamma = float(agent_hyperparams.get("gamma", 0.985))
        tau = float(agent_hyperparams.get("tau", 0.003))
        noise_sigma = max(float(agent_hyperparams.get("d4pg_action_noise_sigma", agent_hyperparams.get("ddpg_action_noise_sigma", 0.10))), 0.0)
        nstep = max(int(agent_hyperparams.get("d4pg_n_step", 3)), 1)
        n_atom = max(int(agent_hyperparams.get("d4pg_n_atom", 51)), 2)

        env_name = str(getattr(getattr(raw_env, "unwrapped", raw_env), "__class__", type(raw_env)).__name__).lower()
        default_v_min = -12_000.0 if "cigre" in env_name else -500.0
        default_v_max = 50.0 if "cigre" in env_name else 20.0
        v_min = float(agent_hyperparams.get("d4pg_v_min", default_v_min))
        v_max = float(agent_hyperparams.get("d4pg_v_max", default_v_max))
        if v_max <= v_min:
            v_max = v_min + 1.0

        cfg = D4PGPolicy.default_config()
        cfg.cuda = self.device == "cuda"
        cfg.priority = True
        cfg.priority_IS_weight = True
        cfg.random_collect_size = 0
        cfg.reward_batch_norm = False
        cfg.nstep = nstep
        cfg.model.obs_shape = obs_shape
        cfg.model.action_shape = action_shape
        cfg.model.action_space = "regression"
        cfg.model.actor_head_hidden_size = hidden_size
        cfg.model.actor_head_layer_num = layer_num
        cfg.model.critic_head_hidden_size = hidden_size
        cfg.model.critic_head_layer_num = layer_num
        cfg.model.v_min = v_min
        cfg.model.v_max = v_max
        cfg.model.n_atom = n_atom
        cfg.learn.batch_size = batch_size
        cfg.learn.learning_rate_actor = learning_rate
        cfg.learn.learning_rate_critic = learning_rate
        cfg.learn.discount_factor = gamma
        cfg.learn.target_theta = tau
        cfg.learn.update_per_collect = update_per_collect
        cfg.learn.actor_update_freq = 1
        cfg.learn.noise = False
        cfg.learn.ignore_done = False
        cfg.learn.warmup_steps = warmup_steps
        cfg.collect.noise_sigma = noise_sigma
        cfg.collect.n_sample = collect_n_sample
        cfg.collect.unroll_len = 1
        cfg.eval.evaluator.eval_freq = 10**9
        cfg.other.replay_buffer.replay_buffer_size = replay_buffer_size
        return cfg

    def _flatten_obs(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        return obs.reshape(-1)

    def _select_action(self, observation: np.ndarray, use_random_action: bool) -> tuple[torch.Tensor, np.ndarray]:
        if use_random_action:
            action_np = np.asarray(self._env.action_space.sample(), dtype=np.float32).reshape(self._action_shape)
            action_tensor = torch.as_tensor(action_np.reshape(-1), dtype=torch.float32)
            return action_tensor, action_np
        policy_output = self._policy.collect_mode.forward({0: observation})[0]
        action_tensor = policy_output["action"]
        if not isinstance(action_tensor, torch.Tensor):
            action_tensor = torch.as_tensor(action_tensor, dtype=torch.float32)
        action_np = action_tensor.detach().cpu().numpy().astype(np.float32).reshape(self._action_shape)
        return action_tensor.detach().cpu().reshape(-1), action_np

    def _emit_ready_samples(self, done: bool) -> None:
        if done:
            if self._episode_window:
                tail_samples = list(self._policy.collect_mode.get_train_sample(copy.deepcopy(self._episode_window)))
                if tail_samples:
                    self._buffer.push(tail_samples, cur_collector_envstep=self._env_step)
            self._episode_window.clear()
            return

        while len(self._episode_window) >= self._nstep + 1:
            prefix = deque(copy.deepcopy(list(self._episode_window)[: self._nstep + 1]))
            samples = list(self._policy.collect_mode.get_train_sample(prefix))
            if samples:
                self._buffer.push(samples[0], cur_collector_envstep=self._env_step)
            self._episode_window.popleft()

    def _run_update_block(self) -> None:
        if self._buffer.count() < self._batch_size:
            return
        for _ in range(self._update_per_collect):
            batch = self._buffer.sample(self._batch_size, self._train_iter)
            if batch is None:
                break
            learn_output = self._policy.learn_mode.forward(batch)
            priorities = learn_output.get("priority")
            if priorities is not None:
                self._buffer.update(
                    {
                        "replay_unique_id": [item["replay_unique_id"] for item in batch],
                        "replay_buffer_idx": [item["replay_buffer_idx"] for item in batch],
                        "priority": priorities,
                    }
                )
            self._train_iter += 1
