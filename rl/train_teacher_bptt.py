from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch import nn
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

from rl.env import make_env
from rl.features import TeacherDictFeaturesExtractor


def _sanitize_config_for_logging(config: dict) -> dict:
    def _convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): _convert(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    return _convert(config)


def maybe_init_wandb(config: dict, out_dir: Path):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    run = wandb.init(
        project=wandb_cfg.get("project", "quadruped-teacher"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        group=wandb_cfg.get("group"),
        tags=list(wandb_cfg.get("tags", [])),
        dir=str(out_dir),
        config=_sanitize_config_for_logging(config),
        sync_tensorboard=False,
        save_code=False,
        mode=wandb_cfg.get("mode", "online"),
    )
    if run is not None:
        wandb.define_metric("train/timesteps")
        wandb.define_metric("eval/timesteps")
        wandb.define_metric("train/*", step_metric="train/timesteps")
        wandb.define_metric("eval/*", step_metric="eval/timesteps")
        wandb.define_metric("curriculum/*", step_metric="train/timesteps")
    return run


def maybe_log_wandb(data: dict, commit: bool = True) -> None:
    if wandb.run is not None:
        wandb.log(data, commit=commit)


def maybe_log_wandb_video(key: str, video_path: Path, fps: int) -> None:
    if wandb.run is None or not video_path.exists():
        return
    wandb.log({key: wandb.Video(str(video_path), fps=fps, format="mp4")}, commit=False)


def maybe_save_wandb(path: Path, out_dir: Path) -> None:
    if wandb.run is not None and path.exists():
        wandb.save(str(path), base_path=str(out_dir))


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.size == 0:
            return
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, batch: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (batch - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class DictObsNormalizer:
    def __init__(self, observation_space, keys: list[str], clip: float = 10.0):
        self.keys = list(keys)
        self.clip = float(clip)
        self.stats = {
            key: RunningMeanStd(tuple(observation_space.spaces[key].shape))
            for key in self.keys
            if key in observation_space.spaces and int(np.prod(observation_space.spaces[key].shape)) > 0
        }

    def update(self, obs: dict[str, np.ndarray]) -> None:
        for key, rms in self.stats.items():
            if key in obs:
                rms.update(obs[key])

    def normalize(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {key: np.asarray(value, dtype=np.float32).copy() for key, value in obs.items()}
        for key, rms in self.stats.items():
            if key in out:
                out[key] = rms.normalize(out[key], clip=self.clip)
        return out

    def state_dict(self) -> dict:
        return {key: rms.state_dict() for key, rms in self.stats.items()}

    def load_state_dict(self, state: dict) -> None:
        for key, rms_state in state.items():
            if key in self.stats:
                self.stats[key].load_state_dict(rms_state)


def stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = obs_list[0].keys()
    return {key: np.stack([obs[key] for obs in obs_list], axis=0).astype(np.float32) for key in keys}


def obs_to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items()}


def make_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
        last_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class TeacherGruActorCritic(nn.Module):
    def __init__(self, observation_space, action_dim: int, policy_cfg: dict):
        super().__init__()
        self.extractor = TeacherDictFeaturesExtractor(
            observation_space,
            proprio_hidden_dim=int(policy_cfg.get("proprio_hidden_dim", 128)),
            scandot_hidden_dim=int(policy_cfg.get("scandot_hidden_dim", 128)),
            command_hidden_dim=int(policy_cfg.get("command_hidden_dim", 32)),
            privileged_hidden_dim=int(policy_cfg.get("privileged_hidden_dim", 32)),
            fused_dim=int(policy_cfg.get("fused_dim", 256)),
            encoder_variant=str(policy_cfg.get("teacher_encoder_variant", "paper")),
            include_privileged=bool(policy_cfg.get("include_privileged", False)),
        )
        hidden_size = int(policy_cfg.get("lstm_hidden_size", 256))
        n_layers = int(policy_cfg.get("n_lstm_layers", 1))
        self.gru = nn.GRU(self.extractor.features_dim, hidden_size, num_layers=n_layers, batch_first=True)
        hidden_sizes = list(policy_cfg.get("hidden_sizes", [256, 256]))
        self.actor = make_mlp(hidden_size, hidden_sizes, action_dim)
        self.critic = make_mlp(hidden_size, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * float(policy_cfg.get("init_log_std", -1.0)))
        self.min_log_std = float(policy_cfg.get("min_log_std", -2.0))
        self.max_log_std = float(policy_cfg.get("max_log_std", -0.5))

    @property
    def hidden_size(self) -> int:
        return int(self.gru.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self.gru.num_layers)

    def _features_sequence(self, obs: dict[str, torch.Tensor], batch_size: int, seq_len: int) -> torch.Tensor:
        flat_obs = {key: value.reshape((batch_size * seq_len, *value.shape[2:])) for key, value in obs.items()}
        features = self.extractor(flat_obs)
        return features.reshape(batch_size, seq_len, -1)

    def recurrent_forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = next(iter(obs.values())).shape[:2]
        features = self._features_sequence(obs, batch_size, seq_len)
        outputs = []
        h = hidden
        for step_idx in range(seq_len):
            reset_mask = episode_starts[:, step_idx].float().view(1, batch_size, 1)
            h = h * (1.0 - reset_mask)
            out, h = self.gru(features[:, step_idx : step_idx + 1], h)
            outputs.append(out)
        return torch.cat(outputs, dim=1), h

    def distribution_and_value(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        latent, next_hidden = self.recurrent_forward(obs, hidden, episode_starts)
        mean = self.actor(latent)
        value = self.critic(latent).squeeze(-1)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).view(1, 1, -1).expand_as(mean)
        return torch.distributions.Normal(mean, std), value, next_hidden

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        seq_obs = {key: value.unsqueeze(1) for key, value in obs.items()}
        starts = episode_start.view(-1, 1)
        dist, value, next_hidden = self.distribution_and_value(seq_obs, hidden, starts)
        action_tensor = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        raw_action = action_tensor.squeeze(1)
        env_action = torch.clamp(raw_action, -1.0, 1.0)
        return (
            env_action.cpu().numpy(),
            raw_action.cpu().numpy(),
            next_hidden,
            log_prob.squeeze(1).cpu().numpy(),
            value.squeeze(1).cpu().numpy(),
            dist.entropy().sum(dim=-1).squeeze(1).cpu().numpy(),
        )

    def evaluate_sequence(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, _ = self.distribution_and_value(obs, hidden, episode_starts)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class RolloutBatch:
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_starts: np.ndarray
    values: np.ndarray
    hidden_states: np.ndarray
    last_values: np.ndarray
    last_dones: np.ndarray


def collect_rollout(
    envs,
    policy: TeacherGruActorCritic,
    normalizer: DictObsNormalizer,
    rollout_steps: int,
    device: torch.device,
) -> tuple[RolloutBatch, list[dict], list[dict]]:
    num_envs = len(envs)
    if not hasattr(collect_rollout, "_obs"):
        obs_list = []
        infos = []
        for env in envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos.append(info)
        collect_rollout._obs = obs_list
        collect_rollout._episode_start = np.ones(num_envs, dtype=bool)
        collect_rollout._hidden = torch.zeros(policy.num_layers, num_envs, policy.hidden_size, device=device)

    obs_list = collect_rollout._obs
    episode_start = collect_rollout._episode_start
    hidden = collect_rollout._hidden

    obs_storage = {key: [] for key in obs_list[0].keys()}
    actions = []
    log_probs = []
    rewards = []
    dones = []
    episode_starts = []
    values = []
    hidden_states = []
    infos_flat: list[dict] = []

    for _ in range(rollout_steps):
        stacked = stack_obs(obs_list)
        normalizer.update(stacked)
        norm_obs = normalizer.normalize(stacked)
        for key, value in norm_obs.items():
            obs_storage[key].append(value)
        episode_starts.append(episode_start.copy())
        hidden_states.append(hidden.detach().cpu().numpy())
        torch_obs = obs_to_torch(norm_obs, device)
        env_action, raw_action, next_hidden, log_prob, value, _ = policy.act(
            torch_obs,
            hidden,
            torch.as_tensor(episode_start, dtype=torch.bool, device=device),
            deterministic=False,
        )

        next_obs_list = []
        step_rewards = []
        step_dones = []
        next_episode_start = np.zeros(num_envs, dtype=bool)
        for env_idx, env in enumerate(envs):
            next_obs, reward, terminated, truncated, info = env.step(env_action[env_idx])
            done = bool(terminated or truncated)
            infos_flat.append(info)
            if done:
                next_obs, _ = env.reset()
                next_hidden[:, env_idx] = 0.0
                next_episode_start[env_idx] = True
            next_obs_list.append(next_obs)
            step_rewards.append(float(reward))
            step_dones.append(done)

        actions.append(raw_action.astype(np.float32))
        log_probs.append(log_prob.astype(np.float32))
        values.append(value.astype(np.float32))
        rewards.append(np.asarray(step_rewards, dtype=np.float32))
        dones.append(np.asarray(step_dones, dtype=bool))

        obs_list = next_obs_list
        episode_start = next_episode_start
        hidden = next_hidden.detach()

    stacked = stack_obs(obs_list)
    norm_obs = normalizer.normalize(stacked)
    with torch.no_grad():
        torch_obs = {key: value.unsqueeze(1) for key, value in obs_to_torch(norm_obs, device).items()}
        starts = torch.as_tensor(episode_start, dtype=torch.bool, device=device).view(num_envs, 1)
        _, last_values_tensor, _ = policy.distribution_and_value(torch_obs, hidden, starts)

    collect_rollout._obs = obs_list
    collect_rollout._episode_start = episode_start
    collect_rollout._hidden = hidden

    batch = RolloutBatch(
        observations={key: np.stack(value, axis=0).astype(np.float32) for key, value in obs_storage.items()},
        actions=np.stack(actions, axis=0).astype(np.float32),
        log_probs=np.stack(log_probs, axis=0).astype(np.float32),
        rewards=np.stack(rewards, axis=0).astype(np.float32),
        dones=np.stack(dones, axis=0),
        episode_starts=np.stack(episode_starts, axis=0),
        values=np.stack(values, axis=0).astype(np.float32),
        hidden_states=np.stack(hidden_states, axis=0).astype(np.float32),
        last_values=last_values_tensor.squeeze(1).cpu().numpy().astype(np.float32),
        last_dones=episode_start.copy(),
    )
    return batch, infos_flat, []


def compute_gae(batch: RolloutBatch, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(batch.rewards, dtype=np.float32)
    last_gae = np.zeros(batch.rewards.shape[1], dtype=np.float32)
    for step in reversed(range(batch.rewards.shape[0])):
        if step == batch.rewards.shape[0] - 1:
            next_non_terminal = 1.0 - batch.last_dones.astype(np.float32)
            next_values = batch.last_values
        else:
            next_non_terminal = 1.0 - batch.dones[step].astype(np.float32)
            next_values = batch.values[step + 1]
        delta = batch.rewards[step] + gamma * next_values * next_non_terminal - batch.values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + batch.values
    return advantages, returns


def iter_sequences(
    batch: RolloutBatch,
    advantages: np.ndarray,
    returns: np.ndarray,
    sequence_length: int,
    minibatch_size: int,
) -> Iterator[dict]:
    rollout_steps, num_envs = batch.rewards.shape
    chunks = []
    for env_idx in range(num_envs):
        for start in range(0, rollout_steps, sequence_length):
            end = min(start + sequence_length, rollout_steps)
            chunks.append((env_idx, start, end))
    np.random.shuffle(chunks)
    seqs_per_batch = max(1, minibatch_size // sequence_length)
    for offset in range(0, len(chunks), seqs_per_batch):
        selected = chunks[offset : offset + seqs_per_batch]
        batch_size = len(selected)
        max_len = max(end - start for _, start, end in selected)
        mask = np.zeros((batch_size, max_len), dtype=bool)
        obs = {
            key: np.zeros((batch_size, max_len, *value.shape[2:]), dtype=np.float32)
            for key, value in batch.observations.items()
        }
        actions = np.zeros((batch_size, max_len, batch.actions.shape[-1]), dtype=np.float32)
        old_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
        old_values = np.zeros((batch_size, max_len), dtype=np.float32)
        adv = np.zeros((batch_size, max_len), dtype=np.float32)
        ret = np.zeros((batch_size, max_len), dtype=np.float32)
        starts_arr = np.zeros((batch_size, max_len), dtype=bool)
        hidden = np.zeros((batch.hidden_states.shape[1], batch_size, batch.hidden_states.shape[3]), dtype=np.float32)
        for seq_idx, (env_idx, start, end) in enumerate(selected):
            length = end - start
            mask[seq_idx, :length] = True
            for key in obs:
                obs[key][seq_idx, :length] = batch.observations[key][start:end, env_idx]
            actions[seq_idx, :length] = batch.actions[start:end, env_idx]
            old_log_probs[seq_idx, :length] = batch.log_probs[start:end, env_idx]
            old_values[seq_idx, :length] = batch.values[start:end, env_idx]
            adv[seq_idx, :length] = advantages[start:end, env_idx]
            ret[seq_idx, :length] = returns[start:end, env_idx]
            starts_arr[seq_idx, :length] = batch.episode_starts[start:end, env_idx]
            hidden[:, seq_idx] = batch.hidden_states[start, :, env_idx]
        yield {
            "obs": obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": adv,
            "returns": ret,
            "episode_starts": starts_arr,
            "hidden": hidden,
            "mask": mask,
        }


def _video_output_path(out_dir: Path, prefix: str, curriculum_level: int | None) -> Path:
    if curriculum_level is None:
        return out_dir / f"{prefix}.mp4"
    return out_dir / f"{prefix}_level_{curriculum_level}.mp4"


def _is_strict_success(distance_m: float, avg_forward_velocity: float, terminated: bool, eval_cfg: dict) -> bool:
    return (
        distance_m >= float(eval_cfg.get("success_distance_m", 1.2))
        and avg_forward_velocity >= float(eval_cfg.get("min_avg_forward_velocity", 0.06))
        and not terminated
    )


def _is_weak_success(distance_m: float, terminated: bool, eval_cfg: dict) -> bool:
    return distance_m >= float(eval_cfg.get("weak_success_distance_m", 0.45)) and not terminated


def record_video(
    policy,
    env_cfg,
    normalizer,
    eval_cfg,
    device,
    output_path: Path,
    reset_options: dict | None = None,
) -> dict:
    env = make_env({**env_cfg, "seed": int(env_cfg.get("seed", 0)) + 2001, "reset_noise_scale": 0.0})
    if hasattr(env, "set_preview_highlight"):
        env.set_preview_highlight(bool(eval_cfg.get("highlight_preview_terrain", True)))
        env.set_scandot_overlay(bool(eval_cfg.get("render_scandots", True)))
    obs, _ = env.reset(options=reset_options)
    hidden = torch.zeros(policy.num_layers, 1, policy.hidden_size, device=device)
    episode_start = np.ones(1, dtype=bool)
    frames = []
    done = False
    total_reward = 0.0
    steps = 0
    terminated = False
    last_info = {"x_position": 0.0}
    try:
        frames.append(
            env.render_frame(
                width=int(eval_cfg.get("video_width", 512)),
                height=int(eval_cfg.get("video_height", 384)),
                camera=str(eval_cfg.get("video_camera", "tracking")),
            )
        )
        while not done:
            stacked = stack_obs([obs])
            norm_obs = normalizer.normalize(stacked)
            env_action, _, hidden, _, _, _ = policy.act(
                obs_to_torch(norm_obs, device),
                hidden,
                torch.as_tensor(episode_start, dtype=torch.bool, device=device),
                deterministic=True,
            )
            obs, reward, terminated, truncated, last_info = env.step(env_action[0])
            done = bool(terminated or truncated)
            episode_start[:] = done
            total_reward += float(reward)
            steps += 1
            frames.append(
                env.render_frame(
                    width=int(eval_cfg.get("video_width", 512)),
                    height=int(eval_cfg.get("video_height", 384)),
                    camera=str(eval_cfg.get("video_camera", "tracking")),
                )
            )
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=int(eval_cfg.get("video_fps", 24)))
    distance = float(last_info["x_position"])
    avg_speed = distance / max(steps * float(env_cfg.get("control_dt", 0.02)), 1e-6)
    return {
        "distance": distance,
        "avg_forward_velocity": avg_speed,
        "return": total_reward,
        "weak_success": _is_weak_success(distance, bool(terminated), eval_cfg),
        "strict_success": _is_strict_success(distance, avg_speed, bool(terminated), eval_cfg),
    }


def evaluate(policy, env_cfg, normalizer, eval_cfg, device, out_dir: Path, step: int) -> dict:
    env = make_env({**env_cfg, "seed": int(env_cfg.get("seed", 0)) + 1001, "reset_noise_scale": 0.0})
    if hasattr(env, "set_preview_highlight"):
        env.set_preview_highlight(bool(eval_cfg.get("highlight_preview_terrain", True)))
        env.set_scandot_overlay(bool(eval_cfg.get("render_scandots", True)))
    episode_count = int(eval_cfg.get("eval_episodes", 5))
    distances, returns, speeds, falls = [], [], [], []
    for ep_idx in range(episode_count):
        obs, _ = env.reset(options={"terrain_type": eval_cfg.get("showcase_terrain_type", None)})
        hidden = torch.zeros(policy.num_layers, 1, policy.hidden_size, device=device)
        episode_start = np.ones(1, dtype=bool)
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {"x_position": 0.0}
        while not done:
            stacked = stack_obs([obs])
            norm_obs = normalizer.normalize(stacked)
            env_action, _, hidden, _, _, _ = policy.act(
                obs_to_torch(norm_obs, device),
                hidden,
                torch.as_tensor(episode_start, dtype=torch.bool, device=device),
                deterministic=True,
            )
            obs, reward, terminated, truncated, last_info = env.step(env_action[0])
            done = bool(terminated or truncated)
            episode_start[:] = done
            total_reward += float(reward)
            steps += 1
        distance = float(last_info["x_position"])
        distances.append(distance)
        returns.append(total_reward)
        speeds.append(distance / max(steps * env.config.control_dt, 1e-6))
        falls.append(float(last_info.get("TimeLimit.truncated", False) is False and done))
    env.close()
    weak_threshold = float(eval_cfg.get("weak_success_distance_m", 0.45))
    strict_threshold = float(eval_cfg.get("success_distance_m", 1.2))
    min_speed = float(eval_cfg.get("min_avg_forward_velocity", 0.06))
    return {
        "avg_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "avg_return": float(np.mean(returns)),
        "avg_forward_velocity": float(np.mean(speeds)),
        "weak_success_rate": float(np.mean(np.asarray(distances) >= weak_threshold)),
        "strict_success_rate": float(np.mean((np.asarray(distances) >= strict_threshold) & (np.asarray(speeds) >= min_speed))),
    }


def save_checkpoint(path: Path, policy, optimizer, normalizer, config, timesteps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "config": config,
            "timesteps": timesteps,
        },
        path,
    )


def train(config: dict) -> None:
    train_cfg = config["train"]
    env_cfg = config["env"]
    eval_cfg = config.get("eval", {})
    device_name = str(train_cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("requested CUDA but no GPU is available; falling back to CPU", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)
    seed = int(train_cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    for attr in ("_obs", "_episode_start", "_hidden"):
        if hasattr(collect_rollout, attr):
            delattr(collect_rollout, attr)

    out_dir = Path(train_cfg.get("output_dir", "artifacts/teacher_bptt"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    wandb_run = maybe_init_wandb(config, out_dir)

    envs = [make_env({**env_cfg, "seed": int(env_cfg.get("seed", seed)) + idx}) for idx in range(int(train_cfg.get("num_envs", 1)))]
    normalizer = DictObsNormalizer(
        envs[0].observation_space,
        keys=list(train_cfg.get("normalize_obs_keys", [])),
        clip=float(train_cfg.get("clip_observations", 10.0)),
    )
    policy = TeacherGruActorCritic(envs[0].observation_space, envs[0].action_space.shape[0], config["policy"]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)))

    rollout_steps = int(train_cfg.get("rollout_steps", 512))
    sequence_length = int(train_cfg.get("truncated_bptt_steps", train_cfg.get("recurrent_sequence_length", 24)))
    minibatch_size = int(train_cfg.get("minibatch_size", 512))
    total_timesteps = int(train_cfg.get("total_timesteps", rollout_steps * int(train_cfg.get("total_updates", 1000))))
    eval_interval = int(eval_cfg.get("eval_interval_timesteps", 327680))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 25))
    num_envs = len(envs)
    timesteps = 0
    update_idx = 0
    best_distance = -np.inf
    eval_csv = out_dir / "eval_metrics.csv"
    train_csv = out_dir / "train_metrics.csv"

    print(
        f"[bptt-trainer] rollout_steps={rollout_steps} sequence_length={sequence_length} "
        f"num_envs={num_envs} minibatch_size={minibatch_size}",
        flush=True,
    )
    while timesteps < total_timesteps:
        batch, infos, _ = collect_rollout(envs, policy, normalizer, rollout_steps, device)
        timesteps += rollout_steps * num_envs
        update_idx += 1
        advantages, returns = compute_gae(batch, float(train_cfg["gamma"]), float(train_cfg["gae_lambda"]))
        valid_adv = advantages.reshape(-1)
        adv_mean = float(np.mean(valid_adv))
        adv_std = float(np.std(valid_adv) + 1e-8)
        advantages = (advantages - adv_mean) / adv_std

        approx_kls = []
        clip_fracs = []
        losses = []
        for _ in range(int(train_cfg.get("ppo_epochs", 4))):
            for mini in iter_sequences(batch, advantages, returns, sequence_length, minibatch_size):
                obs = obs_to_torch(mini["obs"], device)
                actions = torch.as_tensor(mini["actions"], dtype=torch.float32, device=device)
                old_log_probs = torch.as_tensor(mini["old_log_probs"], dtype=torch.float32, device=device)
                old_values = torch.as_tensor(mini["old_values"], dtype=torch.float32, device=device)
                adv = torch.as_tensor(mini["advantages"], dtype=torch.float32, device=device)
                ret = torch.as_tensor(mini["returns"], dtype=torch.float32, device=device)
                starts = torch.as_tensor(mini["episode_starts"], dtype=torch.bool, device=device)
                hidden = torch.as_tensor(mini["hidden"], dtype=torch.float32, device=device)
                mask = torch.as_tensor(mini["mask"], dtype=torch.bool, device=device)

                log_prob, entropy, values = policy.evaluate_sequence(obs, hidden, starts, actions)
                ratio = torch.exp(log_prob - old_log_probs)
                clip_coef = float(train_cfg.get("clip_coef", 0.2))
                pg_loss_1 = -adv * ratio
                pg_loss_2 = -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.mean(torch.maximum(pg_loss_1, pg_loss_2)[mask])
                value_loss = torch.mean(torch.square(ret - values)[mask])
                entropy_loss = -torch.mean(entropy[mask])
                loss = (
                    policy_loss
                    + float(train_cfg.get("value_coef", 0.5)) * value_loss
                    + float(train_cfg.get("entropy_coef", 0.0)) * entropy_loss
                )

                with torch.no_grad():
                    log_ratio = log_prob - old_log_probs
                    approx_kl = torch.mean(((torch.exp(log_ratio) - 1.0) - log_ratio)[mask]).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_coef).float()[mask]).item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), float(train_cfg.get("max_grad_norm", 0.5)))
                optimizer.step()
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                losses.append(float(loss.item()))
            target_kl = train_cfg.get("target_kl")
            if target_kl is not None and approx_kls and np.mean(approx_kls[-4:]) > 1.5 * float(target_kl):
                break

        if update_idx % max(1, checkpoint_interval) == 0:
            save_checkpoint(out_dir / f"checkpoint_{timesteps}.pt", policy, optimizer, normalizer, config, timesteps)

        recent_infos = infos[-num_envs * 10 :] if infos else []
        avg_x = float(np.mean([info.get("x_position", 0.0) for info in recent_infos])) if recent_infos else 0.0
        avg_forward_velocity = (
            float(np.mean([info.get("forward_velocity", 0.0) for info in recent_infos])) if recent_infos else 0.0
        )
        avg_terrain_level = (
            float(np.mean([info.get("terrain_level", 0.0) for info in recent_infos])) if recent_infos else 0.0
        )
        mean_reward = float(np.mean(batch.rewards))
        train_metrics = {
            "timesteps": timesteps,
            "update": update_idx,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "mean_reward": mean_reward,
            "recent_x_position": avg_x,
            "recent_forward_velocity": avg_forward_velocity,
            "terrain_level": avg_terrain_level,
            "log_std_mean": float(torch.clamp(policy.log_std, policy.min_log_std, policy.max_log_std).mean().item()),
        }
        write_train_header = not train_csv.exists()
        with open(train_csv, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(train_metrics.keys()))
            if write_train_header:
                writer.writeheader()
            writer.writerow(train_metrics)
        maybe_log_wandb(
            {
                "train/timesteps": timesteps,
                "train/update": update_idx,
                "train/loss": train_metrics["loss"],
                "train/approx_kl": train_metrics["approx_kl"],
                "train/clip_fraction": train_metrics["clip_fraction"],
                "train/mean_reward": train_metrics["mean_reward"],
                "train/recent_x_position": train_metrics["recent_x_position"],
                "train/recent_forward_velocity": train_metrics["recent_forward_velocity"],
                "train/log_std_mean": train_metrics["log_std_mean"],
                "curriculum/mean_level": train_metrics["terrain_level"],
            },
            commit=False,
        )

        if update_idx % 5 == 0:
            print(
                f"[train] steps={timesteps} update={update_idx} loss={train_metrics['loss']:.3f} "
                f"kl={train_metrics['approx_kl']:.5f} clip={train_metrics['clip_fraction']:.3f} "
                f"recent_x={avg_x:.3f} level={avg_terrain_level:.1f}",
                flush=True,
            )

        if timesteps >= eval_interval and (timesteps - rollout_steps * num_envs) // eval_interval < timesteps // eval_interval:
            metrics = evaluate(policy, env_cfg, normalizer, eval_cfg, device, out_dir, timesteps)
            print(
                f"[eval] steps={timesteps} weak={metrics['weak_success_rate']:.3f} "
                f"strict={metrics['strict_success_rate']:.3f} avg_x={metrics['avg_distance']:.3f} "
                f"max_x={metrics['max_distance']:.3f} return={metrics['avg_return']:.1f}",
                flush=True,
            )
            write_header = not eval_csv.exists()
            with open(eval_csv, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timesteps", *metrics.keys()])
                if write_header:
                    writer.writeheader()
                writer.writerow({"timesteps": timesteps, **metrics})
            maybe_log_wandb(
                {
                    "eval/timesteps": timesteps,
                    "eval/weak_success_rate": metrics["weak_success_rate"],
                    "eval/strict_success_rate": metrics["strict_success_rate"],
                    "eval/avg_distance": metrics["avg_distance"],
                    "eval/max_distance": metrics["max_distance"],
                    "eval/avg_forward_velocity": metrics["avg_forward_velocity"],
                    "eval/avg_return": metrics["avg_return"],
                },
                commit=False,
            )
            curriculum_level = int(round(avg_terrain_level)) if recent_infos else None
            reset_options = {"terrain_type": eval_cfg.get("showcase_terrain_type", None)}
            if curriculum_level is not None:
                reset_options["terrain_level"] = curriculum_level
            if bool(eval_cfg.get("record_recent_video", True)):
                recent_video_path = _video_output_path(out_dir, "most_recent_teacher", curriculum_level)
                try:
                    recent_video_metrics = record_video(
                        policy,
                        env_cfg,
                        normalizer,
                        eval_cfg,
                        device,
                        recent_video_path,
                        reset_options=reset_options,
                    )
                    print(
                        f"[recent] steps={timesteps} saved_video={recent_video_path} "
                        f"video_distance={recent_video_metrics['distance']:.2f} "
                        f"video_avg_speed={recent_video_metrics['avg_forward_velocity']:.2f} "
                        f"video_weak_success={recent_video_metrics['weak_success']} "
                        f"video_strict_success={recent_video_metrics['strict_success']}",
                        flush=True,
                    )
                    maybe_log_wandb(
                        {
                            "eval/recent_video_distance": recent_video_metrics["distance"],
                            "eval/recent_video_avg_forward_velocity": recent_video_metrics["avg_forward_velocity"],
                            "eval/recent_video_return": recent_video_metrics["return"],
                            "eval/recent_video_weak_success": float(recent_video_metrics["weak_success"]),
                            "eval/recent_video_strict_success": float(recent_video_metrics["strict_success"]),
                        },
                        commit=False,
                    )
                    maybe_log_wandb_video(
                        "media/most_recent_teacher_video",
                        recent_video_path,
                        fps=int(eval_cfg.get("video_fps", 24)),
                    )
                    maybe_save_wandb(recent_video_path, out_dir)
                except Exception as exc:
                    print(f"[recent] steps={timesteps} video_save_failed={type(exc).__name__}: {exc}", flush=True)
            if metrics["avg_distance"] > best_distance:
                best_distance = metrics["avg_distance"]
                best_checkpoint = out_dir / "best_teacher_bptt.pt"
                save_checkpoint(best_checkpoint, policy, optimizer, normalizer, config, timesteps)
                maybe_save_wandb(best_checkpoint, out_dir)
                if bool(eval_cfg.get("record_best_video", True)):
                    best_video_path = _video_output_path(out_dir, "best_teacher", curriculum_level)
                    try:
                        best_video_metrics = record_video(
                            policy,
                            env_cfg,
                            normalizer,
                            eval_cfg,
                            device,
                            best_video_path,
                            reset_options=reset_options,
                        )
                        print(
                            f"[best] steps={timesteps} saved_checkpoint={best_checkpoint} "
                            f"saved_video={best_video_path} video_distance={best_video_metrics['distance']:.2f} "
                            f"video_avg_speed={best_video_metrics['avg_forward_velocity']:.2f} "
                            f"video_weak_success={best_video_metrics['weak_success']} "
                            f"video_strict_success={best_video_metrics['strict_success']}",
                            flush=True,
                        )
                        maybe_log_wandb(
                            {
                                "eval/best_video_distance": best_video_metrics["distance"],
                                "eval/best_video_avg_forward_velocity": best_video_metrics["avg_forward_velocity"],
                                "eval/best_video_return": best_video_metrics["return"],
                                "eval/best_video_weak_success": float(best_video_metrics["weak_success"]),
                                "eval/best_video_strict_success": float(best_video_metrics["strict_success"]),
                            },
                            commit=False,
                        )
                        maybe_log_wandb_video(
                            "media/best_teacher_video",
                            best_video_path,
                            fps=int(eval_cfg.get("video_fps", 24)),
                        )
                        maybe_save_wandb(best_video_path, out_dir)
                    except Exception as exc:
                        print(f"[best] steps={timesteps} video_save_failed={type(exc).__name__}: {exc}", flush=True)
            maybe_save_wandb(eval_csv, out_dir)
            maybe_log_wandb({}, commit=True)
        else:
            maybe_log_wandb({}, commit=True)

    save_checkpoint(out_dir / "final_teacher_bptt.pt", policy, optimizer, normalizer, config, timesteps)
    maybe_save_wandb(out_dir / "final_teacher_bptt.pt", out_dir)
    maybe_save_wandb(train_csv, out_dir)
    for env in envs:
        env.close()
    if wandb_run is not None:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config)


if __name__ == "__main__":
    main()
