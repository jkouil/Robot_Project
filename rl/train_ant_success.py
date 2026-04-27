from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.normal import Normal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden), nn.Tanh()])
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.policy_mean = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        feat = self.backbone(obs)
        mean = self.policy_mean(feat)
        value = self.value_head(feat).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    def act(self, obs: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_obs: torch.Tensor
    episode_returns: list[float]
    episode_distances: list[float]
    episode_successes: list[bool]


def make_ant_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_ant_render_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_adv = delta + gamma * gae_lambda * not_done * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def minibatch_indices(batch_size: int, minibatch_size: int):
    perm = torch.randperm(batch_size)
    for start in range(0, batch_size, minibatch_size):
        yield perm[start : start + minibatch_size]


def normalize_obs(obs_rms: RunningMeanStd, obs: np.ndarray, update_stats: bool) -> np.ndarray:
    obs_2d = obs[None, :]
    if update_stats:
        obs_rms.update(obs_2d)
    return obs_rms.normalize(obs).astype(np.float32)


def collect_rollout(env, agent, obs_rms, rollout_steps, device, success_distance):
    obs_list, action_list, log_prob_list = [], [], []
    reward_list, done_list, value_list = [], [], []

    episode_returns: list[float] = []
    episode_distances: list[float] = []
    episode_successes: list[bool] = []

    obs, _ = env.reset()
    obs = normalize_obs(obs_rms, obs, update_stats=True)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    current_return = 0.0
    current_best_x = 0.0

    for _ in range(rollout_steps):
        with torch.no_grad():
            action_t, log_prob_t, value_t = agent.act(obs_t.unsqueeze(0))
        action = action_t.squeeze(0).cpu().numpy()
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs_raw, reward, terminated, truncated, info = env.step(clipped_action)
        done = terminated or truncated
        current_return += float(reward)
        current_best_x = max(current_best_x, float(info.get("x_position", 0.0)))

        obs_list.append(obs_t)
        action_list.append(torch.tensor(clipped_action, dtype=torch.float32, device=device))
        log_prob_list.append(log_prob_t.squeeze(0))
        reward_list.append(torch.tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(float(done), dtype=torch.float32, device=device))
        value_list.append(value_t.squeeze(0))

        if done:
            episode_returns.append(current_return)
            episode_distances.append(current_best_x)
            episode_successes.append(current_best_x >= success_distance)
            next_obs_raw, _ = env.reset()
            current_return = 0.0
            current_best_x = 0.0

        next_obs = normalize_obs(obs_rms, next_obs_raw, update_stats=True)
        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

    return RolloutBatch(
        obs=torch.stack(obs_list),
        actions=torch.stack(action_list),
        log_probs=torch.stack(log_prob_list),
        rewards=torch.stack(reward_list),
        dones=torch.stack(done_list),
        values=torch.stack(value_list),
        next_obs=obs_t,
        episode_returns=episode_returns,
        episode_distances=episode_distances,
        episode_successes=episode_successes,
    )


def evaluate_policy(agent, env, obs_rms, eval_episodes, success_distance, device):
    episode_returns = []
    episode_distances = []
    episode_successes = []

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        obs = normalize_obs(obs_rms, obs, update_stats=False)
        done = False
        total_reward = 0.0
        best_x = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = agent.forward(obs_t)
            action = mean.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            best_x = max(best_x, float(info.get("x_position", 0.0)))
            done = terminated or truncated
            obs = normalize_obs(obs_rms, next_obs, update_stats=False)

        episode_returns.append(total_reward)
        episode_distances.append(best_x)
        episode_successes.append(best_x >= success_distance)

    return {
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "avg_distance": float(np.mean(episode_distances)) if episode_distances else 0.0,
        "max_distance": float(np.max(episode_distances)) if episode_distances else 0.0,
        "avg_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
    }


def record_video(env_id: str, seed: int, agent, obs_rms, output_path: Path, success_distance, device):
    env = make_ant_render_env(env_id, seed)
    frames = []
    try:
        obs, _ = env.reset()
        obs = normalize_obs(obs_rms, obs, update_stats=False)
        done = False
        total_reward = 0.0
        best_x = 0.0

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = agent.forward(obs_t)
            action = mean.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            best_x = max(best_x, float(info.get("x_position", 0.0)))
            done = terminated or truncated
            obs = normalize_obs(obs_rms, next_obs, update_stats=False)
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_path, frames, fps=30)
        return {
            "distance": best_x,
            "return": total_reward,
            "success": best_x >= success_distance,
            "num_frames": len(frames),
            "video_saved": True,
        }
    finally:
        env.close()


def save_checkpoint(output_path: Path, agent, obs_rms, config, metrics):
    payload = {
        "model_state_dict": agent.state_dict(),
        "obs_mean": obs_rms.mean,
        "obs_var": obs_rms.var,
        "obs_count": obs_rms.count,
        "config": config,
        "metrics": metrics,
    }
    torch.save(payload, output_path)


def train(config: dict) -> None:
    train_cfg = config["train"]
    env_cfg = config["env"]
    eval_cfg = config["eval"]

    device_name = train_cfg.get("device", "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("requested CUDA but no GPU is available; falling back to CPU", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)

    seed = int(train_cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_ant_env(env_cfg["env_id"], seed)
    eval_env = make_ant_env(env_cfg["env_id"], seed + 1)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    agent = ActorCritic(obs_dim, action_dim, hidden_sizes=list(config["policy"]["hidden_sizes"])).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(train_cfg["learning_rate"]))

    total_updates = int(train_cfg["total_updates"])
    rollout_steps = int(train_cfg["rollout_steps"])
    ppo_epochs = int(train_cfg["ppo_epochs"])
    minibatch_size = int(train_cfg["minibatch_size"])
    gamma = float(train_cfg["gamma"])
    gae_lambda = float(train_cfg["gae_lambda"])
    clip_coef = float(train_cfg["clip_coef"])
    value_coef = float(train_cfg["value_coef"])
    entropy_coef = float(train_cfg["entropy_coef"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    success_distance = float(eval_cfg["success_distance"])
    eval_interval = int(eval_cfg["eval_interval_updates"])
    eval_episodes = int(eval_cfg["eval_episodes"])

    out_dir = Path(train_cfg.get("output_dir", "artifacts/ant_success"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_success_rate = -1.0
    best_avg_distance = -1.0

    for update in range(1, total_updates + 1):
        batch = collect_rollout(env, agent, obs_rms, rollout_steps, device, success_distance)
        with torch.no_grad():
            _, _, next_value = agent.forward(batch.next_obs.unsqueeze(0))
            next_value = next_value.squeeze(0)

        advantages, returns = compute_gae(
            batch.rewards, batch.dones, batch.values, next_value, gamma=gamma, gae_lambda=gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = batch.obs.shape[0]
        for _ in range(ppo_epochs):
            for indices in minibatch_indices(batch_size, minibatch_size):
                obs_mb = batch.obs[indices]
                actions_mb = batch.actions[indices]
                old_log_probs_mb = batch.log_probs[indices]
                advantages_mb = advantages[indices]
                returns_mb = returns[indices]
                old_values_mb = batch.values[indices]

                new_log_probs, entropy, new_values = agent.evaluate_actions(obs_mb, actions_mb)
                log_ratio = new_log_probs - old_log_probs_mb
                ratio = log_ratio.exp()

                unclipped = ratio * advantages_mb
                clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages_mb
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred_clipped = old_values_mb + torch.clamp(new_values - old_values_mb, -clip_coef, clip_coef)
                value_loss_unclipped = torch.square(new_values - returns_mb)
                value_loss_clipped = torch.square(value_pred_clipped - returns_mb)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        train_success_rate = float(np.mean(batch.episode_successes)) if batch.episode_successes else 0.0
        train_avg_distance = float(np.mean(batch.episode_distances)) if batch.episode_distances else 0.0
        train_max_distance = float(np.max(batch.episode_distances)) if batch.episode_distances else 0.0
        train_avg_return = float(np.mean(batch.episode_returns)) if batch.episode_returns else float(batch.rewards.sum().cpu().item())
        print(
            f"update={update:05d} train_success_rate={train_success_rate:.3f} "
            f"train_avg_distance={train_avg_distance:.2f} train_max_distance={train_max_distance:.2f} "
            f"train_avg_return={train_avg_return:.2f} action_std={math.exp(agent.log_std.mean().item()):.3f}",
            flush=True,
        )

        if update % eval_interval != 0 and update != total_updates:
            continue

        metrics = evaluate_policy(agent, eval_env, obs_rms, eval_episodes, success_distance, device)
        print(
            f"[eval] update={update:05d} success_rate={metrics['success_rate']:.3f} "
            f"avg_distance={metrics['avg_distance']:.2f} max_distance={metrics['max_distance']:.2f} "
            f"avg_return={metrics['avg_return']:.2f}",
            flush=True,
        )

        improved = False
        if metrics["success_rate"] > best_success_rate:
            improved = True
        elif metrics["success_rate"] == best_success_rate and metrics["avg_distance"] > best_avg_distance:
            improved = True

        if improved:
            best_success_rate = metrics["success_rate"]
            best_avg_distance = metrics["avg_distance"]

            model_path = out_dir / "best_model.pt"
            save_checkpoint(model_path, agent, obs_rms, config, metrics)

            video_path = out_dir / "best_success.mp4"
            try:
                video_metrics = record_video(
                    env_cfg["env_id"], seed + update + 1000, agent, obs_rms, video_path, success_distance, device
                )
                print(
                    f"[best] update={update:05d} saved_model={model_path} saved_video={video_path} "
                    f"video_distance={video_metrics['distance']:.2f} video_success={video_metrics['success']}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[best] update={update:05d} saved_model={model_path} video_save_failed={type(exc).__name__}: {exc}",
                    flush=True,
                )

    env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "ant_success.yaml",
    )
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
