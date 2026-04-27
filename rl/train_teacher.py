from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

def _preparse_flag(flag: str) -> str | None:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


requested_mujoco_gl = _preparse_flag("--mujoco-gl")
if requested_mujoco_gl is not None:
    os.environ["MUJOCO_GL"] = requested_mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = requested_mujoco_gl
else:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import torch
import yaml
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

# Work around a broken tensorboard/tensorflow combination in the current env.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

from rl.env import make_env
from rl.features import TeacherDictFeaturesExtractor
from rl.gru_policy import ScandotGruPolicy
from rl.paper_bptt import PaperBpttRecurrentPPO


def apply_egl_device_config(config: dict) -> None:
    if os.environ.get("MUJOCO_GL") != "egl":
        return
    eval_cfg = config.get("eval", {})
    egl_device_id = eval_cfg.get("egl_device_id")
    if egl_device_id is None:
        return
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device_id))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    norm_obs_keys = train_cfg.get("normalize_obs_keys")
    if norm_obs_keys is not None:
        train_cfg["normalize_obs_keys"] = ["scandots" if key == "terrain" else key for key in norm_obs_keys]
    _apply_truncated_bptt_config(train_cfg)
    if "eval_interval_timesteps" not in eval_cfg and "eval_interval_updates" in eval_cfg:
        rollout_steps = int(train_cfg.get("rollout_steps", 1))
        num_envs = int(train_cfg.get("num_envs", 1))
        eval_cfg["eval_interval_timesteps"] = rollout_steps * num_envs * int(eval_cfg["eval_interval_updates"])
    apply_egl_device_config(config)
    return config


def _apply_truncated_bptt_config(train_cfg: dict) -> None:
    truncated_steps = int(train_cfg.get("truncated_bptt_steps", 0) or 0)
    if truncated_steps <= 0:
        return
    if not bool(train_cfg.get("recurrent", False)):
        raise ValueError("truncated_bptt_steps requires train.recurrent: true")
    if truncated_steps < 2:
        raise ValueError("truncated_bptt_steps must be at least 2")

    train_cfg["effective_bptt_steps"] = truncated_steps


def _sanitize_config_for_logging(config: dict) -> dict:
    def _convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
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
        wandb.define_metric("curriculum/timesteps")
        wandb.define_metric("eval/*", step_metric="eval/timesteps")
        wandb.define_metric("curriculum/*", step_metric="curriculum/timesteps")
    return run


def maybe_log_wandb(data: dict, commit: bool = True) -> None:
    if wandb.run is not None:
        wandb.log(data, commit=commit)


def maybe_log_wandb_video(key: str, video_path: Path, fps: int) -> None:
    if wandb.run is None or not video_path.exists():
        return
    wandb.log({key: wandb.Video(str(video_path), fps=fps, format="mp4")}, commit=False)


def _overlay_video_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(int(bbox[2] - bbox[0]))
        line_heights.append(int(bbox[3] - bbox[1]))
    padding = 8
    spacing = 4
    box_width = max(line_widths, default=0) + 2 * padding
    box_height = sum(line_heights) + max(0, len(lines) - 1) * spacing + 2 * padding
    draw.rounded_rectangle(
        (10, 10, 10 + box_width, 10 + box_height),
        radius=8,
        fill=(0, 0, 0, 155),
    )
    y = 10 + padding
    for line, line_height in zip(lines, line_heights):
        draw.text((10 + padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height + spacing
    return np.asarray(image)


def _render_video_frame_with_overlay(
    video_env,
    eval_cfg: dict,
    *,
    level: int | None,
    step_count: int,
    step_reward: float,
    episode_return: float,
    distance: float,
    avg_speed: float,
    success: bool,
    success_target_x: float,
    success_progress: float,
) -> np.ndarray:
    frame = video_env.render_frame(
        width=int(eval_cfg["video_width"]),
        height=int(eval_cfg["video_height"]),
        camera=eval_cfg["video_camera"],
    )
    if not bool(eval_cfg.get("video_overlay_metrics", True)):
        return frame
    terrain_slope_deg = 0.0
    if hasattr(video_env, "get_metrics"):
        try:
            terrain_slope_deg = float(video_env.get_metrics().get("terrain_slope_deg", 0.0))
        except Exception:
            terrain_slope_deg = 0.0
    lines = [
        f"level: {level if level is not None else -1}",
        f"success: {'yes' if success else 'no'}  target_x: {success_target_x:.2f} m",
        f"step_reward: {step_reward:.3f}  return: {episode_return:.2f}",
        f"distance: {distance:.2f} m  progress: {success_progress:.2f}  avg_speed: {avg_speed:.2f} m/s",
        f"terrain_slope: {terrain_slope_deg:+.1f} deg",
        f"step: {step_count}",
    ]
    return _overlay_video_text(frame, lines)


def is_teacher_strict_success(info: dict, terminated: bool, eval_cfg: dict) -> bool:
    _ = eval_cfg
    return bool(info.get("episode_success", 0.0)) and not terminated


def is_teacher_weak_success(info: dict, terminated: bool, eval_cfg: dict) -> bool:
    _ = eval_cfg
    return bool(info.get("episode_success", 0.0)) and not terminated


def is_recurrent_config(config: dict) -> bool:
    return bool(config["train"].get("recurrent", False))


def build_policy_kwargs(config: dict) -> dict:
    hidden_sizes = list(config["policy"]["hidden_sizes"])
    policy_cfg = config["policy"]
    policy_kwargs = {
        "net_arch": {"pi": hidden_sizes, "vf": hidden_sizes},
        "log_std_init": float(policy_cfg.get("init_log_std", -1.0)),
    }
    if config["env"].get("observation_mode") == "teacher_dict":
        encoder_variant = str(policy_cfg.get("teacher_encoder_variant", "current")).lower()
        include_privileged = bool(policy_cfg.get("include_privileged", True))
        policy_kwargs["features_extractor_class"] = TeacherDictFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "proprio_hidden_dim": int(policy_cfg.get("proprio_hidden_dim", 128)),
            "scandot_hidden_dim": int(policy_cfg.get("scandot_hidden_dim", policy_cfg.get("terrain_hidden_dim", 128))),
            "command_hidden_dim": int(policy_cfg.get("command_hidden_dim", 32)),
            "privileged_hidden_dim": int(policy_cfg.get("privileged_hidden_dim", 32)),
            "fused_dim": int(policy_cfg.get("fused_dim", 256)),
            "encoder_variant": encoder_variant,
            "include_privileged": include_privileged,
        }
    if is_recurrent_config(config):
        policy_kwargs["lstm_hidden_size"] = int(policy_cfg.get("lstm_hidden_size", 256))
        policy_kwargs["n_lstm_layers"] = int(policy_cfg.get("n_lstm_layers", 1))
        policy_kwargs["shared_lstm"] = bool(policy_cfg.get("shared_lstm", False))
        policy_kwargs["enable_critic_lstm"] = bool(policy_cfg.get("enable_critic_lstm", True))
    return policy_kwargs


def make_single_env(config_dict: dict, monitor_dir: str | None = None):
    def _thunk():
        env = make_env(config_dict)
        return Monitor(
            env,
            filename=monitor_dir,
            info_keywords=("x_position", "forward_velocity", "terrain_level"),
        )

    return _thunk


def build_train_vec_env(config: dict, out_dir: Path):
    train_cfg = config["train"]
    num_envs = int(train_cfg.get("num_envs", 1))
    seed = int(train_cfg.get("seed", 0))
    env_fns = []
    for env_idx in range(num_envs):
        env_cfg = {
            **config["env"],
            "seed": int(config["env"].get("seed", seed)) + env_idx,
        }
        monitor_path = str(out_dir / f"train_monitor_env{env_idx}.csv") if num_envs > 1 else str(out_dir / "train_monitor.csv")
        env_fns.append(make_single_env(env_cfg, monitor_dir=monitor_path))
    if num_envs > 1:
        start_method = str(train_cfg.get("subproc_start_method", "fork"))
        return SubprocVecEnv(env_fns, start_method=start_method)
    return DummyVecEnv(env_fns)


def build_vec_normalize(venv, config: dict, training: bool) -> VecNormalize:
    train_cfg = config["train"]
    kwargs = {
        "training": training,
        "norm_obs": bool(train_cfg.get("normalize_observations", True)),
        "norm_reward": bool(train_cfg.get("normalize_rewards", False)),
        "clip_obs": float(train_cfg.get("clip_observations", 10.0)),
        "clip_reward": float(train_cfg.get("clip_rewards", 10.0)),
        "gamma": float(train_cfg["gamma"]),
    }
    norm_obs_keys = train_cfg.get("normalize_obs_keys")
    if norm_obs_keys is not None and isinstance(venv.observation_space, spaces.Dict):
        kwargs["norm_obs_keys"] = list(norm_obs_keys)
    return VecNormalize(venv, **kwargs)


def _resolve_optional_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def maybe_load_vec_normalize(venv, config: dict, training: bool) -> VecNormalize:
    train_cfg = config["train"]
    vecnorm_path = _resolve_optional_path(train_cfg.get("finetune_vecnormalize_path"))
    if vecnorm_path is None:
        return build_vec_normalize(venv, config, training=training)
    vec_env = VecNormalize.load(str(vecnorm_path), venv)
    vec_env.training = training
    vec_env.norm_reward = bool(train_cfg.get("normalize_rewards", False)) if training else False
    print(f"[finetune] loaded_vecnormalize={vecnorm_path} training={training}", flush=True)
    return vec_env


def build_model(config: dict, train_env: VecNormalize, out_dir: Path, device_name: str, seed: int):
    train_cfg = config["train"]
    rollout_steps = int(train_cfg["rollout_steps"])
    policy_kwargs = build_policy_kwargs(config)
    truncated_bptt_steps = int(train_cfg.get("truncated_bptt_steps", 0) or 0)

    if is_recurrent_config(config):
        recurrent_cell = str(config["train"].get("recurrent_cell", "lstm")).lower()
        if config["env"].get("observation_mode") == "teacher_dict":
            policy_name = ScandotGruPolicy if recurrent_cell == "gru" else "MultiInputLstmPolicy"
        else:
            policy_name = "MlpLstmPolicy"
        recurrent_ppo_cls = PaperBpttRecurrentPPO if truncated_bptt_steps > 0 else RecurrentPPO
        recurrent_ppo_kwargs = {}
        if truncated_bptt_steps > 0:
            recurrent_ppo_kwargs["recurrent_sequence_length"] = truncated_bptt_steps
        return recurrent_ppo_cls(
            policy=policy_name,
            env=train_env,
            learning_rate=float(train_cfg["learning_rate"]),
            n_steps=rollout_steps,
            batch_size=int(train_cfg["minibatch_size"]),
            n_epochs=int(train_cfg["ppo_epochs"]),
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            clip_range=float(train_cfg["clip_coef"]),
            ent_coef=float(train_cfg["entropy_coef"]),
            vf_coef=float(train_cfg["value_coef"]),
            max_grad_norm=float(train_cfg["max_grad_norm"]),
            target_kl=None if train_cfg.get("target_kl") is None else float(train_cfg["target_kl"]),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device_name,
            seed=seed,
            tensorboard_log=str(out_dir / "tb") if bool(train_cfg.get("tensorboard", False)) else None,
            **recurrent_ppo_kwargs,
        )

    policy_name = "MultiInputPolicy" if config["env"].get("observation_mode") == "teacher_dict" else "MlpPolicy"
    return PPO(
        policy=policy_name,
        env=train_env,
        learning_rate=float(train_cfg["learning_rate"]),
        n_steps=rollout_steps,
        batch_size=int(train_cfg["minibatch_size"]),
        n_epochs=int(train_cfg["ppo_epochs"]),
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
        clip_range=float(train_cfg["clip_coef"]),
        ent_coef=float(train_cfg["entropy_coef"]),
        vf_coef=float(train_cfg["value_coef"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        target_kl=None if train_cfg.get("target_kl") is None else float(train_cfg["target_kl"]),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device_name,
        seed=seed,
        tensorboard_log=str(out_dir / "tb") if bool(train_cfg.get("tensorboard", False)) else None,
    )


def maybe_load_finetune_parameters(model, config: dict, device_name: str) -> None:
    model_path = _resolve_optional_path(config["train"].get("finetune_model_path"))
    if model_path is None:
        return
    _, params, _ = load_from_zip_file(
        str(model_path),
        device=device_name,
        print_system_info=False,
    )
    if params is None:
        raise FileNotFoundError(f"Could not load parameters from {model_path}")
    model.set_parameters(params, exact_match=True, device=device_name)
    print(f"[finetune] loaded_model_parameters={model_path}", flush=True)


def _predict_with_optional_state(model, obs, recurrent: bool, lstm_states, episode_starts):
    if recurrent:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        return action, lstm_states
    action, _ = model.predict(obs, deterministic=True)
    return action, lstm_states


def _unwrap_base_env(env) -> object:
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def _get_train_curriculum_levels(training_env) -> list[int]:
    vec_env = training_env
    while hasattr(vec_env, "venv"):
        vec_env = vec_env.venv
    if hasattr(vec_env, "env_method"):
        try:
            levels = vec_env.env_method("get_curriculum_level")
            return [int(level) for level in levels]
        except Exception:
            pass
    if hasattr(vec_env, "get_attr"):
        try:
            levels = vec_env.get_attr("curriculum_level")
            return [int(level) for level in levels]
        except Exception:
            pass
    if not hasattr(vec_env, "envs") or not vec_env.envs:
        return []
    levels = []
    for env in vec_env.envs:
        base_env = _unwrap_base_env(env)
        if hasattr(base_env, "get_curriculum_level"):
            levels.append(int(base_env.get_curriculum_level()))
    return levels


def _set_eval_curriculum_level(eval_env, level: int | None) -> None:
    if level is None:
        return
    vec_env = eval_env
    while hasattr(vec_env, "venv"):
        vec_env = vec_env.venv
    if hasattr(vec_env, "env_method"):
        try:
            vec_env.env_method("set_curriculum_level", level)
            return
        except Exception:
            pass
    if hasattr(vec_env, "set_attr"):
        try:
            vec_env.set_attr("curriculum_level", level)
            return
        except Exception:
            pass
    if not hasattr(vec_env, "envs"):
        return
    for env in vec_env.envs:
        base_env = _unwrap_base_env(env)
        if hasattr(base_env, "set_curriculum_level"):
            base_env.set_curriculum_level(level)


def _video_output_path(out_dir: Path, prefix: str, curriculum_level: int | None) -> Path:
    if curriculum_level is None:
        return out_dir / f"{prefix}.mp4"
    return out_dir / f"{prefix}_level_{curriculum_level}.mp4"


def _save_static_scene_assets(env, output_dir: Path, prefix: str, level: int, eval_cfg: dict, frame: np.ndarray) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{prefix}_level_{level}.png"
    video_path = output_dir / f"{prefix}_level_{level}.mp4"
    import imageio.v2 as imageio

    imageio.imwrite(image_path, frame)
    frames = [frame.copy() for _ in range(max(1, int(eval_cfg.get("video_fps", 24))))]
    env.save_video(frames, video_path, fps=int(eval_cfg.get("video_fps", 24)))
    return image_path, video_path


def generate_pretrain_scene_showcase(config: dict, out_dir: Path) -> None:
    eval_cfg = config["eval"]
    if not bool(eval_cfg.get("record_pretrain_scene_showcase", True)):
        return
    env_cfg = {
        **config["env"],
        "seed": int(config["train"].get("seed", 0)) + 999,
        "reset_noise_scale": 0.0,
    }
    preview_env = make_env(env_cfg)
    preview_env.set_preview_highlight(bool(eval_cfg.get("highlight_preview_terrain", False)))
    preview_env.set_scandot_overlay(bool(eval_cfg.get("render_scandots", True)))
    preview_env.set_stair_edge_overlay(bool(eval_cfg.get("render_stair_edges", True)))
    pretrain_dir = out_dir / "pretrain_scene_showcase"
    train_dir = pretrain_dir / "train"
    eval_dir = pretrain_dir / "eval"
    min_level = int(config["env"].get("curriculum", {}).get("min_level", 0))
    max_level = int(eval_cfg.get("showcase_max_level", config["env"].get("curriculum", {}).get("max_level", 5)))
    terrain_type = str(eval_cfg.get("showcase_terrain_type", "stairs"))
    try:
        for level in range(min_level, max_level + 1):
            preview_env.reset(options={"terrain_type": terrain_type, "terrain_level": level})
            try:
                frame = preview_env.render_frame(
                    width=int(eval_cfg["video_width"]),
                    height=int(eval_cfg["video_height"]),
                    camera=eval_cfg["video_camera"],
                )
            except Exception as exc:
                print(
                    f"[pretrain-scene] skipped due to render init failure: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return
            train_image_path, train_video_path = _save_static_scene_assets(
                preview_env,
                train_dir,
                terrain_type,
                level,
                eval_cfg,
                frame,
            )
            eval_image_path = eval_dir / train_image_path.name
            eval_video_path = eval_dir / train_video_path.name
            eval_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(train_image_path, eval_image_path)
            shutil.copy2(train_video_path, eval_video_path)
            print(
                f"[pretrain-scene] terrain={terrain_type} level={level} "
                f"train_image={train_image_path} train_video={train_video_path} "
                f"eval_image={eval_image_path} eval_video={eval_video_path}",
                flush=True,
            )
    finally:
        preview_env.close()


def evaluate_model(model, eval_env: VecNormalize, eval_cfg: dict, recurrent: bool) -> dict:
    episode_returns = []
    episode_distances = []
    episode_avg_speeds = []
    episode_weak_successes = []
    episode_strict_successes = []
    fall_count = 0
    eval_episodes = int(eval_cfg["eval_episodes"])
    monitor_env = eval_env.venv.envs[0]
    base_env = monitor_env.env

    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        terminated = False
        last_info = {"x_position": 0.0}
        lstm_states = None
        episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

        while not done:
            action, lstm_states = _predict_with_optional_state(model, obs, recurrent, lstm_states, episode_starts)
            obs, reward, dones, infos = eval_env.step(action)
            total_reward += float(reward[0])
            step_count += 1
            done = bool(dones[0])
            last_info = infos[0]
            terminated = done and not bool(last_info.get("TimeLimit.truncated", False))
            episode_starts = dones

        distance = float(last_info["x_position"])
        avg_speed = distance / max(step_count * base_env.config.control_dt, 1e-6)
        weak_success = is_teacher_weak_success(last_info, terminated, eval_cfg)
        strict_success = is_teacher_strict_success(last_info, terminated, eval_cfg)

        episode_returns.append(total_reward)
        episode_distances.append(distance)
        episode_avg_speeds.append(avg_speed)
        episode_weak_successes.append(weak_success)
        episode_strict_successes.append(strict_success)
        fall_count += int(terminated)

    return {
        "weak_success_rate": float(np.mean(episode_weak_successes)) if episode_weak_successes else 0.0,
        "strict_success_rate": float(np.mean(episode_strict_successes)) if episode_strict_successes else 0.0,
        "avg_distance": float(np.mean(episode_distances)) if episode_distances else 0.0,
        "max_distance": float(np.max(episode_distances)) if episode_distances else 0.0,
        "avg_forward_velocity": float(np.mean(episode_avg_speeds)) if episode_avg_speeds else 0.0,
        "avg_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "fall_rate": float(fall_count / max(len(episode_strict_successes), 1)),
    }


def record_video(
    model,
    video_env,
    obs_normalizer: VecNormalize,
    output_path: Path,
    eval_cfg: dict,
    recurrent: bool,
    reset_options: dict | None = None,
) -> dict:
    obs, _ = video_env.reset(options=reset_options)
    frames = []
    done = False
    total_reward = 0.0
    step_count = 0
    terminated = False
    last_info = {"x_position": 0.0}
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    max_video_steps = eval_cfg.get("video_max_steps")
    max_video_steps = None if max_video_steps is None else int(max_video_steps)

    highlight_preview = bool(eval_cfg.get("highlight_preview_terrain", True))
    video_env.set_preview_highlight(highlight_preview)
    video_env.set_scandot_overlay(bool(eval_cfg.get("render_scandots", True)))
    video_env.set_stair_edge_overlay(bool(eval_cfg.get("render_stair_edges", True)))
    curriculum_level = None
    if reset_options is not None and "terrain_level" in reset_options:
        curriculum_level = int(reset_options["terrain_level"])
    elif hasattr(video_env, "get_curriculum_level"):
        curriculum_level = int(video_env.get_curriculum_level())
    try:
        frames.append(
            _render_video_frame_with_overlay(
                video_env,
                eval_cfg,
                level=curriculum_level,
                step_count=0,
                step_reward=0.0,
                episode_return=0.0,
                distance=0.0,
                avg_speed=0.0,
                success=False,
                success_target_x=0.0,
                success_progress=0.0,
            )
        )
        while not done:
            normalized_obs = obs_normalizer.normalize_obs(obs.copy() if isinstance(obs, dict) else obs.copy())
            action, lstm_states = _predict_with_optional_state(
                model,
                normalized_obs,
                recurrent,
                lstm_states,
                episode_starts,
            )
            obs, reward, terminated, truncated, last_info = video_env.step(action)
            step_reward = float(reward)
            total_reward += step_reward
            step_count += 1
            done = terminated or truncated
            if max_video_steps is not None and step_count >= max_video_steps:
                done = True
                truncated = True
            episode_starts = np.array([done], dtype=bool)
            distance = float(last_info.get("x_position", 0.0))
            avg_speed = distance / max(step_count * video_env.config.control_dt, 1e-6)
            frames.append(
                _render_video_frame_with_overlay(
                    video_env,
                    eval_cfg,
                    level=curriculum_level,
                    step_count=step_count,
                    step_reward=step_reward,
                    episode_return=total_reward,
                    distance=distance,
                    avg_speed=avg_speed,
                    success=bool(last_info.get("episode_success", 0.0)) and not bool(terminated),
                    success_target_x=float(last_info.get("success_target_x", 0.0)),
                    success_progress=float(last_info.get("success_progress", 0.0)),
                )
            )
    finally:
        video_env.set_preview_highlight(False)
        video_env.set_scandot_overlay(False)
        video_env.set_stair_edge_overlay(False)

    video_env.save_video(frames, output_path, fps=int(eval_cfg["video_fps"]))
    distance = float(last_info["x_position"])
    avg_speed = distance / max(step_count * video_env.config.control_dt, 1e-6)
    return {
        "distance": distance,
        "avg_forward_velocity": avg_speed,
        "return": total_reward,
        "weak_success": is_teacher_weak_success(last_info, terminated, eval_cfg),
        "strict_success": is_teacher_strict_success(last_info, terminated, eval_cfg),
    }


def record_curriculum_showcase(
    model,
    video_env,
    obs_normalizer: VecNormalize,
    out_dir: Path,
    eval_cfg: dict,
    recurrent: bool,
    prefix: str = "best_teacher",
) -> list[dict]:
    if not hasattr(video_env, "get_curriculum_level"):
        return []
    max_level = int(eval_cfg.get("showcase_max_level", getattr(video_env, "curriculum_max_level", video_env.get_curriculum_level())))
    terrain_type = eval_cfg.get("showcase_terrain_type")
    success_metric = str(eval_cfg.get("showcase_success_metric", "weak"))
    results = []
    for level in range(0, max_level + 1):
        video_path = out_dir / f"{prefix}_level_{level}.mp4"
        metrics = record_video(
            model,
            video_env,
            obs_normalizer,
            video_path,
            eval_cfg,
            recurrent=recurrent,
            reset_options={
                "terrain_level": level,
                "terrain_type": terrain_type,
            },
        )
        metrics["level"] = level
        metrics["video_path"] = str(video_path)
        results.append(metrics)
        success = metrics["strict_success"] if success_metric == "strict" else metrics["weak_success"]
        if bool(eval_cfg.get("showcase_stop_on_failure", False)) and not success:
            break
    video_env.set_fixed_terrain_type(None)
    return results


class TeacherEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecNormalize,
        video_env,
        out_dir: Path,
        eval_cfg: dict,
        eval_freq_timesteps: int,
        recurrent: bool,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_env = video_env
        self.out_dir = out_dir
        self.eval_cfg = eval_cfg
        self.eval_freq_timesteps = max(1, int(eval_freq_timesteps))
        self.recurrent = recurrent
        self.best_strict_success_rate = float(eval_cfg.get("initial_best_strict_success_rate", -1.0))
        self.best_weak_success_rate = float(eval_cfg.get("initial_best_weak_success_rate", -1.0))
        self.best_avg_distance = float(eval_cfg.get("initial_best_avg_distance", -1.0))
        self.best_fall_rate = float(eval_cfg.get("initial_best_fall_rate", float("inf")))
        self.best_no_fall_distance = float(eval_cfg.get("initial_best_no_fall_distance", -1.0))
        self.no_improvement_eval_count = 0
        self.eval_count = 0
        self.eval_csv_path = self.out_dir / "eval_metrics.csv"
        self._csv_initialized = False
        self._last_reported_curriculum_level: int | None = None
        self._last_eval_timestep = 0

    def _on_rollout_end(self) -> None:
        logger_values = getattr(self.model.logger, "name_to_value", {})
        if not logger_values:
            return
        payload = {"train/timesteps": self.num_timesteps}
        for key, value in logger_values.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                payload[f"train/{key.replace('/', '_')}"] = float(value)
        curriculum_levels = _get_train_curriculum_levels(self.training_env)
        if curriculum_levels:
            payload["curriculum/timesteps"] = self.num_timesteps
            payload["curriculum/mean_level"] = float(np.mean(curriculum_levels))
            payload["curriculum/max_level"] = float(np.max(curriculum_levels))
            payload["curriculum/min_level"] = float(np.min(curriculum_levels))
            current_level = int(round(float(np.mean(curriculum_levels))))
            if current_level != self._last_reported_curriculum_level:
                print(
                    f"[curriculum] step={self.num_timesteps:08d} level={current_level} "
                    f"min_level={int(np.min(curriculum_levels))} max_level={int(np.max(curriculum_levels))}",
                    flush=True,
                )
                self._last_reported_curriculum_level = current_level
        maybe_log_wandb(payload)

    def _append_eval_csv(self, metrics: dict) -> None:
        fieldnames = [
            "timesteps",
            "weak_success_rate",
            "strict_success_rate",
            "avg_distance",
            "max_distance",
            "avg_forward_velocity",
            "avg_return",
            "fall_rate",
            "best_weak_success_rate",
            "best_strict_success_rate",
            "best_avg_distance",
            "best_fall_rate",
            "best_no_fall_distance",
            "no_improvement_eval_count",
            "eval_count",
        ]
        if not self._csv_initialized:
            with open(self.eval_csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            self._csv_initialized = True
        row = {
            "timesteps": self.num_timesteps,
            "weak_success_rate": metrics["weak_success_rate"],
            "strict_success_rate": metrics["strict_success_rate"],
            "avg_distance": metrics["avg_distance"],
            "max_distance": metrics["max_distance"],
            "avg_forward_velocity": metrics["avg_forward_velocity"],
            "avg_return": metrics["avg_return"],
            "fall_rate": metrics["fall_rate"],
            "best_weak_success_rate": self.best_weak_success_rate,
            "best_strict_success_rate": self.best_strict_success_rate,
            "best_avg_distance": self.best_avg_distance,
            "best_fall_rate": self.best_fall_rate,
            "best_no_fall_distance": self.best_no_fall_distance,
            "no_improvement_eval_count": self.no_improvement_eval_count,
            "eval_count": self.eval_count,
        }
        with open(self.eval_csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)

    def _is_eval_improved(self, metrics: dict) -> bool:
        mode = str(self.eval_cfg.get("best_metric_mode", "success_then_distance")).lower()
        min_distance_delta = float(self.eval_cfg.get("early_stop_min_distance_delta", 1e-3))
        fall_rate_epsilon = float(self.eval_cfg.get("best_fall_rate_epsilon", 0.05))

        if mode == "fall_rate_then_distance":
            fall_rate = float(metrics["fall_rate"])
            avg_distance = float(metrics["avg_distance"])
            if fall_rate < self.best_fall_rate - fall_rate_epsilon:
                return True
            if abs(fall_rate - self.best_fall_rate) <= fall_rate_epsilon:
                return avg_distance > self.best_avg_distance + min_distance_delta
            return False

        if metrics["strict_success_rate"] > self.best_strict_success_rate:
            return True
        if metrics["strict_success_rate"] == self.best_strict_success_rate:
            if metrics["weak_success_rate"] > self.best_weak_success_rate:
                return True
            if metrics["weak_success_rate"] == self.best_weak_success_rate:
                return metrics["avg_distance"] > self.best_avg_distance
        return False

    def _record_best_metrics(self, metrics: dict) -> None:
        self.best_strict_success_rate = metrics["strict_success_rate"]
        self.best_weak_success_rate = metrics["weak_success_rate"]
        self.best_avg_distance = metrics["avg_distance"]
        self.best_fall_rate = metrics["fall_rate"]
        if metrics["avg_distance"] > self.best_no_fall_distance:
            self.best_no_fall_distance = metrics["avg_distance"]

    @staticmethod
    def _combine_eval_metrics(primary: dict, primary_episodes: int, extra: dict, extra_episodes: int) -> dict:
        total_episodes = max(1, int(primary_episodes) + int(extra_episodes))
        primary_weight = float(primary_episodes) / float(total_episodes)
        extra_weight = float(extra_episodes) / float(total_episodes)
        combined = dict(primary)
        for key in (
            "weak_success_rate",
            "strict_success_rate",
            "avg_distance",
            "avg_forward_velocity",
            "avg_return",
            "fall_rate",
        ):
            combined[key] = float(primary[key]) * primary_weight + float(extra[key]) * extra_weight
        combined["max_distance"] = max(float(primary["max_distance"]), float(extra["max_distance"]))
        return combined

    def _selection_metrics_for_best(self, metrics: dict) -> dict:
        mode = str(self.eval_cfg.get("best_metric_mode", "success_then_distance")).lower()
        if mode != "fall_rate_then_distance":
            return metrics

        trigger_threshold = float(self.eval_cfg.get("candidate_eval_trigger_fall_rate_below", 1.0))
        if float(metrics["fall_rate"]) >= trigger_threshold:
            return metrics

        base_eval_episodes = int(self.eval_cfg["eval_episodes"])
        additional_eval_episodes = int(self.eval_cfg.get("candidate_eval_additional_episodes", 0) or 0)
        if additional_eval_episodes > 0:
            candidate_eval_cfg = dict(self.eval_cfg)
            candidate_eval_cfg["eval_episodes"] = additional_eval_episodes
            extra_metrics = evaluate_model(self.model, self.eval_env, candidate_eval_cfg, recurrent=self.recurrent)
            total_candidate_episodes = base_eval_episodes + additional_eval_episodes
            candidate_metrics = self._combine_eval_metrics(
                metrics,
                base_eval_episodes,
                extra_metrics,
                additional_eval_episodes,
            )
            candidate_metrics["candidate_eval_episodes"] = float(total_candidate_episodes)
            candidate_metrics["candidate_eval_additional_episodes"] = float(additional_eval_episodes)
            candidate_metrics["screening_fall_rate"] = metrics["fall_rate"]
            candidate_metrics["screening_avg_distance"] = metrics["avg_distance"]
            print(
                f"[candidate-eval] step={self.num_timesteps:08d} "
                f"screening_fall_rate={metrics['fall_rate']:.3f} screening_avg_distance={metrics['avg_distance']:.2f} "
                f"additional_episodes={additional_eval_episodes} total_episodes={total_candidate_episodes} "
                f"fall_rate={candidate_metrics['fall_rate']:.3f} "
                f"avg_distance={candidate_metrics['avg_distance']:.2f} max_distance={candidate_metrics['max_distance']:.2f}",
                flush=True,
            )
            maybe_log_wandb(
                {
                    "candidate_eval/episodes": float(total_candidate_episodes),
                    "candidate_eval/additional_episodes": float(additional_eval_episodes),
                    "candidate_eval/screening_fall_rate": metrics["fall_rate"],
                    "candidate_eval/screening_avg_distance": metrics["avg_distance"],
                    "candidate_eval/fall_rate": candidate_metrics["fall_rate"],
                    "candidate_eval/avg_distance": candidate_metrics["avg_distance"],
                    "candidate_eval/max_distance": candidate_metrics["max_distance"],
                    "candidate_eval/avg_forward_velocity": candidate_metrics["avg_forward_velocity"],
                    "candidate_eval/avg_return": candidate_metrics["avg_return"],
                },
                commit=False,
            )
            return candidate_metrics

        candidate_eval_episodes = int(self.eval_cfg.get("candidate_eval_episodes", base_eval_episodes))
        if candidate_eval_episodes <= base_eval_episodes:
            return metrics

        candidate_eval_cfg = dict(self.eval_cfg)
        candidate_eval_cfg["eval_episodes"] = candidate_eval_episodes
        candidate_metrics = evaluate_model(self.model, self.eval_env, candidate_eval_cfg, recurrent=self.recurrent)
        candidate_metrics["candidate_eval_episodes"] = float(candidate_eval_episodes)
        candidate_metrics["screening_fall_rate"] = metrics["fall_rate"]
        candidate_metrics["screening_avg_distance"] = metrics["avg_distance"]
        print(
            f"[candidate-eval] step={self.num_timesteps:08d} "
            f"screening_fall_rate={metrics['fall_rate']:.3f} screening_avg_distance={metrics['avg_distance']:.2f} "
            f"episodes={candidate_eval_episodes} fall_rate={candidate_metrics['fall_rate']:.3f} "
            f"avg_distance={candidate_metrics['avg_distance']:.2f} max_distance={candidate_metrics['max_distance']:.2f}",
            flush=True,
        )
        maybe_log_wandb(
            {
                "candidate_eval/episodes": float(candidate_eval_episodes),
                "candidate_eval/screening_fall_rate": metrics["fall_rate"],
                "candidate_eval/screening_avg_distance": metrics["avg_distance"],
                "candidate_eval/fall_rate": candidate_metrics["fall_rate"],
                "candidate_eval/avg_distance": candidate_metrics["avg_distance"],
                "candidate_eval/max_distance": candidate_metrics["max_distance"],
                "candidate_eval/avg_forward_velocity": candidate_metrics["avg_forward_velocity"],
                "candidate_eval/avg_return": candidate_metrics["avg_return"],
            },
            commit=False,
        )
        return candidate_metrics

    def _update_eval_early_stop(self, metrics: dict, improved: bool) -> bool:
        patience = int(self.eval_cfg.get("early_stop_no_improvement_evals", 0) or 0)
        if patience <= 0:
            return False

        mode = str(self.eval_cfg.get("best_metric_mode", "success_then_distance")).lower()
        require_no_fall = bool(self.eval_cfg.get("early_stop_require_no_fall", True)) and mode != "fall_rate_then_distance"
        fell_during_eval = float(metrics.get("fall_rate", 1.0)) > 0.0
        eligible_improved = improved and ((not require_no_fall) or (not fell_during_eval))
        if eligible_improved:
            self.no_improvement_eval_count = 0
        else:
            self.no_improvement_eval_count += 1

        should_stop = self.no_improvement_eval_count >= patience
        print(
            f"[early-stop] step={self.num_timesteps:08d} mode={mode} improved={eligible_improved} "
            f"require_no_fall={require_no_fall} fell_during_eval={fell_during_eval} "
            f"fall_rate={metrics['fall_rate']:.3f} best_fall_rate={self.best_fall_rate:.3f} "
            f"avg_distance={metrics['avg_distance']:.2f} "
            f"best_no_fall_distance={self.best_no_fall_distance:.2f} "
            f"no_improvement_evals={self.no_improvement_eval_count}/{patience} "
            f"stop={should_stop}",
            flush=True,
        )
        maybe_log_wandb(
            {
                "early_stop/best_no_fall_distance": self.best_no_fall_distance,
                "early_stop/best_fall_rate": self.best_fall_rate,
                "early_stop/no_improvement_eval_count": float(self.no_improvement_eval_count),
                "early_stop/improved": float(eligible_improved),
                "early_stop/require_no_fall": float(require_no_fall),
                "early_stop/fell_during_eval": float(fell_during_eval),
                "early_stop/should_stop": float(should_stop),
            },
            commit=False,
        )
        return should_stop

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq_timesteps:
            return True
        self._last_eval_timestep = self.num_timesteps

        curriculum_levels = _get_train_curriculum_levels(self.training_env)
        curriculum_level = int(round(float(np.mean(curriculum_levels)))) if curriculum_levels else None
        _set_eval_curriculum_level(self.eval_env, curriculum_level)
        if curriculum_level is not None and hasattr(self.video_env, "set_curriculum_level"):
            self.video_env.set_curriculum_level(curriculum_level)

        sync_envs_normalization(self.training_env, self.eval_env)
        metrics = evaluate_model(self.model, self.eval_env, self.eval_cfg, recurrent=self.recurrent)
        self.eval_count += 1
        print(
            f"[eval] step={self.num_timesteps:08d} level={curriculum_level if curriculum_level is not None else -1} "
            f"weak_success_rate={metrics['weak_success_rate']:.3f} "
            f"strict_success_rate={metrics['strict_success_rate']:.3f} avg_distance={metrics['avg_distance']:.2f} "
            f"max_distance={metrics['max_distance']:.2f} avg_speed={metrics['avg_forward_velocity']:.2f} "
            f"fall_rate={metrics['fall_rate']:.3f} avg_return={metrics['avg_return']:.2f}",
            flush=True,
        )
        maybe_log_wandb(
            {
                "eval/timesteps": self.num_timesteps,
                "eval/weak_success_rate": metrics["weak_success_rate"],
                "eval/strict_success_rate": metrics["strict_success_rate"],
                "eval/avg_distance": metrics["avg_distance"],
                "eval/max_distance": metrics["max_distance"],
                "eval/avg_forward_velocity": metrics["avg_forward_velocity"],
                "eval/avg_return": metrics["avg_return"],
                "eval/fall_rate": metrics["fall_rate"],
                "eval/best_weak_success_rate": self.best_weak_success_rate,
                "eval/best_strict_success_rate": self.best_strict_success_rate,
                "eval/best_avg_distance": self.best_avg_distance,
                "eval/best_fall_rate": self.best_fall_rate,
                "eval/eval_count": float(self.eval_count),
            },
            commit=False,
        )
        if bool(self.eval_cfg.get("record_recent_video", True)):
            recent_video_path = _video_output_path(self.out_dir, "most_recent_teacher", curriculum_level)
            try:
                recent_video_metrics = record_video(
                    self.model,
                    self.video_env,
                    self.eval_env,
                    recent_video_path,
                    self.eval_cfg,
                    recurrent=self.recurrent,
                )
                print(
                    f"[recent] step={self.num_timesteps:08d} saved_video={recent_video_path} "
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
                    fps=int(self.eval_cfg["video_fps"]),
                )
            except Exception as exc:
                print(
                    f"[recent] step={self.num_timesteps:08d} video_save_failed={type(exc).__name__}: {exc}",
                    flush=True,
                )
        if curriculum_level is not None:
            maybe_log_wandb(
                {
                    "curriculum/timesteps": self.num_timesteps,
                    "curriculum/level": curriculum_level,
                    "curriculum/mean_level_eval_sync": float(np.mean(curriculum_levels)),
                    "curriculum/max_level_eval_sync": float(np.max(curriculum_levels)),
                    "curriculum/min_level_eval_sync": float(np.min(curriculum_levels)),
                },
                commit=False,
            )

        selection_metrics = self._selection_metrics_for_best(metrics)
        improved = self._is_eval_improved(selection_metrics)

        if improved:
            self._record_best_metrics(selection_metrics)

            best_model_path = self.out_dir / "best_teacher"
            self.model.save(best_model_path)
            self.training_env.save(str(self.out_dir / "best_teacher_vecnormalize.pkl"))
            best_metrics_path = self.out_dir / "best_teacher_metrics.json"
            with open(best_metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "timesteps": self.num_timesteps,
                        "eval_count": self.eval_count,
                        "metrics": selection_metrics,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )

            if bool(self.eval_cfg.get("record_best_video", True)):
                video_path = _video_output_path(self.out_dir, "best_teacher", curriculum_level)
                try:
                    video_metrics = record_video(
                        self.model,
                        self.video_env,
                        self.eval_env,
                        video_path,
                        self.eval_cfg,
                        recurrent=self.recurrent,
                    )
                    print(
                        f"[best] step={self.num_timesteps:08d} saved_model={best_model_path}.zip "
                        f"saved_video={video_path} video_distance={video_metrics['distance']:.2f} "
                        f"video_avg_speed={video_metrics['avg_forward_velocity']:.2f} "
                        f"video_weak_success={video_metrics['weak_success']} "
                        f"video_strict_success={video_metrics['strict_success']}",
                        flush=True,
                    )
                    maybe_log_wandb(
                        {
                            "eval/best_video_distance": video_metrics["distance"],
                            "eval/best_video_avg_forward_velocity": video_metrics["avg_forward_velocity"],
                            "eval/best_video_return": video_metrics["return"],
                            "eval/best_video_weak_success": float(video_metrics["weak_success"]),
                            "eval/best_video_strict_success": float(video_metrics["strict_success"]),
                        },
                        commit=False,
                    )
                    maybe_log_wandb_video(
                        "media/best_teacher_video",
                        video_path,
                        fps=int(self.eval_cfg["video_fps"]),
                    )
                except Exception as exc:
                    print(
                        f"[best] step={self.num_timesteps:08d} saved_model={best_model_path}.zip "
                        f"video_save_failed={type(exc).__name__}: {exc}",
                        flush=True,
                    )
            else:
                print(f"[best] step={self.num_timesteps:08d} saved_model={best_model_path}.zip", flush=True)
            if wandb.run is not None:
                wandb.save(str(best_model_path) + ".zip", base_path=str(self.out_dir))
                wandb.save(str(self.out_dir / "best_teacher_vecnormalize.pkl"), base_path=str(self.out_dir))
                wandb.save(str(best_metrics_path), base_path=str(self.out_dir))
                if self.eval_csv_path.exists():
                    wandb.save(str(self.eval_csv_path), base_path=str(self.out_dir))
                recent_video_path = _video_output_path(self.out_dir, "most_recent_teacher", curriculum_level)
                if recent_video_path.exists():
                    wandb.save(str(recent_video_path), base_path=str(self.out_dir))
        should_stop = self._update_eval_early_stop(selection_metrics, improved)
        if improved and bool(self.eval_cfg.get("stop_on_best_improvement", False)):
            print(
                f"[stop-on-best] step={self.num_timesteps:08d} eval_count={self.eval_count} "
                f"fall_rate={metrics['fall_rate']:.3f} avg_distance={metrics['avg_distance']:.2f}",
                flush=True,
            )
            maybe_log_wandb({"early_stop/stop_on_best_improvement": 1.0}, commit=False)
            should_stop = True
        self._append_eval_csv(metrics)
        maybe_log_wandb({}, commit=True)
        return not should_stop


def train(config: dict) -> None:
    train_cfg = config["train"]
    eval_cfg = config["eval"]
    recurrent = is_recurrent_config(config)
    device_name = train_cfg.get("device", "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("requested CUDA but no GPU is available; falling back to CPU", flush=True)
        device_name = "cpu"

    seed = int(train_cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(
        f"using_mujoco_gl={os.environ.get('MUJOCO_GL')} pyopengl_platform={os.environ.get('PYOPENGL_PLATFORM')}",
        flush=True,
    )

    out_dir = Path(train_cfg.get("output_dir", "artifacts/teacher_walk"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = maybe_init_wandb(config, out_dir)
    generate_pretrain_scene_showcase(config, out_dir)
    if wandb_run is not None:
        pretrain_dir = out_dir / "pretrain_scene_showcase"
        if pretrain_dir.exists():
            wandb.save(str(pretrain_dir / "**" / "*"), base_path=str(out_dir))

    train_env = build_train_vec_env(config, out_dir)
    train_env = maybe_load_vec_normalize(train_env, config, training=True)

    eval_env = DummyVecEnv([make_single_env({**config["env"], "seed": seed + 1, "reset_noise_scale": 0.0})])
    eval_env = maybe_load_vec_normalize(eval_env, config, training=False)

    video_env = make_env({**config["env"], "seed": seed + 2, "reset_noise_scale": 0.0})

    rollout_steps = int(train_cfg["rollout_steps"])
    total_updates = int(train_cfg["total_updates"])
    total_timesteps = int(train_cfg.get("total_timesteps", rollout_steps * total_updates))
    eval_freq_timesteps = int(eval_cfg["eval_interval_timesteps"])
    truncated_bptt_steps = int(train_cfg.get("truncated_bptt_steps", 0) or 0)
    if recurrent and truncated_bptt_steps > 0:
        print(
            f"[bptt] requested truncated_bptt_steps={truncated_bptt_steps} with rollout_steps={rollout_steps}. "
            "Using long PPO rollouts with fixed-length recurrent training chunks.",
            flush=True,
        )

    model = build_model(config, train_env, out_dir, device_name, seed)
    maybe_load_finetune_parameters(model, config, device_name)

    eval_callback = TeacherEvalCallback(
        eval_env=eval_env,
        video_env=video_env,
        out_dir=out_dir,
        eval_cfg=eval_cfg,
        eval_freq_timesteps=eval_freq_timesteps,
        recurrent=recurrent,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)
        final_model_path = out_dir / "final_teacher"
        model.save(final_model_path)
        train_env.save(str(out_dir / "final_teacher_vecnormalize.pkl"))
        print(f"saved_final_model={final_model_path}.zip", flush=True)
        if wandb.run is not None:
            wandb.save(str(final_model_path) + ".zip", base_path=str(out_dir))
            wandb.save(str(out_dir / "final_teacher_vecnormalize.pkl"), base_path=str(out_dir))
            for monitor_path in out_dir.glob("train_monitor*.csv"):
                wandb.save(str(monitor_path), base_path=str(out_dir))
            eval_csv = out_dir / "eval_metrics.csv"
            if eval_csv.exists():
                wandb.save(str(eval_csv), base_path=str(out_dir))
            maybe_log_wandb(
                {
                    "train/timesteps": total_timesteps,
                    "train/final_total_timesteps": total_timesteps,
                }
            )
    finally:
        train_env.close()
        eval_env.close()
        video_env.close()
        if wandb_run is not None:
            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk.yaml",
    )
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
