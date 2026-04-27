from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl.env import make_env
from rl.student_policy import normalize_teacher_obs, render_student_depth


def _stack_buffers(buffers: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        "depth": np.asarray(buffers["depth"], dtype=np.float32),
        "proprio": np.asarray(buffers["proprio"], dtype=np.float32),
        "command": np.asarray(buffers["command"], dtype=np.float32),
        "action": np.asarray(buffers["action"], dtype=np.float32),
        "done": np.asarray(buffers["done"], dtype=np.bool_),
        "episode_start": np.asarray(buffers["episode_start"], dtype=np.bool_),
    }


def collect_teacher_bc_rollout(
    *,
    teacher_model,
    teacher_vecnormalize,
    teacher_config: dict,
    student_config: dict,
    num_steps: int,
    seed: int,
    terrain_type: str | None = None,
    terrain_level: int | None = None,
) -> dict[str, np.ndarray]:
    env_cfg = {**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0}
    env = make_env(env_cfg)
    if terrain_type is not None:
        env.set_fixed_terrain_type(terrain_type)
    if terrain_level is not None:
        env.set_curriculum_level(int(terrain_level))
    buffers = {key: [] for key in ["depth", "proprio", "command", "action", "done", "episode_start"]}
    obs, _ = env.reset(options={"terrain_type": terrain_type} if terrain_type is not None else None)
    teacher_state = None
    episode_start = True
    try:
        for _ in range(int(num_steps)):
            normalized = normalize_teacher_obs(teacher_vecnormalize, obs)
            depth = render_student_depth(env, student_config["student"]["depth"])
            action, teacher_state = teacher_model.predict(
                {key: value[None] for key, value in normalized.items()},
                state=teacher_state,
                episode_start=np.asarray([episode_start], dtype=bool),
                deterministic=True,
            )
            action = np.asarray(action[0], dtype=np.float32)
            buffers["depth"].append(depth)
            buffers["proprio"].append(normalized["proprio"])
            buffers["command"].append(normalized["command"])
            buffers["action"].append(action)
            buffers["episode_start"].append(bool(episode_start))
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            buffers["done"].append(done)
            episode_start = done
            if done:
                obs, _ = env.reset(options={"terrain_type": terrain_type} if terrain_type is not None else None)
                teacher_state = None
    finally:
        env.close()
    return _stack_buffers(buffers)


def collect_dagger_rollout(
    *,
    student,
    teacher_model,
    teacher_vecnormalize,
    teacher_config: dict,
    student_config: dict,
    num_steps: int,
    seed: int,
    device: torch.device,
    terrain_type: str | None = None,
    terrain_level: int | None = None,
) -> dict[str, np.ndarray]:
    env_cfg = {**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0}
    env = make_env(env_cfg)
    if terrain_type is not None:
        env.set_fixed_terrain_type(terrain_type)
    if terrain_level is not None:
        env.set_curriculum_level(int(terrain_level))
    buffers = {key: [] for key in ["depth", "proprio", "command", "action", "done", "episode_start"]}
    obs, _ = env.reset(options={"terrain_type": terrain_type} if terrain_type is not None else None)
    teacher_state = None
    student_hidden = None
    episode_start = True
    student.eval()
    try:
        for _ in range(int(num_steps)):
            normalized = normalize_teacher_obs(teacher_vecnormalize, obs)
            depth = render_student_depth(env, student_config["student"]["depth"])
            student_action, student_hidden = student.predict(
                depth,
                normalized["proprio"],
                normalized["command"],
                student_hidden,
                episode_start,
                device,
            )
            teacher_action, teacher_state = teacher_model.predict(
                {key: value[None] for key, value in normalized.items()},
                state=teacher_state,
                episode_start=np.asarray([episode_start], dtype=bool),
                deterministic=True,
            )
            teacher_action = np.asarray(teacher_action[0], dtype=np.float32)
            buffers["depth"].append(depth)
            buffers["proprio"].append(normalized["proprio"])
            buffers["command"].append(normalized["command"])
            buffers["action"].append(teacher_action)
            buffers["episode_start"].append(bool(episode_start))
            obs, _reward, terminated, truncated, _info = env.step(student_action)
            done = bool(terminated or truncated)
            buffers["done"].append(done)
            episode_start = done
            if done:
                obs, _ = env.reset(options={"terrain_type": terrain_type} if terrain_type is not None else None)
                teacher_state = None
                student_hidden = None
    finally:
        env.close()
    return _stack_buffers(buffers)
