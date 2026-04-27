from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
import types


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
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.env import make_env
from rl.gru_policy import ScandotGruPolicy  # noqa: F401
from rl.student_policy import load_student_checkpoint, normalize_teacher_obs, render_student_depth, resolve_path
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces
from rl.train_teacher import (
    build_model,
    evaluate_model,
    is_recurrent_config,
    is_teacher_strict_success,
    is_teacher_weak_success,
    load_config,
    make_single_env,
)


def _make_vecnormalize(teacher_config: dict, vecnormalize_path: Path, seed: int) -> VecNormalize:
    env_cfg = {**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0}
    vec_env = DummyVecEnv([make_single_env(env_cfg)])
    vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _load_teacher_model(config_path: Path, checkpoint: Path, vecnormalize_path: Path, seed: int, device: torch.device, output_dir: Path):
    teacher_config = load_config(config_path)
    vec_env = _make_vecnormalize(teacher_config, vecnormalize_path, seed)
    model = build_model(teacher_config, vec_env, output_dir, str(device), seed)
    _, params, _ = load_from_zip_file(str(checkpoint), device=str(device), print_system_info=False)
    if params is None:
        raise FileNotFoundError(f"could not load teacher params from {checkpoint}")
    model.set_parameters(params, exact_match=True, device=str(device))
    return model, vec_env, teacher_config


def _eval_teacher(
    *,
    config_path: Path,
    checkpoint: Path,
    vecnormalize_path: Path,
    episodes: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    video_path: Path | None,
) -> dict:
    model, vec_env, teacher_config = _load_teacher_model(config_path, checkpoint, vecnormalize_path, seed, device, output_dir)
    try:
        base_vec = vec_env.venv
        if hasattr(base_vec, "env_method"):
            base_vec.env_method("set_fixed_terrain_type", "random_slope_up_down")
        eval_cfg = dict(teacher_config["eval"])
        eval_cfg["eval_episodes"] = int(episodes)
        metrics = evaluate_model(model, vec_env, eval_cfg, recurrent=is_recurrent_config(teacher_config))
    finally:
        vec_env.close()

    if video_path is not None:
        model, vec_env, teacher_config = _load_teacher_model(config_path, checkpoint, vecnormalize_path, seed + 999, device, output_dir)
        env = make_env({**teacher_config["env"], "seed": int(seed) + 999, "reset_noise_scale": 0.0})
        frames = []
        obs, _ = env.reset(options={"terrain_type": "random_slope_up_down"})
        teacher_state = None
        episode_start = True
        done = False
        try:
            while not done:
                norm_obs = vec_env.normalize_obs({key: value[None] for key, value in obs.items()})
                action, teacher_state = model.predict(
                    norm_obs,
                    state=teacher_state,
                    episode_start=np.asarray([episode_start], dtype=bool),
                    deterministic=True,
                )
                frames.append(env.render_frame(width=512, height=384, camera="tracking"))
                obs, _reward, terminated, truncated, _info = env.step(np.asarray(action[0], dtype=np.float32))
                done = bool(terminated or truncated)
                episode_start = done
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=int(teacher_config["eval"].get("video_fps", 24)))
        finally:
            env.close()
            vec_env.close()
    return metrics


def _eval_student(
    *,
    student,
    student_config: dict,
    teacher_config_path: Path,
    teacher_vecnormalize_path: Path,
    episodes: int,
    seed: int,
    device: torch.device,
    video_path: Path | None,
) -> dict:
    teacher_config = load_config(teacher_config_path)
    vecnormalize = _make_vecnormalize(teacher_config, teacher_vecnormalize_path, seed)
    env = make_env({**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0})
    depth_cfg = student_config["student"]["depth"]
    eval_cfg = teacher_config["eval"]
    returns: list[float] = []
    distances: list[float] = []
    speeds: list[float] = []
    weak_success: list[bool] = []
    strict_success: list[bool] = []
    falls = 0
    video_frames = []
    try:
        for episode in range(int(episodes)):
            obs, _ = env.reset(options={"terrain_type": "random_slope_up_down"})
            done = False
            terminated = False
            total_reward = 0.0
            step_count = 0
            hidden = None
            episode_start = True
            last_info = {"x_position": 0.0}
            while not done:
                norm_obs = normalize_teacher_obs(vecnormalize, obs)
                depth = render_student_depth(env, depth_cfg)
                action, hidden = student.predict(
                    depth,
                    norm_obs["proprio"],
                    norm_obs["command"],
                    hidden,
                    episode_start,
                    device,
                )
                if video_path is not None and episode == 0:
                    video_frames.append(env.render_frame(width=512, height=384, camera="tracking"))
                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                step_count += 1
                done = bool(terminated or truncated)
                episode_start = done
            distance = float(last_info.get("x_position", 0.0))
            returns.append(total_reward)
            distances.append(distance)
            speeds.append(distance / max(step_count * env.config.control_dt, 1e-6))
            weak_success.append(is_teacher_weak_success(last_info, bool(terminated), eval_cfg))
            strict_success.append(is_teacher_strict_success(last_info, bool(terminated), eval_cfg))
            falls += int(bool(terminated))
        if video_path is not None and video_frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, video_frames, fps=int(eval_cfg.get("video_fps", 24)))
    finally:
        env.close()
        vecnormalize.close()
    return {
        "fall_rate": float(falls / max(int(episodes), 1)),
        "avg_distance": float(np.mean(distances)) if distances else 0.0,
        "max_distance": float(np.max(distances)) if distances else 0.0,
        "avg_forward_velocity": float(np.mean(speeds)) if speeds else 0.0,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "weak_success_rate": float(np.mean(weak_success)) if weak_success else 0.0,
        "strict_success_rate": float(np.mean(strict_success)) if strict_success else 0.0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _eval_tasks_from_config(config: dict, seed: int) -> list[tuple[str, Path, int]]:
    old_config = resolve_path(config["eval"]["old_config"])
    shift_config = resolve_path(config["eval"]["shift_config"])
    tasks = [
        ("random_slope_old", old_config, int(seed)),
        ("random_slope_shift", shift_config, int(seed) + 100_000),
    ]
    hard_config = config.get("eval", {}).get("hard_heldout_config")
    if hard_config:
        tasks.append(("random_slope_shift_hard_heldout", resolve_path(hard_config), int(seed) + 200_000))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-vecnormalize", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_bc" / "best_student.pt")
    parser.add_argument("--dagger-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_dagger_resume_from_round3" / "best_student.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_eval" / "teacher_bc_dagger_100eps.csv")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_eval" / "teacher_bc_dagger_videos")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = load_student_config(args.config)
    if args.debug:
        args.episodes = min(int(args.episodes), 2)
        config.setdefault("wandb", {})["enabled"] = False
    device = device_from_config(config, args.device)
    teacher_checkpoint = resolve_path(args.teacher_checkpoint or config["teacher"]["checkpoint"])
    teacher_vecnormalize = resolve_path(args.teacher_vecnormalize or config["teacher"]["vecnormalize"])
    bc_checkpoint = resolve_path(args.bc_checkpoint)
    dagger_checkpoint = resolve_path(args.dagger_checkpoint)
    output_path = resolve_path(args.output)
    video_dir = resolve_path(args.video_dir)
    tasks = _eval_tasks_from_config(config, int(args.seed))
    observation_space, action_space = make_student_env_spaces(config, seed=int(args.seed))
    bc_student = load_student_checkpoint(bc_checkpoint, config, observation_space, action_space, device)
    dagger_student = load_student_checkpoint(dagger_checkpoint, config, observation_space, action_space, device)
    rows = []
    for policy_label in ("Teacher", "BC", "DAgger"):
        for task_name, task_config, task_seed in tasks:
            video_path = video_dir / f"{policy_label}_{task_name}.mp4" if args.record_video else None
            if policy_label == "Teacher":
                metrics = _eval_teacher(
                    config_path=task_config,
                    checkpoint=teacher_checkpoint,
                    vecnormalize_path=teacher_vecnormalize,
                    episodes=int(args.episodes),
                    seed=task_seed,
                    device=device,
                    output_dir=output_path.parent,
                    video_path=video_path,
                )
                checkpoint = teacher_checkpoint
            else:
                student = bc_student if policy_label == "BC" else dagger_student
                checkpoint = bc_checkpoint if policy_label == "BC" else dagger_checkpoint
                metrics = _eval_student(
                    student=student,
                    student_config=config,
                    teacher_config_path=task_config,
                    teacher_vecnormalize_path=teacher_vecnormalize,
                    episodes=int(args.episodes),
                    seed=task_seed,
                    device=device,
                    video_path=video_path,
                )
            row = {
                "policy": policy_label,
                "task": task_name,
                "checkpoint": str(checkpoint),
                "episodes": int(args.episodes),
                "seed": int(task_seed),
                **metrics,
                "video_path": "" if video_path is None else str(video_path),
            }
            rows.append(row)
            print(
                f"[eval] policy={policy_label} task={task_name} "
                f"fall_rate={metrics['fall_rate']:.4f} avg_distance={metrics['avg_distance']:.4f} "
                f"avg_speed={metrics['avg_forward_velocity']:.4f}",
                flush=True,
            )
    _write_csv(output_path, rows)
    print(f"saved_summary={output_path}", flush=True)


if __name__ == "__main__":
    main()
