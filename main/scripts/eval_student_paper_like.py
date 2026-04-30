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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wandb

from rl.env import make_env
from rl.student_policy import load_student_checkpoint, normalize_teacher_obs, render_student_depth, resolve_path
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces
from rl.train_teacher import is_teacher_strict_success, is_teacher_weak_success, load_config, make_single_env


def _make_vecnormalize(config: dict, teacher_config: dict, vecnormalize_path: Path, seed: int):
    env_cfg = {**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0}
    vec_env = DummyVecEnv([make_single_env(env_cfg)])
    vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _eval_one(
    *,
    student,
    config: dict,
    teacher_config: dict,
    vecnormalize: VecNormalize,
    episodes: int,
    seed: int,
    device: torch.device,
    record_video_path: Path | None = None,
) -> dict:
    env = make_env({**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0})
    depth_cfg = config["student"]["depth"]
    eval_cfg = teacher_config["eval"]
    returns = []
    distances = []
    speeds = []
    weak_success = []
    strict_success = []
    falls = 0
    frames = []
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
                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                step_count += 1
                done = bool(terminated or truncated)
                episode_start = done
                if record_video_path is not None and episode == 0:
                    frames.append(env.render_frame(width=512, height=384, camera="tracking"))
            distance = float(last_info.get("x_position", 0.0))
            avg_speed = distance / max(step_count * env.config.control_dt, 1e-6)
            returns.append(total_reward)
            distances.append(distance)
            speeds.append(avg_speed)
            weak_success.append(is_teacher_weak_success(last_info, bool(terminated), eval_cfg))
            strict_success.append(is_teacher_strict_success(last_info, bool(terminated), eval_cfg))
            falls += int(bool(terminated))
        if record_video_path is not None and frames:
            record_video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(record_video_path, frames, fps=int(eval_cfg.get("video_fps", 24)))
    finally:
        env.close()
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
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _eval_tasks_from_config(config: dict, seed: int, task_filter: str | None = None) -> list[tuple[str, Path, int]]:
    all_tasks = [
        ("old", "random_slope_old", resolve_path(config["eval"]["old_config"]), int(seed)),
        ("shift", "random_slope_shift", resolve_path(config["eval"]["shift_config"]), int(seed) + 100_000),
    ]
    hard_config = config.get("eval", {}).get("hard_heldout_config")
    if hard_config:
        all_tasks.append(("hard_heldout", "random_slope_shift_hard_heldout", resolve_path(hard_config), int(seed) + 200_000))
    if task_filter:
        requested = {item.strip() for item in task_filter.split(",") if item.strip()}
        unknown = requested.difference({task[0] for task in all_tasks})
        if unknown:
            raise ValueError(f"unknown --tasks value(s): {sorted(unknown)}; valid values are old,shift,hard_heldout")
        all_tasks = [task for task in all_tasks if task[0] in requested]
    return [(task_name, config_path, task_seed) for _alias, task_name, config_path, task_seed in all_tasks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-vecnormalize", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_bc" / "best_student.pt")
    parser.add_argument("--dagger-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_dagger_resume_from_round3" / "best_student.pt")
    parser.add_argument("--student-checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--checkpoint-label", type=str, action="append", default=None)
    parser.add_argument("--policies", type=str, default=None, help="Comma-separated student policy labels to evaluate, e.g. paper_like_finetune")
    parser.add_argument("--student-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated subset: old,shift,hard_heldout")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = load_student_config(args.config)
    device = device_from_config(config, args.device)
    eval_cfg = dict(config["eval"])
    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 100))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", 12345))
    if args.debug:
        episodes = min(episodes, 2)
        config.setdefault("wandb", {})["enabled"] = False
    checkpoint_args = args.student_checkpoint
    requested_policies = [item.strip() for item in args.policies.split(",") if item.strip()] if args.policies else []
    compare_teacher_bc_dagger = checkpoint_args is None and not args.student_only and not requested_policies
    observation_space, action_space = make_student_env_spaces(config, seed=seed)
    vecnormalize_path = resolve_path(args.teacher_vecnormalize or config["teacher"]["vecnormalize"])
    tasks = _eval_tasks_from_config(config, seed, args.tasks)

    output_dir = resolve_path(eval_cfg.get("save_dir", "artifacts/student_paper_like_eval"))
    rows = []
    run = None
    wandb_cfg = config.get("wandb", {})
    if bool(wandb_cfg.get("enabled", False)):
        run = wandb.init(
            project=wandb_cfg.get("project", "quadruped-student"),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("run_name", "student-paper-like")) + "-eval",
            group=wandb_cfg.get("group", "student-paper-like"),
            tags=[*wandb_cfg.get("tags", []), "eval"],
            dir=str(output_dir),
            config=config,
            mode=wandb_cfg.get("mode", "online"),
        )
    try:
        if compare_teacher_bc_dagger:
            from scripts.eval_teacher_bc_dagger_old_shift import _eval_student as _eval_student_compare
            from scripts.eval_teacher_bc_dagger_old_shift import _eval_teacher

            teacher_checkpoint = resolve_path(args.teacher_checkpoint or config["teacher"]["checkpoint"])
            bc_checkpoint = resolve_path(args.bc_checkpoint)
            dagger_checkpoint = resolve_path(args.dagger_checkpoint)
            missing = [path for path in (teacher_checkpoint, vecnormalize_path, bc_checkpoint, dagger_checkpoint) if not path.exists()]
            if missing:
                raise FileNotFoundError("missing checkpoint/normalization file(s): " + ", ".join(str(path) for path in missing))
            bc_student = load_student_checkpoint(bc_checkpoint, config, observation_space, action_space, device)
            dagger_student = load_student_checkpoint(dagger_checkpoint, config, observation_space, action_space, device)
            for label in ("Teacher", "BC", "DAgger"):
                for task_name, teacher_cfg_path, task_seed in tasks:
                    video_path = output_dir / f"{label}_{task_name}.mp4" if args.record_video else None
                    if label == "Teacher":
                        metrics = _eval_teacher(
                            config_path=teacher_cfg_path,
                            checkpoint=teacher_checkpoint,
                            vecnormalize_path=vecnormalize_path,
                            episodes=episodes,
                            seed=task_seed,
                            device=device,
                            output_dir=output_dir,
                            video_path=video_path,
                        )
                        checkpoint = teacher_checkpoint
                    else:
                        student = bc_student if label == "BC" else dagger_student
                        checkpoint = bc_checkpoint if label == "BC" else dagger_checkpoint
                        metrics = _eval_student_compare(
                            student=student,
                            student_config=config,
                            teacher_config_path=teacher_cfg_path,
                            teacher_vecnormalize_path=vecnormalize_path,
                            episodes=episodes,
                            seed=task_seed,
                            device=device,
                            video_path=video_path,
                        )
                    row = {
                        "policy": label,
                        "task": task_name,
                        "checkpoint": str(checkpoint),
                        "episodes": episodes,
                        "seed": task_seed,
                        **metrics,
                        "video_path": "" if video_path is None else str(video_path),
                    }
                    rows.append(row)
                    print(
                        f"[eval] policy={label} task={task_name} "
                        f"fall_rate={metrics['fall_rate']:.4f} avg_distance={metrics['avg_distance']:.4f} "
                        f"avg_speed={metrics['avg_forward_velocity']:.4f}",
                        flush=True,
                    )
                    if wandb.run is not None:
                        wandb.log({f"eval/{label}_{task_name}_{key}": value for key, value in metrics.items()})
                        if video_path is not None and video_path.exists():
                            wandb.log({f"media/{label}_{task_name}_video": wandb.Video(str(video_path), fps=24, format="mp4")})
        else:
            if not checkpoint_args:
                checkpoint_args = [Path(eval_cfg.get("student_checkpoint", config["dagger"].get("best_checkpoint", "artifacts/student_paper_like_dagger/best_student.pt")))]
            checkpoints = [resolve_path(path) for path in checkpoint_args]
            labels = requested_policies or args.checkpoint_label or []
            if labels and len(labels) != len(checkpoints):
                raise ValueError("--policies/--checkpoint-label must be provided the same number of times as --student-checkpoint")
            if not labels:
                labels = [path.parent.name if path.name == "best_student.pt" else path.stem for path in checkpoints]
            for checkpoint, label in zip(checkpoints, labels, strict=True):
                if not checkpoint.exists():
                    raise FileNotFoundError(f"student checkpoint not found: {checkpoint}")
                student = load_student_checkpoint(checkpoint, config, observation_space, action_space, device)
                for task_name, teacher_cfg_path, task_seed in tasks:
                    teacher_config = load_config(teacher_cfg_path)
                    vecnormalize = _make_vecnormalize(config, teacher_config, vecnormalize_path, seed=task_seed)
                    video_path = output_dir / f"{label}_{task_name}.mp4" if args.record_video else None
                    metrics = _eval_one(
                        student=student,
                        config=config,
                        teacher_config=teacher_config,
                        vecnormalize=vecnormalize,
                        episodes=episodes,
                        seed=task_seed,
                        device=device,
                        record_video_path=video_path,
                    )
                    vecnormalize.close()
                    row = {
                        "policy": label,
                        "task": task_name,
                        "checkpoint": str(checkpoint),
                        "episodes": episodes,
                        "seed": task_seed,
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"[student-eval] label={label} {task_name} "
                        f"fall_rate={metrics['fall_rate']:.4f} avg_distance={metrics['avg_distance']:.4f} "
                        f"avg_speed={metrics['avg_forward_velocity']:.4f}",
                        flush=True,
                    )
                    if wandb.run is not None:
                        wandb.log({f"eval/{label}_{task_name}_{key}": value for key, value in metrics.items()})
                        if video_path is not None and video_path.exists():
                            wandb.log({f"media/{label}_{task_name}_video": wandb.Video(str(video_path), fps=24, format="mp4")})
        output_path = resolve_path(args.output) if args.output is not None else output_dir / "student_eval_summary.csv"
        _write_csv(output_path, rows)
        print(f"[student-eval] saved_summary={output_path}", flush=True)
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
