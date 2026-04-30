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

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.student_policy import get_student_gate_stats, load_student_checkpoint, resolve_path
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces
from scripts.eval_teacher_bc_dagger_old_shift import _eval_student


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tasks(config: dict, seed: int) -> list[tuple[str, Path, int]]:
    return [
        ("random_slope_old", resolve_path(config["eval"]["old_config"]), int(seed)),
        ("random_slope_shift", resolve_path(config["eval"]["shift_config"]), int(seed) + 100_000),
        ("random_slope_shift_hard_heldout", resolve_path(config["eval"]["hard_heldout_config"]), int(seed) + 200_000),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_residual_gated_imu_clean.yaml")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-vecnormalize", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = load_student_config(args.config)
    config.setdefault("wandb", {})["enabled"] = False
    device = device_from_config(config, args.device)
    episodes = int(args.episodes if args.episodes is not None else config["eval"].get("episodes", 100))
    seed = int(args.seed if args.seed is not None else config["eval"].get("seed", 12345))
    if args.debug:
        episodes = min(episodes, 2)
    checkpoint = resolve_path(args.student_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint}")
    observation_space, action_space = make_student_env_spaces(config, seed=seed)
    student = load_student_checkpoint(checkpoint, config, observation_space, action_space, device)
    teacher_vecnormalize_path = resolve_path(args.teacher_vecnormalize or config["teacher"]["vecnormalize"])
    rows = []
    for task_name, task_config, task_seed in _tasks(config, seed):
        metrics = _eval_student(
            student=student,
            student_config=config,
            teacher_config_path=task_config,
            teacher_vecnormalize_path=teacher_vecnormalize_path,
            episodes=episodes,
            seed=task_seed,
            device=device,
            video_path=None,
        )
        row = {
            "policy": "ResidualGated",
            "task": task_name,
            "checkpoint": str(checkpoint),
            "episodes": episodes,
            "seed": task_seed,
            **metrics,
            **get_student_gate_stats(student),
        }
        rows.append(row)
        print(
            f"[gated-eval] task={task_name} fall_rate={metrics['fall_rate']:.4f} "
            f"avg_distance={metrics['avg_distance']:.4f} avg_speed={metrics['avg_forward_velocity']:.4f} "
            f"beta={row['beta_value']:.6f} alpha_mean={row['alpha_mean']:.4f}",
            flush=True,
        )
    output = resolve_path(args.output) if args.output else resolve_path(config["eval"].get("save_dir", "artifacts/student_gated_eval")) / "gated_eval_summary.csv"
    _write_csv(output, rows)
    print(f"[gated-eval] saved_summary={output}", flush=True)


if __name__ == "__main__":
    main()
