from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_teacher_bc_dagger_old_shift import _eval_student, _eval_teacher, _write_csv
from rl.student_policy import load_student_checkpoint, resolve_path
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces


def _print_guidance() -> None:
    print(
        "\n[difficulty-guidance]\n"
        "Ideal hard held-out range:\n"
        "  Teacher fall_rate: 0.05 - 0.20\n"
        "  BC fall_rate:      0.15 - 0.35\n"
        "If Teacher fall_rate > 0.30, the held-out task is probably too hard.\n"
        "If BC fall_rate is near 0, it is probably too easy for student ablations.\n"
        "If BC avg_distance is very low or avg_forward_velocity is near 0, inspect whether the policy is surviving by not moving.\n"
        "If Teacher and BC are both poor, this terrain is not a good primary student-ablation scenario.\n",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-vecnormalize", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_bc" / "best_student.pt")
    parser.add_argument("--dagger-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_dagger_resume_from_round3" / "best_student.pt")
    parser.add_argument("--skip-dagger", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_eval" / "hard_heldout_difficulty.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = load_student_config(args.config)
    if args.debug:
        args.episodes = min(int(args.episodes), 2)
        config.setdefault("wandb", {})["enabled"] = False
    if not config.get("eval", {}).get("hard_heldout_config"):
        raise KeyError("configs/student_paper_like.yaml eval.hard_heldout_config is required")

    device = device_from_config(config, args.device)
    teacher_checkpoint = resolve_path(args.teacher_checkpoint or config["teacher"]["checkpoint"])
    teacher_vecnormalize = resolve_path(args.teacher_vecnormalize or config["teacher"]["vecnormalize"])
    hard_config = resolve_path(config["eval"]["hard_heldout_config"])
    task_seed = int(args.seed) + 200_000
    output_path = resolve_path(args.output)
    rows = []

    teacher_metrics = _eval_teacher(
        config_path=hard_config,
        checkpoint=teacher_checkpoint,
        vecnormalize_path=teacher_vecnormalize,
        episodes=int(args.episodes),
        seed=task_seed,
        device=device,
        output_dir=output_path.parent,
        video_path=None,
    )
    rows.append({"policy": "Teacher", "task": "random_slope_shift_hard_heldout", "checkpoint": str(teacher_checkpoint), "episodes": int(args.episodes), "seed": task_seed, **teacher_metrics, "video_path": ""})
    print(f"[hard-heldout] Teacher fall_rate={teacher_metrics['fall_rate']:.4f} avg_distance={teacher_metrics['avg_distance']:.4f} avg_speed={teacher_metrics['avg_forward_velocity']:.4f} avg_return={teacher_metrics['avg_return']:.4f}", flush=True)

    observation_space, action_space = make_student_env_spaces(config, seed=int(args.seed))
    bc_checkpoint = resolve_path(args.bc_checkpoint)
    bc_student = load_student_checkpoint(bc_checkpoint, config, observation_space, action_space, device)
    bc_metrics = _eval_student(
        student=bc_student,
        student_config=config,
        teacher_config_path=hard_config,
        teacher_vecnormalize_path=teacher_vecnormalize,
        episodes=int(args.episodes),
        seed=task_seed,
        device=device,
        video_path=None,
    )
    rows.append({"policy": "BC", "task": "random_slope_shift_hard_heldout", "checkpoint": str(bc_checkpoint), "episodes": int(args.episodes), "seed": task_seed, **bc_metrics, "video_path": ""})
    print(f"[hard-heldout] BC fall_rate={bc_metrics['fall_rate']:.4f} avg_distance={bc_metrics['avg_distance']:.4f} avg_speed={bc_metrics['avg_forward_velocity']:.4f} avg_return={bc_metrics['avg_return']:.4f}", flush=True)

    dagger_checkpoint = resolve_path(args.dagger_checkpoint)
    if not args.skip_dagger and dagger_checkpoint.exists():
        dagger_student = load_student_checkpoint(dagger_checkpoint, config, observation_space, action_space, device)
        dagger_metrics = _eval_student(
            student=dagger_student,
            student_config=config,
            teacher_config_path=hard_config,
            teacher_vecnormalize_path=teacher_vecnormalize,
            episodes=int(args.episodes),
            seed=task_seed,
            device=device,
            video_path=None,
        )
        rows.append({"policy": "DAgger", "task": "random_slope_shift_hard_heldout", "checkpoint": str(dagger_checkpoint), "episodes": int(args.episodes), "seed": task_seed, **dagger_metrics, "video_path": ""})
        print(f"[hard-heldout] DAgger fall_rate={dagger_metrics['fall_rate']:.4f} avg_distance={dagger_metrics['avg_distance']:.4f} avg_speed={dagger_metrics['avg_forward_velocity']:.4f} avg_return={dagger_metrics['avg_return']:.4f}", flush=True)
    elif not args.skip_dagger:
        print(f"[hard-heldout] skipping DAgger because checkpoint does not exist: {dagger_checkpoint}", flush=True)

    _write_csv(output_path, rows)
    print(f"saved_summary={output_path}", flush=True)
    _print_guidance()


if __name__ == "__main__":
    main()
