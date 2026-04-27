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

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.student_policy import load_student_checkpoint, resolve_path
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces
from scripts.eval_teacher_bc_dagger_old_shift import _eval_student, _eval_teacher


TASKS = {
    "random_slope_shift": ("shift_config", 100_000),
    "random_slope_shift_hard_heldout": ("hard_heldout_config", 200_000),
}

FIELDNAMES = [
    "seed_index",
    "base_seed",
    "task_seed",
    "policy",
    "task",
    "checkpoint",
    "config",
    "episodes",
    "fall_rate",
    "avg_distance",
    "max_distance",
    "avg_forward_velocity",
    "avg_return",
    "weak_success_rate",
    "strict_success_rate",
    "eval_score",
]


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _append_seed_rows(path: Path, seed_rows: list[dict]) -> None:
    rows = _read_rows(path)
    rows.extend({key: row.get(key, "") for key in FIELDNAMES} for row in seed_rows)
    _write_rows(path, rows)


def _seed_complete(rows: list[dict], base_seed: int, policies: list[str], tasks: list[str]) -> bool:
    expected = {(policy, task) for policy in policies for task in tasks}
    observed = {
        (row.get("policy"), row.get("task"))
        for row in rows
        if row.get("base_seed") == str(int(base_seed))
    }
    return expected.issubset(observed)


def _student_specs(args) -> list[dict]:
    return [
        {
            "policy": "paper_like_finetune",
            "config": resolve_path(args.paper_finetune_config),
            "checkpoint": resolve_path(args.paper_finetune_checkpoint),
        },
        {
            "policy": "residual_gated_imu_clean",
            "config": resolve_path(args.gated_config),
            "checkpoint": resolve_path(args.gated_checkpoint),
        },
        {
            "policy": "residual_bilinear_imu_clean",
            "config": resolve_path(args.bilinear_config),
            "checkpoint": resolve_path(args.bilinear_checkpoint),
        },
    ]


def _load_students(specs: list[dict], device, seed: int) -> dict[str, tuple[dict, object]]:
    loaded = {}
    for spec in specs:
        config = load_student_config(spec["config"])
        config.setdefault("wandb", {})["enabled"] = False
        observation_space, action_space = make_student_env_spaces(config, seed=seed)
        student = load_student_checkpoint(spec["checkpoint"], config, observation_space, action_space, device)
        loaded[spec["policy"]] = (config, student)
    return loaded


def _eval_score(metrics: dict) -> float:
    return float(metrics["fall_rate"] - 0.05 * metrics["avg_distance"])


def _eval_teacher_row(*, base_config: dict, args, task: str, base_seed: int, seed_index: int, episodes: int, device, video_path: Path | None = None) -> dict:
    config_key, seed_offset = TASKS[task]
    task_seed = int(base_seed) + int(seed_offset)
    config_path = resolve_path(base_config["eval"][config_key])
    metrics = _eval_teacher(
        config_path=config_path,
        checkpoint=resolve_path(args.teacher_checkpoint),
        vecnormalize_path=resolve_path(args.teacher_vecnormalize),
        episodes=int(episodes),
        seed=task_seed,
        device=device,
        output_dir=resolve_path(args.output).parent,
        video_path=video_path,
    )
    return {
        "seed_index": seed_index,
        "base_seed": int(base_seed),
        "task_seed": task_seed,
        "policy": "teacher",
        "task": task,
        "checkpoint": str(resolve_path(args.teacher_checkpoint)),
        "config": str(config_path),
        "episodes": int(episodes),
        **metrics,
        "eval_score": _eval_score(metrics),
    }


def _eval_student_row(*, spec: dict, config: dict, student, args, task: str, base_seed: int, seed_index: int, episodes: int, device, video_path: Path | None = None) -> dict:
    config_key, seed_offset = TASKS[task]
    task_seed = int(base_seed) + int(seed_offset)
    metrics = _eval_student(
        student=student,
        student_config=config,
        teacher_config_path=resolve_path(config["eval"][config_key]),
        teacher_vecnormalize_path=resolve_path(args.teacher_vecnormalize),
        episodes=int(episodes),
        seed=task_seed,
        device=device,
        video_path=video_path,
    )
    return {
        "seed_index": seed_index,
        "base_seed": int(base_seed),
        "task_seed": task_seed,
        "policy": spec["policy"],
        "task": task,
        "checkpoint": str(spec["checkpoint"]),
        "config": str(spec["config"]),
        "episodes": int(episodes),
        **metrics,
        "eval_score": _eval_score(metrics),
    }


def _log_row(row: dict, *, prefix: str) -> None:
    if wandb.run is None:
        return
    key_prefix = f"{prefix}/{row['policy']}/{row['task']}/seed_{row['base_seed']}"
    wandb.log(
        {
            f"{key_prefix}/fall_rate": float(row["fall_rate"]),
            f"{key_prefix}/avg_distance": float(row["avg_distance"]),
            f"{key_prefix}/avg_forward_velocity": float(row["avg_forward_velocity"]),
            f"{key_prefix}/avg_return": float(row["avg_return"]),
            f"{key_prefix}/weak_success_rate": float(row["weak_success_rate"]),
            f"{key_prefix}/strict_success_rate": float(row["strict_success_rate"]),
            f"{key_prefix}/eval_score": float(row["eval_score"]),
            "seed_index": int(row["seed_index"]),
        }
    )


def _record_videos(*, args, base_config: dict, student_specs: list[dict], loaded_students: dict, device, tasks: list[str]) -> None:
    if args.skip_videos:
        return
    video_dir = resolve_path(args.video_dir)
    video_seed = int(args.video_seed)
    for task in tasks:
        video_path = video_dir / f"teacher_{task}.mp4"
        row = _eval_teacher_row(
            base_config=base_config,
            args=args,
            task=task,
            base_seed=video_seed,
            seed_index=-1,
            episodes=1,
            device=device,
            video_path=video_path,
        )
        print(f"[final-eval-video] policy=teacher task={task} video={video_path}", flush=True)
        if wandb.run is not None and video_path.exists():
            wandb.log({f"videos/teacher/{task}": wandb.Video(str(video_path), fps=24, format="mp4")})
        _log_row(row, prefix="video_eval")

    for spec in student_specs:
        config, student = loaded_students[spec["policy"]]
        for task in tasks:
            video_path = video_dir / f"{spec['policy']}_{task}.mp4"
            row = _eval_student_row(
                spec=spec,
                config=config,
                student=student,
                args=args,
                task=task,
                base_seed=video_seed,
                seed_index=-1,
                episodes=1,
                device=device,
                video_path=video_path,
            )
            print(f"[final-eval-video] policy={spec['policy']} task={task} video={video_path}", flush=True)
            if wandb.run is not None and video_path.exists():
                wandb.log({f"videos/{spec['policy']}/{task}": wandb.Video(str(video_path), fps=24, format="mp4")})
            _log_row(row, prefix="video_eval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--paper-finetune-config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like_finetune_same_budget.yaml")
    parser.add_argument("--gated-config", type=Path, default=REPO_ROOT / "configs" / "student_residual_gated_imu_clean.yaml")
    parser.add_argument("--bilinear-config", type=Path, default=REPO_ROOT / "configs" / "student_residual_bilinear_imu_clean.yaml")
    parser.add_argument("--paper-finetune-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_finetune_same_budget" / "best_by_eval.pt")
    parser.add_argument("--gated-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_residual_gated_imu_clean_bc" / "best_by_val_loss.pt")
    parser.add_argument("--bilinear-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_residual_bilinear_imu_clean_bc" / "best_by_eval.pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_shifted_no_corridor_iterative_finetune" / "v11" / "best_teacher.zip")
    parser.add_argument("--teacher-vecnormalize", type=Path, default=REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_shifted_no_corridor_iterative_finetune" / "v11" / "best_teacher_vecnormalize.pkl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "final_model_comparison" / "teacher_paper_gated_bilinear_6seed_50eps.csv")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "artifacts" / "final_model_comparison" / "videos")
    parser.add_argument("--seeds", type=str, default="12345,22345,32345,42345,52345,62345")
    parser.add_argument("--video-seed", type=int, default=12345)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    parser.add_argument("--wandb-project", type=str, default="quadruped-student")
    parser.add_argument("--wandb-run-name", type=str, default="final-teacher-paper-gated-bilinear-6seed")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if args.debug:
        seeds = seeds[:1]
        args.episodes = 1
        args.wandb_mode = "disabled"

    device = device_from_config({"student": {"device": args.device}}, args.device)
    base_config = load_student_config(args.base_config)
    base_config.setdefault("wandb", {})["enabled"] = False
    student_specs = _student_specs(args)
    for path in [args.teacher_checkpoint, args.teacher_vecnormalize, args.base_config, *[spec["config"] for spec in student_specs], *[spec["checkpoint"] for spec in student_specs]]:
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"missing required file: {resolved}")
    loaded_students = _load_students(student_specs, device, seed=seeds[0])
    tasks = list(TASKS.keys())
    policies = ["teacher", *[spec["policy"] for spec in student_specs]]

    run = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group="final-model-comparison",
            tags=["final-eval", "teacher", "paper-like", "gated", "bilinear", "multiseed"],
            mode=args.wandb_mode,
            config={
                "seeds": seeds,
                "episodes": int(args.episodes),
                "tasks": tasks,
                "teacher_checkpoint": str(resolve_path(args.teacher_checkpoint)),
                "teacher_vecnormalize": str(resolve_path(args.teacher_vecnormalize)),
                "student_specs": student_specs,
            },
        )

    try:
        _record_videos(args=args, base_config=base_config, student_specs=student_specs, loaded_students=loaded_students, device=device, tasks=tasks)
        output = resolve_path(args.output)
        for seed_index, base_seed in enumerate(seeds):
            existing_rows = _read_rows(output)
            if _seed_complete(existing_rows, base_seed, policies, tasks):
                print(f"[final-eval] skip complete base_seed={base_seed}", flush=True)
                continue
            seed_rows = []
            for task in tasks:
                seed_rows.append(
                    _eval_teacher_row(
                        base_config=base_config,
                        args=args,
                        task=task,
                        base_seed=base_seed,
                        seed_index=seed_index,
                        episodes=int(args.episodes),
                        device=device,
                    )
                )
                print(
                    f"[final-eval] seed={base_seed} policy=teacher task={task} "
                    f"fall_rate={seed_rows[-1]['fall_rate']:.4f} avg_distance={seed_rows[-1]['avg_distance']:.4f}",
                    flush=True,
                )
            for spec in student_specs:
                config, student = loaded_students[spec["policy"]]
                for task in tasks:
                    seed_rows.append(
                        _eval_student_row(
                            spec=spec,
                            config=config,
                            student=student,
                            args=args,
                            task=task,
                            base_seed=base_seed,
                            seed_index=seed_index,
                            episodes=int(args.episodes),
                            device=device,
                        )
                    )
                    print(
                        f"[final-eval] seed={base_seed} policy={spec['policy']} task={task} "
                        f"fall_rate={seed_rows[-1]['fall_rate']:.4f} avg_distance={seed_rows[-1]['avg_distance']:.4f}",
                        flush=True,
                    )
            _append_seed_rows(output, seed_rows)
            for row in seed_rows:
                _log_row(row, prefix="final_eval")
            if wandb.run is not None:
                wandb.log({"completed_base_seed": int(base_seed), "completed_seed_index": int(seed_index)})
            print(f"[final-eval] wrote base_seed={base_seed} rows={len(seed_rows)} csv={output}", flush=True)
        print(f"[final-eval] saved_summary={resolve_path(args.output)}", flush=True)
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
