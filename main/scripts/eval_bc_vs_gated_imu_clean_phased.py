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


FIELDNAMES = [
    "phase",
    "policy",
    "task",
    "config",
    "checkpoint",
    "episodes",
    "seed",
    "fall_rate",
    "avg_distance",
    "max_distance",
    "avg_forward_velocity",
    "avg_return",
    "weak_success_rate",
    "strict_success_rate",
    "eval_score",
    "beta_value",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
]

TASKS = {
    "random_slope_shift": ("shift_config", 100_000),
    "random_slope_shift_hard_heldout": ("hard_heldout_config", 200_000),
}


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


def _append_row(path: Path, row: dict) -> None:
    rows = _read_rows(path)
    rows.append({key: row.get(key, "") for key in FIELDNAMES})
    _write_rows(path, rows)


def _row_exists(rows: list[dict], phase: str, policy: str, task: str) -> bool:
    return any(row.get("phase") == phase and row.get("policy") == policy and row.get("task") == task for row in rows)


def _to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value == "" else float(value)


def _weighted_average(rows: list[dict], key: str) -> float:
    total_episodes = sum(int(row["episodes"]) for row in rows)
    return sum(_to_float(row, key) * int(row["episodes"]) for row in rows) / max(total_episodes, 1)


def _total_row(rows: list[dict], policy: str, task: str) -> dict:
    episodes = sum(int(row["episodes"]) for row in rows)
    fall_rate = _weighted_average(rows, "fall_rate")
    avg_distance = _weighted_average(rows, "avg_distance")
    avg_forward_velocity = _weighted_average(rows, "avg_forward_velocity")
    eval_score = fall_rate - 0.05 * avg_distance
    latest = rows[-1]
    return {
        "phase": f"total_{episodes}",
        "policy": policy,
        "task": task,
        "config": latest["config"],
        "checkpoint": latest["checkpoint"],
        "episodes": episodes,
        "seed": "",
        "fall_rate": fall_rate,
        "avg_distance": avg_distance,
        "max_distance": max(_to_float(row, "max_distance") for row in rows),
        "avg_forward_velocity": avg_forward_velocity,
        "avg_return": _weighted_average(rows, "avg_return"),
        "weak_success_rate": _weighted_average(rows, "weak_success_rate"),
        "strict_success_rate": _weighted_average(rows, "strict_success_rate"),
        "eval_score": eval_score,
        "beta_value": latest.get("beta_value", ""),
        "alpha_mean": latest.get("alpha_mean", ""),
        "alpha_std": latest.get("alpha_std", ""),
        "alpha_min": latest.get("alpha_min", ""),
        "alpha_max": latest.get("alpha_max", ""),
    }


def _model_specs(args) -> list[dict]:
    if args.model:
        specs = []
        for item in args.model:
            parts = item.split("=", 2)
            if len(parts) != 3 or not all(parts):
                raise ValueError("--model must have format label=config_path=checkpoint_path")
            label, config_path, checkpoint_path = parts
            specs.append(
                {
                    "policy": label,
                    "config": resolve_path(config_path),
                    "checkpoint": resolve_path(checkpoint_path),
                }
            )
        return specs
    return [
        {
            "policy": "PaperLike_BC",
            "config": resolve_path(args.paper_config),
            "checkpoint": resolve_path(args.paper_bc_checkpoint),
        },
        {
            "policy": "Gated_IMUClean_best_by_eval",
            "config": resolve_path(args.gated_config),
            "checkpoint": resolve_path(args.gated_best_by_eval),
        },
        {
            "policy": "Gated_IMUClean_best_by_val_loss",
            "config": resolve_path(args.gated_config),
            "checkpoint": resolve_path(args.gated_best_by_val_loss),
        },
    ]


def _load_model(spec: dict, device: torch.device, seed: int) -> tuple[dict, object]:
    config = load_student_config(spec["config"])
    config.setdefault("wandb", {})["enabled"] = False
    observation_space, action_space = make_student_env_spaces(config, seed=seed)
    student = load_student_checkpoint(spec["checkpoint"], config, observation_space, action_space, device)
    return config, student


def _eval_row(*, spec: dict, config: dict, student, task: str, episodes: int, seed: int, phase: str, device: torch.device) -> dict:
    config_key, task_seed_offset = TASKS[task]
    task_seed = int(seed) + int(task_seed_offset)
    teacher_config_path = resolve_path(config["eval"][config_key])
    teacher_vecnormalize_path = resolve_path(config["teacher"]["vecnormalize"])
    metrics = _eval_student(
        student=student,
        student_config=config,
        teacher_config_path=teacher_config_path,
        teacher_vecnormalize_path=teacher_vecnormalize_path,
        episodes=int(episodes),
        seed=task_seed,
        device=device,
        video_path=None,
    )
    gate_stats = get_student_gate_stats(student)
    eval_score = float(metrics["fall_rate"] - 0.05 * metrics["avg_distance"])
    return {
        "phase": phase,
        "policy": spec["policy"],
        "task": task,
        "config": str(spec["config"]),
        "checkpoint": str(spec["checkpoint"]),
        "episodes": int(episodes),
        "seed": task_seed,
        **metrics,
        "eval_score": eval_score,
        **gate_stats,
    }


def _write_total_rows(output: Path, phase_names: list[str], policies: list[str], tasks: list[str]) -> None:
    rows = _read_rows(output)
    changed = False
    for policy in policies:
        for task in tasks:
            phase_rows = [
                row
                for row in rows
                if row.get("policy") == policy and row.get("task") == task and row.get("phase") in phase_names
            ]
            if len(phase_rows) != len(phase_names):
                continue
            total_phase = f"total_{sum(int(row['episodes']) for row in phase_rows)}"
            if _row_exists(rows, total_phase, policy, task):
                continue
            rows.append({key: _total_row(phase_rows, policy, task).get(key, "") for key in FIELDNAMES})
            changed = True
    if changed:
        _write_rows(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--gated-config", type=Path, default=REPO_ROOT / "configs" / "student_residual_gated_imu_clean.yaml")
    parser.add_argument("--paper-bc-checkpoint", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_bc" / "best_student.pt")
    parser.add_argument("--gated-best-by-eval", type=Path, default=REPO_ROOT / "artifacts" / "student_residual_gated_imu_clean_bc" / "best_by_eval.pt")
    parser.add_argument("--gated-best-by-val-loss", type=Path, default=REPO_ROOT / "artifacts" / "student_residual_gated_imu_clean_bc" / "best_by_val_loss.pt")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Custom model spec, repeatable: label=config_path=checkpoint_path. If provided, overrides default BC/gated trio.",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "student_residual_gated_imu_clean_eval" / "bc_vs_gated_phased_20_30.csv")
    parser.add_argument("--phase1-episodes", type=int, default=20)
    parser.add_argument("--phase2-episodes", type=int, default=30)
    parser.add_argument("--phase2-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    if args.debug:
        args.phase1_episodes = 1
        args.phase2_episodes = 1
    device = device_from_config({"student": {"device": args.device}}, args.device)
    output = resolve_path(args.output)
    specs = _model_specs(args)
    for spec in specs:
        if not spec["checkpoint"].exists():
            raise FileNotFoundError(f"missing checkpoint for {spec['policy']}: {spec['checkpoint']}")
        if not spec["config"].exists():
            raise FileNotFoundError(f"missing config for {spec['policy']}: {spec['config']}")

    loaded = {}
    for spec in specs:
        loaded[spec["policy"]] = _load_model(spec, device, int(args.seed))

    tasks = ["random_slope_shift", "random_slope_shift_hard_heldout"]
    phases = [
        ("phase20", int(args.phase1_episodes), int(args.seed)),
        ("phase30", int(args.phase2_episodes), int(args.seed) + int(args.phase2_seed_offset)),
    ]
    if args.debug:
        phases = [("debug_phase1", int(args.phase1_episodes), int(args.seed)), ("debug_phase2", int(args.phase2_episodes), int(args.seed) + 10_000)]

    for phase_name, episodes, phase_seed in phases:
        for spec in specs:
            config, student = loaded[spec["policy"]]
            for task in tasks:
                rows = _read_rows(output)
                if _row_exists(rows, phase_name, spec["policy"], task):
                    print(f"[phased-eval] skip existing phase={phase_name} policy={spec['policy']} task={task}", flush=True)
                    continue
                row = _eval_row(
                    spec=spec,
                    config=config,
                    student=student,
                    task=task,
                    episodes=episodes,
                    seed=phase_seed,
                    phase=phase_name,
                    device=device,
                )
                _append_row(output, row)
                print(
                    f"[phased-eval] wrote phase={phase_name} policy={spec['policy']} task={task} "
                    f"episodes={episodes} fall_rate={row['fall_rate']:.4f} "
                    f"avg_distance={row['avg_distance']:.4f} avg_speed={row['avg_forward_velocity']:.4f}",
                    flush=True,
                )
    _write_total_rows(output, [phase[0] for phase in phases], [spec["policy"] for spec in specs], tasks)
    print(f"[phased-eval] saved_summary={output}", flush=True)


if __name__ == "__main__":
    main()
