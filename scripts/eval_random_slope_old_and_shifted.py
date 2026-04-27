from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys


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


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.gru_policy import ScandotGruPolicy  # noqa: F401 - required for custom policy loading
from rl.train_teacher import build_model, evaluate_model, is_recurrent_config, load_config, make_single_env


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _default_vecnormalize_for_model(model_path: Path) -> Path:
    return model_path.with_name(model_path.name.replace(".zip", "_vecnormalize.pkl"))


def _run_eval(
    *,
    config_path: Path,
    checkpoint_path: Path,
    vecnormalize_path: Path,
    episodes: int,
    seed: int,
    terrain_level: int | None,
    output_dir: Path,
) -> tuple[dict, dict]:
    config = load_config(config_path)
    eval_cfg = dict(config["eval"])
    eval_cfg["eval_episodes"] = int(episodes)
    env_cfg = {
        **config["env"],
        "seed": int(seed),
        "reset_noise_scale": 0.0,
    }
    eval_env = DummyVecEnv([make_single_env(env_cfg)])
    eval_env = VecNormalize.load(str(vecnormalize_path), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    base_vec = eval_env.venv
    if hasattr(base_vec, "env_method"):
        base_vec.env_method("set_fixed_terrain_type", "random_slope_up_down")
        if terrain_level is not None:
            base_vec.env_method("set_curriculum_level", int(terrain_level))

    device_name = str(config["train"].get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    model = build_model(config, eval_env, output_dir, device_name, int(seed))
    _, params, _ = load_from_zip_file(str(checkpoint_path), device=device_name, print_system_info=False)
    if params is None:
        raise FileNotFoundError(f"could not load parameters from checkpoint: {checkpoint_path}")
    model.set_parameters(params, exact_match=True, device=device_name)

    metrics = evaluate_model(model, eval_env, eval_cfg, recurrent=is_recurrent_config(config))
    terrain_generation = dict(config["env"].get("terrain_generation", {}) or {})
    metadata = {
        "config_path": str(config_path),
        "terrain_type": "random_slope_up_down",
        "terrain_level": "" if terrain_level is None else int(terrain_level),
        "corridor_half_width": config["env"].get("corridor_half_width"),
        "lateral_shift_max_fraction": terrain_generation.get("random_slope_lateral_shift_max_fraction", 0.0),
        "lateral_center_limit_fraction": terrain_generation.get("random_slope_lateral_center_limit_fraction", 0.0),
    }
    eval_env.close()
    return metrics, metadata


def _write_combined_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "checkpoint",
        "vecnormalize",
        "config_path",
        "episodes",
        "seed",
        "terrain_type",
        "terrain_level",
        "corridor_half_width",
        "lateral_shift_max_fraction",
        "lateral_center_limit_fraction",
        "fall_rate",
        "avg_distance",
        "max_distance",
        "avg_forward_velocity",
        "avg_return",
        "weak_success_rate",
        "strict_success_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "teacher_walk_preview_gru_random_slopes_no_corridor_iterative_finetune"
        / "v12"
        / "final_teacher.zip",
    )
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument(
        "--old-config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes_no_corridor_finetune.yaml",
    )
    parser.add_argument(
        "--shift-config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes_shifted_no_corridor.yaml",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--terrain-level", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "random_slope_old_vs_shifted_eval" / "summary.csv",
    )
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    checkpoint_path = _resolve(args.checkpoint)
    vecnormalize_path = _resolve(args.vecnormalize) if args.vecnormalize is not None else _default_vecnormalize_for_model(checkpoint_path)
    old_config_path = _resolve(args.old_config)
    shift_config_path = _resolve(args.shift_config)
    output_path = _resolve(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not vecnormalize_path.exists():
        raise FileNotFoundError(f"vecnormalize not found: {vecnormalize_path}")

    tasks = [
        ("random_slope_old", old_config_path, int(args.seed)),
        ("random_slope_shift", shift_config_path, int(args.seed) + 100_000),
    ]
    rows = []
    for task_name, config_path, seed in tasks:
        metrics, metadata = _run_eval(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            vecnormalize_path=vecnormalize_path,
            episodes=int(args.episodes),
            seed=seed,
            terrain_level=args.terrain_level,
            output_dir=output_path.parent,
        )
        row = {
            "task": task_name,
            "checkpoint": str(checkpoint_path),
            "vecnormalize": str(vecnormalize_path),
            "episodes": int(args.episodes),
            "seed": int(seed),
            **metadata,
            **metrics,
        }
        rows.append(row)
        print(f"[{task_name}]")
        print(f"config={config_path}")
        print(f"episodes={int(args.episodes)} seed={seed} terrain_level={args.terrain_level}")
        print(f"fall_rate={metrics['fall_rate']:.4f}")
        print(f"avg_distance={metrics['avg_distance']:.4f}")
        print(f"max_distance={metrics['max_distance']:.4f}")
        print(f"avg_forward_velocity={metrics['avg_forward_velocity']:.4f}")
        print(f"avg_return={metrics['avg_return']:.4f}")
        print(f"weak_success_rate={metrics['weak_success_rate']:.4f}")
        print(f"strict_success_rate={metrics['strict_success_rate']:.4f}")

    _write_combined_csv(output_path, rows)
    print(f"saved_summary={output_path}")


if __name__ == "__main__":
    main()
