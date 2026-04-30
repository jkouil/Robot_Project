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

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.save_util import load_from_zip_file

from rl.env import make_env
from rl.train_teacher import build_model, evaluate_model, is_recurrent_config, load_config, make_single_env, record_video
from rl.gru_policy import ScandotGruPolicy  # noqa: F401 - required for custom policy loading
from rl.paper_bptt import PaperBpttRecurrentPPO


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _default_vecnormalize_for_model(model_path: Path) -> Path:
    return model_path.with_name(model_path.name.replace(".zip", "_vecnormalize.pkl"))


def _write_summary_csv(path: Path, metrics: dict, model_path: Path, vecnormalize_path: Path, episodes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_path",
        "vecnormalize_path",
        "episodes",
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
        writer.writerow(
            {
                "model_path": str(model_path),
                "vecnormalize_path": str(vecnormalize_path),
                "episodes": int(episodes),
                "fall_rate": metrics["fall_rate"],
                "avg_distance": metrics["avg_distance"],
                "max_distance": metrics["max_distance"],
                "avg_forward_velocity": metrics["avg_forward_velocity"],
                "avg_return": metrics["avg_return"],
                "weak_success_rate": metrics["weak_success_rate"],
                "strict_success_rate": metrics["strict_success_rate"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes_finetune_conservative.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "teacher_walk_preview_gru_random_slopes_iterative_finetune"
        / "v2"
        / "final_teacher.zip",
    )
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--terrain-level", type=int, default=None)
    parser.add_argument("--terrain-type", type=str, default="random_slope_up_down")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "random_slope_checkpoint_eval" / "eval_summary.csv",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    checkpoint_path = _resolve(args.checkpoint)
    vecnormalize_path = _resolve(args.vecnormalize) if args.vecnormalize is not None else _default_vecnormalize_for_model(checkpoint_path)
    output_path = _resolve(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not vecnormalize_path.exists():
        raise FileNotFoundError(f"vecnormalize not found: {vecnormalize_path}")

    eval_cfg = dict(config["eval"])
    eval_cfg["eval_episodes"] = int(args.episodes)
    env_cfg = {
        **config["env"],
        "seed": int(args.seed),
        "reset_noise_scale": 0.0,
    }
    eval_env = DummyVecEnv([make_single_env(env_cfg)])
    eval_env = VecNormalize.load(str(vecnormalize_path), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    base_vec = eval_env.venv
    if hasattr(base_vec, "env_method"):
        base_vec.env_method("set_fixed_terrain_type", args.terrain_type)
        if args.terrain_level is not None:
            base_vec.env_method("set_curriculum_level", int(args.terrain_level))

    device_name = config["train"].get("device", "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    model = build_model(config, eval_env, output_path.parent, device_name, int(args.seed))
    _, params, _ = load_from_zip_file(str(checkpoint_path), device=device_name, print_system_info=False)
    if params is None:
        raise FileNotFoundError(f"could not load parameters from checkpoint: {checkpoint_path}")
    model.set_parameters(params, exact_match=True, device=device_name)
    metrics = evaluate_model(model, eval_env, eval_cfg, recurrent=is_recurrent_config(config))
    _write_summary_csv(output_path, metrics, checkpoint_path, vecnormalize_path, int(args.episodes))

    print(f"model={checkpoint_path}")
    print(f"vecnormalize={vecnormalize_path}")
    print(f"episodes={int(args.episodes)} terrain_type={args.terrain_type} terrain_level={args.terrain_level}")
    print(f"fall_rate={metrics['fall_rate']:.4f}")
    print(f"avg_distance={metrics['avg_distance']:.4f}")
    print(f"max_distance={metrics['max_distance']:.4f}")
    print(f"avg_forward_velocity={metrics['avg_forward_velocity']:.4f}")
    print(f"avg_return={metrics['avg_return']:.4f}")
    print(f"weak_success_rate={metrics['weak_success_rate']:.4f}")
    print(f"strict_success_rate={metrics['strict_success_rate']:.4f}")
    print(f"saved_summary={output_path}")

    if args.record_video:
        video_path = _resolve(args.video_path) if args.video_path is not None else output_path.with_suffix(".mp4")
        video_env = make_env({**env_cfg, "seed": int(args.seed) + 999})
        video_env.set_fixed_terrain_type(args.terrain_type)
        if args.terrain_level is not None:
            video_env.set_curriculum_level(int(args.terrain_level))
        video_metrics = record_video(
            model,
            video_env,
            eval_env,
            video_path,
            eval_cfg,
            recurrent=is_recurrent_config(config),
            reset_options={"terrain_type": args.terrain_type, "terrain_level": args.terrain_level}
            if args.terrain_level is not None
            else {"terrain_type": args.terrain_type},
        )
        print(f"saved_video={video_path}")
        print(f"video_distance={video_metrics['distance']:.4f}")
        video_env.close()

    eval_env.close()


if __name__ == "__main__":
    main()
