from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import types

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# Work around a broken tensorboard/tensorflow combination in the current env.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import recursive_setattr
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from rl.train_teacher import (
    _set_eval_curriculum_level,
    build_policy_kwargs,
    is_recurrent_config,
    load_config,
    make_env,
    make_single_env,
    record_curriculum_showcase,
    record_video,
)
from rl.paper_bptt import PaperBpttRecurrentPPO


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if path_str is None:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _default_model_path(out_dir: Path) -> Path:
    best_path = out_dir / "best_teacher.zip"
    if best_path.exists():
        return best_path
    final_path = out_dir / "final_teacher.zip"
    if final_path.exists():
        return final_path
    raise FileNotFoundError(f"Neither {best_path} nor {final_path} exists.")


def _default_vecnorm_path(out_dir: Path, model_path: Path) -> Path:
    if "best_teacher" in model_path.name:
        candidate = out_dir / "best_teacher_vecnormalize.pkl"
    else:
        candidate = out_dir / "final_teacher_vecnormalize.pkl"
    if candidate.exists():
        return candidate
    fallback = out_dir / "best_teacher_vecnormalize.pkl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No VecNormalize stats file found in output directory.")


def _load_model(config: dict, model_path: Path, eval_env: VecNormalize):
    recurrent = is_recurrent_config(config)
    train_cfg = config["train"]
    device_name = str(train_cfg.get("device", "cpu"))
    policy_kwargs = build_policy_kwargs(config)
    custom_objects = {
        "policy_kwargs": policy_kwargs,
        "learning_rate": float(train_cfg["learning_rate"]),
        "lr_schedule": lambda _: float(train_cfg["learning_rate"]),
        "clip_range": lambda _: float(train_cfg["clip_coef"]),
    }
    if recurrent:
        recurrent_cell = str(train_cfg.get("recurrent_cell", "lstm")).lower()
        truncated_bptt_steps = int(train_cfg.get("truncated_bptt_steps", 0) or 0)
        recurrent_ppo_cls = PaperBpttRecurrentPPO if truncated_bptt_steps > 0 else RecurrentPPO
        policy_name = "MultiInputLstmPolicy"
        if config["env"].get("observation_mode") == "teacher_dict" and recurrent_cell == "gru":
            from rl.gru_policy import ScandotGruPolicy

            policy_name = ScandotGruPolicy
        if recurrent_ppo_cls is PaperBpttRecurrentPPO:
            data, params, pytorch_variables = load_from_zip_file(
                str(model_path),
                device=device_name,
                custom_objects={**custom_objects, "policy_class": policy_name},
                print_system_info=False,
            )
            assert data is not None and params is not None
            model = PaperBpttRecurrentPPO(
                policy=data["policy_class"],
                env=eval_env,
                device=device_name,
                recurrent_sequence_length=truncated_bptt_steps,
                _init_setup_model=False,
            )
            model.__dict__.update(data)
            model._setup_model()
            model.set_parameters(params, exact_match=True, device=device_name)
            if pytorch_variables is not None:
                for name, value in pytorch_variables.items():
                    recursive_setattr(model, f"{name}.data", value.data)
        else:
            model = recurrent_ppo_cls.load(
                str(model_path),
                env=eval_env,
                device=device_name,
                custom_objects={**custom_objects, "policy_class": policy_name},
            )
    else:
        model = PPO.load(
            str(model_path),
            env=eval_env,
            device=device_name,
            custom_objects=custom_objects,
        )
    return model, recurrent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnormalize", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--showcase-all-levels", action="store_true")
    parser.add_argument("--prefix", type=str, default="offline_export")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["train"]
    eval_cfg = dict(config["eval"])
    if args.max_steps is not None:
        eval_cfg["video_max_steps"] = int(args.max_steps)
    if args.video_width is not None:
        eval_cfg["video_width"] = int(args.video_width)
    if args.video_height is not None:
        eval_cfg["video_height"] = int(args.video_height)
    if args.video_fps is not None:
        eval_cfg["video_fps"] = int(args.video_fps)
    out_dir = _resolve_path(str(train_cfg.get("output_dir", "artifacts/teacher_walk")), REPO_ROOT / "artifacts/teacher_walk")
    model_path = _resolve_path(args.model, _default_model_path(out_dir))
    vecnorm_path = _resolve_path(args.vecnormalize, _default_vecnorm_path(out_dir, model_path))

    seed = int(train_cfg.get("seed", 0))
    eval_dummy_env = DummyVecEnv([make_single_env({**config["env"], "seed": seed + 101, "reset_noise_scale": 0.0})])
    eval_env = VecNormalize.load(str(vecnorm_path), eval_dummy_env)
    eval_env.training = False
    eval_env.norm_reward = False

    video_env = make_env({**config["env"], "seed": seed + 102, "reset_noise_scale": 0.0})
    if args.level is not None:
        _set_eval_curriculum_level(eval_env, args.level)
        if hasattr(video_env, "set_curriculum_level"):
            video_env.set_curriculum_level(args.level)

    model, recurrent = _load_model(config, model_path, eval_env)
    print(f"using_mujoco_gl={os.environ.get('MUJOCO_GL')} pyopengl_platform={os.environ.get('PYOPENGL_PLATFORM')}", flush=True)

    try:
        if args.showcase_all_levels:
            export_dir = _resolve_path(args.output, out_dir / "offline_videos")
            results = record_curriculum_showcase(
                model,
                video_env,
                eval_env,
                export_dir,
                eval_cfg,
                recurrent=recurrent,
                prefix=args.prefix,
            )
            for item in results:
                print(
                    f"saved level={item['level']} video={item['video_path']} "
                    f"distance={item['distance']:.2f} success={item['strict_success']}",
                    flush=True,
                )
            return

        output_path = _resolve_path(args.output, out_dir / f"{args.prefix}.mp4")
        reset_options = None
        if args.level is not None:
            reset_options = {
                "terrain_level": args.level,
                "terrain_type": eval_cfg.get("showcase_terrain_type"),
            }
        metrics = record_video(
            model,
            video_env,
            eval_env,
            output_path,
            eval_cfg,
            recurrent=recurrent,
            reset_options=reset_options,
        )
        print(
            f"saved_video={output_path} distance={metrics['distance']:.2f} "
            f"avg_speed={metrics['avg_forward_velocity']:.2f} "
            f"success={metrics['strict_success']}",
            flush=True,
        )
    finally:
        eval_env.close()
        video_env.close()


if __name__ == "__main__":
    main()
