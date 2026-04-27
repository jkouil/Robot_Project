from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from rl.env import make_env


DEFAULT_TERRAINS = [
    "random_slope_up_down",
    "slope_up_down",
    "stepping_stones",
    "stairs",
    "discrete_obstacles",
    "mixed_course",
]


def apply_egl_device_config(config: dict) -> None:
    eval_cfg = config.get("eval", {})
    egl_device_id = eval_cfg.get("egl_device_id")
    if egl_device_id is None:
        return
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device_id))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_egl_device_config(config)
    return config


def render_static_terrain_video(
    env,
    output_path: Path,
    frame_path: Path,
    fps: int,
    duration_s: float,
    camera: str,
    width: int,
    height: int,
) -> None:
    frame = env.render_frame(width=width, height=height, camera=camera)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame.copy() for _ in range(max(1, int(round(fps * duration_s))))]
    env.save_video(frames, output_path, fps=fps)
    import imageio.v2 as imageio

    imageio.imwrite(frame_path, frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/terrain_smoke"))
    parser.add_argument("--terrain-level", type=int, default=3)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--level-range", nargs=2, type=int, metavar=("MIN_LEVEL", "MAX_LEVEL"))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--camera", type=str, default="terrain_overview")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--terrains", nargs="*", default=DEFAULT_TERRAINS)
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = dict(config["env"])
    env_cfg["reset_noise_scale"] = 0.0
    env_cfg["seed"] = int(env_cfg.get("seed", 0)) + 999

    env = make_env(env_cfg)
    highlight_preview = bool(config.get("eval", {}).get("highlight_preview_terrain", False))
    render_scandots = bool(config.get("eval", {}).get("render_scandots", True))
    render_stair_edges = bool(config.get("eval", {}).get("render_stair_edges", True))
    env.set_preview_highlight(highlight_preview)
    env.set_scandot_overlay(render_scandots)
    env.set_stair_edge_overlay(render_stair_edges)
    curriculum_cfg = env_cfg.get("curriculum", {}) or {}
    min_level = int(curriculum_cfg.get("min_level", 0))
    max_level = int(curriculum_cfg.get("max_level", 5))
    if args.level_range is not None:
        level_lo, level_hi = args.level_range
        terrain_levels = list(range(level_lo, level_hi + 1))
    elif args.all_levels:
        terrain_levels = list(range(min_level, max_level + 1))
    else:
        terrain_levels = [args.terrain_level]
    try:
        for terrain_type in args.terrains:
            normalized = env._normalize_terrain_type(terrain_type)
            for terrain_level in terrain_levels:
                env.reset(options={"terrain_type": normalized, "terrain_level": terrain_level})
                video_path = args.output_dir / f"{normalized}_level_{terrain_level}.mp4"
                frame_path = args.output_dir / f"{normalized}_level_{terrain_level}.png"
                render_static_terrain_video(
                    env,
                    output_path=video_path,
                    frame_path=frame_path,
                    fps=args.fps,
                    duration_s=args.duration_s,
                    camera=args.camera,
                    width=args.width,
                    height=args.height,
                )
                print(f"saved {video_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
