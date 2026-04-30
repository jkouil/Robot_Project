from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


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

import mujoco

from rl.env import make_env


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    padding = 9
    spacing = 4
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    box_width = max(widths, default=0) + 2 * padding
    box_height = sum(heights) + max(0, len(lines) - 1) * spacing + 2 * padding
    draw.rounded_rectangle((10, 10, 10 + box_width, 10 + box_height), radius=8, fill=(0, 0, 0, 170))
    y = 10 + padding
    for line, height in zip(lines, heights):
        draw.text((10 + padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += height + spacing
    return np.asarray(image)


def _render_with_camera(env, camera: mujoco.MjvCamera, width: int, height: int) -> np.ndarray:
    if env.renderer is None or env.renderer_width != width or env.renderer_height != height:
        if env.renderer is not None:
            env.renderer.close()
        env.renderer = mujoco.Renderer(env.model, height=height, width=width)
        env.renderer_width = width
        env.renderer_height = height
    env._refresh_preview_highlight()
    env.renderer.update_scene(env.data, camera=camera)
    env._append_stair_edge_markers()
    env._append_scandot_markers()
    return env.renderer.render()


def _make_closeup_camera(x: float, y: float, z: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([x, y, max(0.08, z + 0.04)], dtype=np.float64)
    camera.distance = 0.82
    camera.azimuth = 90.0
    camera.elevation = -13.0
    return camera


def _make_topdown_camera(infos: list[dict[str, float]]) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if infos:
        x_min = min(info["x_start"] for info in infos)
        x_max = max(info["x_end"] for info in infos)
        y_min = min(info["y_center"] - info["half_width"] for info in infos)
        y_max = max(info["y_center"] + info["half_width"] for info in infos)
        camera.lookat[:] = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.08], dtype=np.float64)
        camera.distance = max(3.5, 0.72 * (x_max - x_min))
    else:
        camera.lookat[:] = np.array([2.35, 0.0, 0.08], dtype=np.float64)
        camera.distance = 3.95
    camera.azimuth = 90.0
    camera.elevation = -89.0
    return camera


def _segment_infos(env) -> list[dict[str, float]]:
    infos = []
    for geom_id in env._active_bump_geom_order:
        geom_pos = np.asarray(env.data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(env.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        sx, sy, sz = [float(value) for value in env.model.geom_size[geom_id]]
        pitch = float(np.arctan2(geom_rot[0, 2], geom_rot[0, 0]))
        grade_deg = float(np.rad2deg(-pitch))
        cos_pitch = float(np.cos(pitch))
        sin_pitch = float(np.sin(pitch))
        x_start = float(geom_pos[0] - cos_pitch * sx - sin_pitch * sz)
        x_end = float(geom_pos[0] + cos_pitch * sx + sin_pitch * sz)
        top_point = geom_pos + geom_rot @ np.array([0.0, 0.0, sz], dtype=np.float64)
        top_normal = geom_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        center_top_z = float(top_point[2])
        infos.append(
            {
                "x_start": x_start,
                "x_end": x_end,
                "x_center": float(geom_pos[0]),
                "y_center": float(geom_pos[1]),
                "half_width": sy,
                "top_z": center_top_z,
                "normal_z": float(top_normal[2]),
                "grade_deg": grade_deg,
            }
        )
    return infos


def _save_scene(
    *,
    env,
    output_dir: Path,
    label: str,
    terrain_level: int,
    width: int,
    height: int,
) -> None:
    infos = _segment_infos(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    grades = [info["grade_deg"] for info in infos]
    y_offsets = [info["y_center"] for info in infos]
    overview = env.render_frame(width=width, height=height, camera="terrain_overview")
    overview_lines = [
        f"{label} random_slope_up_down  level={terrain_level}",
        f"segments={len(infos)}  grade_deg min={min(grades):+.1f} max={max(grades):+.1f}" if grades else "segments=0",
        f"y offset min={min(y_offsets):+.2f} m max={max(y_offsets):+.2f} m" if y_offsets else "y offset unavailable",
        "grade sequence: " + ", ".join(f"{grade:+.1f}" for grade in grades[:12]),
        "white edges are the actual physical slope geoms",
    ]
    imageio.imwrite(output_dir / f"{label}_overview_level_{terrain_level}.png", _annotate(overview, overview_lines))

    topdown = _render_with_camera(env, _make_topdown_camera(infos), width=width, height=height)
    topdown_lines = [
        f"{label} topdown shifted slopes  level={terrain_level}",
        "y sequence: " + ", ".join(f"{offset:+.2f}" for offset in y_offsets[:12]),
        "straight y=0 is the center horizontal line of the original course",
    ]
    imageio.imwrite(output_dir / f"{label}_overview_topdown_level_{terrain_level}.png", _annotate(topdown, topdown_lines))

    for seam_idx, (left, right) in enumerate(zip(infos, infos[1:]), start=1):
        seam_x = 0.5 * (left["x_end"] + right["x_start"])
        seam_y = 0.5 * (left["y_center"] + right["y_center"])
        seam_z = max(left["top_z"], right["top_z"])
        camera = _make_closeup_camera(seam_x, seam_y, seam_z)
        frame = _render_with_camera(env, camera, width=width, height=height)
        left_grade = left["grade_deg"]
        right_grade = right["grade_deg"]
        delta = right_grade - left_grade
        shift_y = right["y_center"] - left["y_center"]
        lines = [
            f"{label} seam {seam_idx:02d}  x={seam_x:.2f} m  y={seam_y:+.2f} m",
            f"left slope: {left_grade:+.1f} deg",
            f"right slope: {right_grade:+.1f} deg",
            f"delta right-left: {delta:+.1f} deg",
            f"y shift right-left: {shift_y:+.2f} m",
        ]
        imageio.imwrite(output_dir / f"{label}_seam_{seam_idx:02d}_x_{seam_x:.2f}.png", _annotate(frame, lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "random_slope_train_eval_smoke",
    )
    parser.add_argument("--terrain-level", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = _load_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    env_cfg = dict(config["env"])
    env_cfg["reset_noise_scale"] = 0.0
    env_cfg["terrain_generation"] = dict(env_cfg.get("terrain_generation", {}))

    base_seed = int(env_cfg.get("seed", 0))
    render_scandots = bool(config.get("eval", {}).get("render_scandots", True))
    render_edges = bool(config.get("eval", {}).get("render_stair_edges", True))
    highlight_preview = bool(config.get("eval", {}).get("highlight_preview_terrain", False))

    for label, seed in (("train", base_seed), ("eval", base_seed + 1)):
        scene_cfg = {**env_cfg, "seed": seed}
        env = make_env(scene_cfg)
        try:
            env.set_preview_highlight(highlight_preview)
            env.set_scandot_overlay(render_scandots)
            env.set_stair_edge_overlay(render_edges)
            env.reset(options={"terrain_type": "random_slope_up_down", "terrain_level": int(args.terrain_level)})
            _save_scene(
                env=env,
                output_dir=args.output_dir,
                label=label,
                terrain_level=int(args.terrain_level),
                width=int(args.width),
                height=int(args.height),
            )
            print(f"saved {label} smoke pngs to {args.output_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
