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

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.env import make_env
from rl.student_policy import load_student_checkpoint, normalize_teacher_obs, render_student_depth, resolve_path
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces
from rl.train_teacher import load_config, make_single_env


def _task_config(config: dict, task: str) -> tuple[str, Path, int]:
    seed_offset = {"old": 0, "shift": 100_000, "hard_heldout": 200_000}
    task_names = {
        "old": "random_slope_old",
        "shift": "random_slope_shift",
        "hard_heldout": "random_slope_shift_hard_heldout",
    }
    config_keys = {
        "old": "old_config",
        "shift": "shift_config",
        "hard_heldout": "hard_heldout_config",
    }
    if task not in task_names:
        raise ValueError("--task must be one of: old, shift, hard_heldout")
    config_path = config.get("eval", {}).get(config_keys[task])
    if not config_path:
        raise KeyError(f"eval.{config_keys[task]} is not configured")
    return task_names[task], resolve_path(config_path), seed_offset[task]


def _make_vecnormalize(teacher_config: dict, vecnormalize_path: Path, seed: int) -> VecNormalize:
    env_cfg = {**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0}
    vec_env = DummyVecEnv([make_single_env(env_cfg)])
    vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _append_capsule(scene, p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray, radius: float = 0.008) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.eye(3, dtype=np.float32).reshape(-1),
        rgba.astype(np.float32),
    )
    connector_fn = getattr(mujoco, "mjv_makeConnector", None)
    if connector_fn is not None:
        connector_fn(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            float(p0[0]),
            float(p0[1]),
            float(p0[2]),
            float(p1[0]),
            float(p1[1]),
            float(p1[2]),
        )
    else:
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p0.astype(np.float64), p1.astype(np.float64))


def _frustum_corners(env, camera_name: str, distance: float, aspect: float) -> tuple[np.ndarray, list[np.ndarray]]:
    if env.renderer is None:
        raise RuntimeError("renderer is not initialized")
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"camera not found in MuJoCo model: {camera_name}")
    pos = np.asarray(env.data.cam_xpos[cam_id], dtype=np.float64)
    rot = np.asarray(env.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    fovy = float(env.model.cam_fovy[cam_id]) * np.pi / 180.0
    half_h = float(distance) * np.tan(0.5 * fovy)
    half_w = half_h * float(aspect)
    # MuJoCo camera frame follows the OpenGL convention: forward is local -Z.
    local_corners = [
        np.array([-half_w, -half_h, -distance], dtype=np.float64),
        np.array([half_w, -half_h, -distance], dtype=np.float64),
        np.array([half_w, half_h, -distance], dtype=np.float64),
        np.array([-half_w, half_h, -distance], dtype=np.float64),
    ]
    return pos, [pos + rot @ corner for corner in local_corners]


def _append_depth_camera_frustum(env, camera_name: str, near: float, far: float, aspect: float) -> None:
    if env.renderer is None:
        return
    pos, near_corners = _frustum_corners(env, camera_name, distance=near, aspect=aspect)
    _, far_corners = _frustum_corners(env, camera_name, distance=far, aspect=aspect)
    outline_rgba = np.array([0.10, 0.85, 0.95, 0.90], dtype=np.float32)
    fill_rgba = np.array([0.62, 0.93, 1.00, 0.18], dtype=np.float32)

    for near_corner, far_corner in zip(near_corners, far_corners, strict=True):
        _append_capsule(env.renderer.scene, near_corner, far_corner, outline_rgba, radius=0.006)
    for i in range(4):
        _append_capsule(env.renderer.scene, near_corners[i], near_corners[(i + 1) % 4], outline_rgba, radius=0.004)
        _append_capsule(env.renderer.scene, far_corners[i], far_corners[(i + 1) % 4], outline_rgba, radius=0.006)

    # A light ray grid makes the clipped near/far volume readable without changing
    # the terrain render. It is derived from the exact depth input camera config.
    for u in np.linspace(0.125, 0.875, 4):
        for v in np.linspace(0.125, 0.875, 4):
            near_point = (
                near_corners[0] * (1.0 - u) * (1.0 - v)
                + near_corners[1] * u * (1.0 - v)
                + near_corners[2] * u * v
                + near_corners[3] * (1.0 - u) * v
            )
            far_point = (
                far_corners[0] * (1.0 - u) * (1.0 - v)
                + far_corners[1] * u * (1.0 - v)
                + far_corners[2] * u * v
                + far_corners[3] * (1.0 - u) * v
            )
            _append_capsule(env.renderer.scene, near_point, far_point, fill_rgba, radius=0.012)

    center_near = np.mean(np.asarray(near_corners), axis=0)
    center_far = np.mean(np.asarray(far_corners), axis=0)
    _append_capsule(env.renderer.scene, pos, center_near, outline_rgba, radius=0.004)
    _append_capsule(env.renderer.scene, center_near, center_far, fill_rgba, radius=0.018)


def _render_main_with_frustum(
    env,
    width: int,
    height: int,
    view_camera: str,
    depth_camera: str,
    near: float,
    far: float,
    aspect: float,
) -> np.ndarray:
    if env.renderer is None or env.renderer_width != width or env.renderer_height != height:
        if env.renderer is not None:
            env.renderer.close()
        env.renderer = mujoco.Renderer(env.model, height=height, width=width)
        env.renderer_width = width
        env.renderer_height = height
    env._refresh_preview_highlight()
    if view_camera == "tracking":
        camera_spec = env._make_tracking_camera()
    elif view_camera == "terrain_overview":
        camera_spec = env._make_terrain_overview_camera()
    else:
        camera_spec = view_camera
    env.renderer.update_scene(env.data, camera=camera_spec)
    env._append_stair_edge_markers()
    env._append_scandot_markers()
    _append_depth_camera_frustum(env, depth_camera, near=near, far=far, aspect=aspect)
    return env.renderer.render()


def _resize_nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    repeat_y = max(1, int(np.ceil(height / rgb.shape[0])))
    repeat_x = max(1, int(np.ceil(width / rgb.shape[1])))
    resized = np.repeat(np.repeat(rgb, repeat_y, axis=0), repeat_x, axis=1)
    return resized[:height, :width]


def _render_depth_camera_rgb(renderer: mujoco.Renderer, env, camera_name: str) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_name)
    return renderer.render()


def _depth_to_rgb(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    img = np.asarray(depth[0], dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    # Invert so closer terrain is brighter and easier to inspect.
    grey = ((1.0 - img) * 255.0).astype(np.uint8)
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    rgb = np.rot90(rgb, k=1)
    return _resize_nearest(rgb, height, width)


def _compose_frame(main: np.ndarray, camera_rgb: np.ndarray, depth: np.ndarray, info: dict, side_width: int) -> np.ndarray:
    h, w = main.shape[:2]
    top_h = int(round(h * 0.62))
    bottom_h = h - top_h - 8
    camera_rgb = np.rot90(camera_rgb, k=1)
    camera_panel = _resize_nearest(camera_rgb, top_h, side_width)
    depth_rgb = _depth_to_rgb(depth, bottom_h, side_width)
    divider = np.full((8, side_width, 3), 255, dtype=np.uint8)
    side = np.concatenate([camera_panel, divider, depth_rgb], axis=0)
    gap = np.full((h, 8, 3), 18, dtype=np.uint8)
    frame = np.concatenate([main, gap, side], axis=1)
    # Lightweight visual reference bars without depending on PIL/OpenCV.
    frame[:4, :, :] = 255
    frame[:, w : w + 8, :] = 255
    distance = min(max(float(info.get("x_position", 0.0)) / max(float(info.get("success_target_x", 4.0)), 1e-6), 0.0), 1.0)
    bar_width = int(distance * (frame.shape[1] - 20))
    frame[h - 12 : h - 6, 10 : 10 + bar_width, :] = np.array([40, 220, 90], dtype=np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-vecnormalize", type=Path, default=None)
    parser.add_argument("--task", choices=["old", "shift", "hard_heldout"], default="hard_heldout")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "student_paper_like_eval" / "student_depth_debug_hard_heldout.mp4")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--side-width", type=int, default=312)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--view-camera", type=str, default="tracking")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = load_student_config(args.config)
    config.setdefault("wandb", {})["enabled"] = False
    device = device_from_config(config, args.device)
    task_name, teacher_cfg_path, seed_offset = _task_config(config, args.task)
    task_seed = int(args.seed) + seed_offset
    teacher_config = load_config(teacher_cfg_path)
    depth_cfg = dict(config["student"]["depth"])
    depth_camera = str(depth_cfg.get("camera", "front_camera"))
    depth_width = int(depth_cfg["width"])
    depth_height = int(depth_cfg["height"])
    depth_near = float(depth_cfg.get("near", 0.05))
    depth_far = float(depth_cfg.get("far", 2.0))
    depth_aspect = float(depth_width) / max(float(depth_height), 1.0)
    vecnormalize_path = resolve_path(args.teacher_vecnormalize or config["teacher"]["vecnormalize"])
    vecnormalize = _make_vecnormalize(teacher_config, vecnormalize_path, task_seed)
    observation_space, action_space = make_student_env_spaces(config, seed=task_seed)
    checkpoint = resolve_path(
        args.student_checkpoint
        or config.get("eval", {}).get("student_checkpoint", config["dagger"].get("best_checkpoint", "artifacts/student_paper_like_dagger_resume_from_round3/best_student.pt"))
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint}")
    student = load_student_checkpoint(checkpoint, config, observation_space, action_space, device)
    env = make_env({**teacher_config["env"], "seed": task_seed, "reset_noise_scale": 0.0})
    if hasattr(env, "set_preview_highlight"):
        env.set_preview_highlight(False)
        env.set_scandot_overlay(True)
        env.set_stair_edge_overlay(True)
    frames: list[np.ndarray] = []
    obs, _ = env.reset(options={"terrain_type": "random_slope_up_down"})
    hidden = None
    episode_start = True
    done = False
    info = {"x_position": 0.0}
    steps = 0
    side_top_height = int(round(int(args.height) * 0.62))
    rgb_renderer = mujoco.Renderer(env.model, height=side_top_height, width=int(args.side_width))
    try:
        while not done and steps < int(args.max_steps):
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
            main_frame = _render_main_with_frustum(
                env,
                width=int(args.width),
                height=int(args.height),
                view_camera=str(args.view_camera),
                depth_camera=depth_camera,
                near=depth_near,
                far=depth_far,
                aspect=depth_aspect,
            )
            camera_rgb = _render_depth_camera_rgb(rgb_renderer, env, depth_camera)
            frames.append(_compose_frame(main_frame, camera_rgb, depth, info, side_width=int(args.side_width)))
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_start = done
            steps += 1
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output, frames, fps=int(args.fps))
        print(
            f"[student-depth-video] saved={output} task={task_name} checkpoint={checkpoint} "
            f"steps={steps} depth_camera={depth_camera} depth_size={depth_width}x{depth_height} "
            f"near={depth_near:.3f} far={depth_far:.3f} normalize={bool(depth_cfg.get('normalize', True))}",
            flush=True,
        )
    finally:
        rgb_renderer.close()
        env.close()
        vecnormalize.close()


if __name__ == "__main__":
    main()
