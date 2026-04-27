from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from gymnasium import spaces

from rl.rewards import compute_teacher_walk_reward


JOINT_NAMES = [
    "fl_abduction",
    "fl_hip",
    "fl_knee",
    "fr_abduction",
    "fr_hip",
    "fr_knee",
    "rl_abduction",
    "rl_hip",
    "rl_knee",
    "rr_abduction",
    "rr_hip",
    "rr_knee",
]

DEFAULT_STAND = np.array(
    [0.05, 0.75, -1.35, -0.05, 0.75, -1.35, 0.05, 0.75, -1.35, -0.05, 0.75, -1.35],
    dtype=np.float64,
)


@dataclass
class EnvConfig:
    model_path: str
    control_dt: float = 0.02
    episode_length_s: float = 10.0
    action_scale: float = 0.35
    reset_noise_scale: float = 0.03
    command_min: float = 0.0
    command_max: float = 0.6
    command_yaw_min: float = 0.0
    command_yaw_max: float = 0.0
    healthy_z_min: float = 0.09
    healthy_torso_up_min: float = 0.45
    reward_weights: dict | None = None
    seed: int = 0
    use_privileged_terrain: bool = False
    terrain_obs_mode: str = "legacy"
    terrain_obs_height_reference: str = "base_z"
    terrain_patch_rows: int = 0
    terrain_patch_cols: int = 0
    terrain_patch_dx: float = 0.10
    terrain_patch_dy: float = 0.10
    terrain_patch_center_offset_x: float = 0.0
    terrain_patch_center_offset_y: float = 0.0
    scandot_layout: str = "custom"
    scandot_points: tuple[tuple[float, float], ...] = ()
    footstep_scandot_points: tuple[tuple[float, float], ...] = ()
    scandot_min_forward_offset: float = 0.18
    terrain_scan_x: tuple[float, ...] = (0.18, 0.32, 0.48, 0.66)
    terrain_scan_y: tuple[float, ...] = (-0.10, -0.04, 0.04, 0.10)
    terrain_scan_origin_z: float = 0.35
    terrain_scan_max_dist: float = 1.2
    corridor_half_width: float | None = None
    observation_mode: str = "flat"
    include_privileged_obs: bool = False
    observation_noise_std: float = 0.0
    privileged_randomization: dict | None = None
    curriculum: dict | None = None
    terrain_generation: dict | None = None


class PupperLikeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, int(round(config.control_dt / self.model.opt.timestep)))
        self.joint_qpos_adr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES])
        self.joint_qvel_adr = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES])
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.robot_body_ids = set(range(1, self.model.nbody))
        self.site_imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        self.front_hip_positions = np.array(
            [
                self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fl_abad_link")],
                self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fr_abad_link")],
            ],
            dtype=np.float64,
        )
        self.default_pose = DEFAULT_STAND.copy()
        self.command_velocity = 0.0
        self.command_yaw_rate = 0.0
        self.previous_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.step_count = 0
        self.max_steps = int(config.episode_length_s / config.control_dt)
        default_reward_weights = {
            "survival": 1.0,
            "command_tracking": 7.0,
            "absolute_work": 0.0001,
            "foot_jerk": 0.0001,
            "foot_drag": 0.0001,
            "collision": 1.0,
            "contact_force_threshold": 1.0,
            "collision_force_threshold": 0.1,
            "fall": 1.0,
        }
        self.reward_weights = dict(default_reward_weights)
        if config.reward_weights:
            self.reward_weights.update(config.reward_weights)
        self.np_random = np.random.default_rng(config.seed)
        self.renderer: mujoco.Renderer | None = None
        self.renderer_width: int | None = None
        self.renderer_height: int | None = None
        self.depth_renderer: mujoco.Renderer | None = None
        self.depth_renderer_width: int | None = None
        self.depth_renderer_height: int | None = None
        self._highlight_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            and mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith("bump_")
        ]
        self._precision_platform_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "precision_platform_geom",
        )
        self._precision_stone_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"precision_stone_{idx:02d}_geom")
            for idx in range(1, 25)
        ]
        self._highlight_geom_ids.extend([self._precision_platform_geom_id, *self._precision_stone_geom_ids])
        self._stair_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"stair_step_{idx:02d}_geom")
            for idx in range(1, 13)
        ]
        self._highlight_geom_ids.extend(self._stair_geom_ids)
        self.bump_geom_order = sorted(
            self._highlight_geom_ids,
            key=lambda geom_id: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        )
        self._default_bump_geom_order = sorted(
            [
                geom_id
                for geom_id in self._highlight_geom_ids
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith("bump_")
            ],
            key=lambda geom_id: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        )
        self._precision_geom_order = [self._precision_platform_geom_id, *self._precision_stone_geom_ids]
        self._stair_geom_order = list(self._stair_geom_ids)
        self._highlight_original_rgba = {
            geom_id: self.model.geom_rgba[geom_id].copy() for geom_id in self._highlight_geom_ids
        }
        self._wall_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            in {"wall_left", "wall_right", "wall_left_outline", "wall_right_outline"}
        ]
        self._preview_highlight_enabled = False
        self._scandot_overlay_enabled = False
        self._stair_edge_overlay_enabled = False
        self.bump_geom_ids = set(self._highlight_geom_ids)
        self._default_geom_pos = self.model.geom_pos.copy()
        self._default_geom_size = self.model.geom_size.copy()
        self._default_geom_type = self.model.geom_type.copy()
        self._default_geom_rgba = self.model.geom_rgba.copy()
        self._precision_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "precision_platform_body"),
            *[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"precision_stone_{idx:02d}_body")
                for idx in range(1, 25)
            ],
        ]
        self._precision_mocap_ids = [int(self.model.body_mocapid[body_id]) for body_id in self._precision_body_ids]
        self._stair_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"stair_step_{idx:02d}_body")
            for idx in range(1, 13)
        ]
        self._stair_mocap_ids = [int(self.model.body_mocapid[body_id]) for body_id in self._stair_body_ids]
        self.foot_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            in {"fl_foot_geom", "fr_foot_geom", "rl_foot_geom", "rr_foot_geom"}
        }
        self.foot_site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in ("fl_foot_site", "fr_foot_site", "rl_foot_site", "rr_foot_site")
            ],
            dtype=np.int32,
        )
        self.foot_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("fl_foot", "fr_foot", "rl_foot", "rr_foot")
            ],
            dtype=np.int32,
        )
        self.foot_body_id_to_index = {
            int(body_id): index for index, body_id in enumerate(self.foot_body_ids.tolist())
        }
        self.front_foot_indices = np.array([0, 1], dtype=np.int32)
        self.rear_foot_indices = np.array([2, 3], dtype=np.int32)
        self.rear_foot_body_ids = self.foot_body_ids[self.rear_foot_indices]
        self.rear_foot_body_id_to_index = {
            int(body_id): index for index, body_id in enumerate(self.rear_foot_body_ids.tolist())
        }
        self.previous_foot_positions = np.zeros((len(self.foot_site_ids), 3), dtype=np.float64)
        self.torso_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) == "torso_geom"
        }
        self.leg_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
                and name not in {"torso_geom"}
                and not name.startswith("bump_")
                and not name.startswith("wall_")
                and geom_id not in self.foot_geom_ids
            )
        }
        self.collision_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (
                    "fl_thigh",
                    "fl_shank",
                    "fr_thigh",
                    "fr_shank",
                    "rl_thigh",
                    "rl_shank",
                    "rr_thigh",
                    "rr_shank",
                )
            ],
            dtype=np.int32,
        )
        self.collision_body_id_to_index = {
            int(body_id): index for index, body_id in enumerate(self.collision_body_ids.tolist())
        }
        self.rear_leg_body_ids = {
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in ("rl_thigh", "rl_shank", "rl_foot", "rr_thigh", "rr_shank", "rr_foot")
        }
        self.previous_foot_contact_forces = np.zeros((len(self.foot_body_ids), 3), dtype=np.float64)
        self.bump_regions = []
        self._active_bump_geom_order = self._default_bump_geom_order
        self._active_stair_layout: list[tuple[float, float, float, float, float]] = []
        for geom_id in sorted(self.bump_geom_ids):
            self.bump_regions.append(
                (
                    self.model.geom_pos[geom_id][0] - self.model.geom_size[geom_id][0],
                    self.model.geom_pos[geom_id][0] + self.model.geom_size[geom_id][0],
                    self.model.geom_pos[geom_id][2] + self.model.geom_size[geom_id][2],
                )
            )
        self.use_privileged_terrain = bool(config.use_privileged_terrain)
        self._privileged_scan_highlight_enabled = self.use_privileged_terrain
        self.terrain_obs_mode = str(config.terrain_obs_mode).lower()
        self.terrain_obs_height_reference = str(config.terrain_obs_height_reference).lower()
        self.terrain_patch_center_offset = np.array(
            [float(config.terrain_patch_center_offset_x), float(config.terrain_patch_center_offset_y)],
            dtype=np.float64,
        )
        if self.terrain_obs_mode == "stable_world_patch":
            self.scandot_points = self._stable_world_patch_points(config)
        else:
            scandot_layout = str(config.scandot_layout).lower()
            if scandot_layout == "footstep_oriented":
                raw_scandot_points = self._footstep_oriented_scandot_points(config)
                enforce_forward_clearance = False
                enforce_body_clearance = False
            elif config.scandot_points:
                raw_scandot_points = np.asarray(config.scandot_points, dtype=np.float64)
                enforce_forward_clearance = True
                enforce_body_clearance = True
            else:
                raw_scandot_points = np.asarray(
                    [(x, y) for x in config.terrain_scan_x for y in config.terrain_scan_y],
                    dtype=np.float64,
                )
                enforce_forward_clearance = True
                enforce_body_clearance = True
            self.scandot_points = self._prepare_scandot_points(
                raw_scandot_points,
                min_forward_offset=float(config.scandot_min_forward_offset),
                enforce_forward_clearance=enforce_forward_clearance,
                enforce_body_clearance=enforce_body_clearance,
            )
        self.terrain_scan_x = np.unique(self.scandot_points[:, 0]) if self.scandot_points.size else np.array([], dtype=np.float64)
        self.terrain_scan_y = np.unique(self.scandot_points[:, 1]) if self.scandot_points.size else np.array([], dtype=np.float64)
        self.terrain_scan_origin_z = float(config.terrain_scan_origin_z)
        self.terrain_scan_max_dist = float(config.terrain_scan_max_dist)
        self.corridor_half_width = config.corridor_half_width
        self.observation_mode = config.observation_mode
        self.include_privileged_obs = bool(config.include_privileged_obs)
        self.observation_noise_std = float(config.observation_noise_std)
        self.privileged_randomization = config.privileged_randomization or {}
        self.curriculum_cfg = config.curriculum or {}
        self.terrain_generation_cfg = config.terrain_generation or {}
        self._default_body_mass = self.model.body_mass.copy()
        self._default_geom_friction = self.model.geom_friction.copy()
        self._default_dof_damping = self.model.dof_damping.copy()
        self._current_randomization = {
            "body_mass_scale": 1.0,
            "friction_scale": 1.0,
            "joint_damping_scale": 1.0,
        }
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.curriculum_enabled = bool(self.curriculum_cfg.get("enabled", False))
        self.curriculum_level = int(self.curriculum_cfg.get("initial_level", 0))
        self.curriculum_min_level = int(self.curriculum_cfg.get("min_level", 0))
        self.curriculum_max_level = int(self.curriculum_cfg.get("max_level", 5))
        self.curriculum_window = int(self.curriculum_cfg.get("window", 20))
        self.curriculum_promotion_threshold = float(self.curriculum_cfg.get("promotion_threshold", 0.8))
        self.curriculum_demotion_threshold = float(self.curriculum_cfg.get("demotion_threshold", 0.25))
        self.curriculum_success_distance = float(self.curriculum_cfg.get("success_distance", 2.2))
        self.curriculum_failure_distance = float(self.curriculum_cfg.get("failure_distance", 0.5))
        self.curriculum_success_x_fraction = float(self.curriculum_cfg.get("success_x_fraction", 2.0 / 3.0))
        self.curriculum_terrain_types = [self._normalize_terrain_type(name) for name in self.curriculum_cfg.get(
            "terrain_types",
            ["rough_flat", "slope_up_down", "stairs", "stepping_stones", "discrete_obstacles"],
        )]
        self.curriculum_history: deque[float] = deque(maxlen=self.curriculum_window)
        self.current_terrain_type = "flat"
        self.last_episode_distance = 0.0
        self.last_episode_success_x = 0.0
        self.last_episode_success_progress = 0.0
        self.last_episode_success = False
        self.last_episode_terminated = False
        self.last_episode_truncated = False
        self.last_episode_recorded = False
        self.fixed_terrain_type: str | None = None
        self.episode_max_x = 0.0
        self._progress_reference_x = 0.0
        self._stuck_steps = 0
        self._passed_stair_edge_count = 0
        self._rear_foot_highest_step = np.full(len(self.rear_foot_indices), -1, dtype=np.int32)
        self._rear_lag_steps = 0
        self._previous_rear_mean_x = 0.0
        self._previous_rear_mean_lift = 0.0
        self._episode_start_x = 0.0

        scandot_obs_dim = len(self.scandot_points) if self.use_privileged_terrain else 0
        obs_dim = 12 + 12 + 3 + 3 + 3 + 2 + scandot_obs_dim
        if self.observation_mode == "teacher_dict":
            scandot_shape = (scandot_obs_dim,) if self.use_privileged_terrain else (0,)
            obs_spaces = {
                "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32),
                "command": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            }
            if self.use_privileged_terrain:
                obs_spaces["scandots"] = spaces.Box(low=-np.inf, high=np.inf, shape=scandot_shape, dtype=np.float32)
            if self.include_privileged_obs:
                obs_spaces["privileged"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            self.observation_space = spaces.Dict(obs_spaces)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(JOINT_NAMES),), dtype=np.float32)

    def _footstep_oriented_scandot_points(self, config: EnvConfig) -> np.ndarray:
        if config.footstep_scandot_points:
            return np.asarray(config.footstep_scandot_points, dtype=np.float64)
        return np.asarray(
            [
                (0.08, -0.07),
                (0.08, 0.07),
                (0.16, -0.07),
                (0.16, 0.07),
                (0.24, -0.10),
                (0.24, 0.00),
                (0.24, 0.10),
                (0.36, -0.10),
                (0.36, 0.00),
                (0.36, 0.10),
                (0.50, -0.08),
                (0.50, 0.08),
            ],
            dtype=np.float64,
        )

    def _stable_world_patch_points(self, config: EnvConfig) -> np.ndarray:
        rows = int(config.terrain_patch_rows or 0)
        cols = int(config.terrain_patch_cols or 0)
        if rows <= 0 or cols <= 0:
            rows, cols = 4, 5
        dx = float(config.terrain_patch_dx)
        dy = float(config.terrain_patch_dy)
        x_coords = (np.arange(cols, dtype=np.float64) - 0.5 * (cols - 1)) * dx
        y_coords = (np.arange(rows, dtype=np.float64) - 0.5 * (rows - 1)) * dy
        points = [(float(x_val), float(y_val)) for x_val in x_coords for y_val in y_coords]
        return np.asarray(points, dtype=np.float64)

    def _prepare_scandot_points(
        self,
        points: np.ndarray,
        min_forward_offset: float,
        enforce_forward_clearance: bool = True,
        enforce_body_clearance: bool = True,
    ) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"scandot_points must have shape (N, 2), got {points.shape}")

        torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
        torso_half_length = float(self.model.geom_size[torso_geom_id][0])
        torso_half_width = float(self.model.geom_size[torso_geom_id][1])
        front_hip_x = float(np.max(self.front_hip_positions[:, 0]))
        front_hip_y = float(np.max(np.abs(self.front_hip_positions[:, 1])))
        forward_clearance = max(min_forward_offset, torso_half_length + 0.05, front_hip_x + 0.06)
        body_lateral_clearance = max(torso_half_width + 0.03, front_hip_y - 0.005)

        filtered = []
        for x_val, y_val in points:
            x_val = float(x_val)
            y_val = float(y_val)
            if enforce_forward_clearance and x_val < forward_clearance:
                continue
            if enforce_body_clearance and x_val < torso_half_length + 0.09 and abs(y_val) < body_lateral_clearance:
                continue
            filtered.append((round(x_val, 4), round(y_val, 4)))

        unique_points = sorted(set(filtered), key=lambda item: (item[0], item[1]))
        if not unique_points:
            raise ValueError("All scandot points were filtered out; widen the template or increase forward reach.")
        return np.asarray(unique_points, dtype=np.float64)

    def _base_rotation(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        return rot.reshape(3, 3)

    def _heading_rotation(self) -> np.ndarray:
        rot = self._base_rotation()
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        return np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _projected_gravity(self) -> np.ndarray:
        rot = self._base_rotation()
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return rot.T @ gravity_world

    def _get_proprio_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr] - self.default_pose
        joint_vel = self.data.qvel[self.joint_qvel_adr]
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        projected_gravity = self._projected_gravity()
        proprio = np.concatenate(
            [
                joint_pos,
                joint_vel,
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                self.previous_action,
            ]
        )
        if self.observation_noise_std > 0.0:
            proprio = proprio + self.np_random.normal(0.0, self.observation_noise_std, size=proprio.shape)
        return proprio.astype(np.float32)

    def _get_privileged_obs(self) -> np.ndarray:
        privileged = np.array(
            [
                self._current_randomization["body_mass_scale"],
                self._current_randomization["friction_scale"],
                self._current_randomization["joint_damping_scale"],
            ],
            dtype=np.float32,
        )
        return privileged

    def _get_obs(self):
        proprio = self._get_proprio_obs()
        command = np.array([self.command_velocity, self.command_yaw_rate], dtype=np.float32)
        scandots = None
        if self.use_privileged_terrain:
            scandots = self._scandot_height_samples().astype(np.float32)
            if self.observation_noise_std > 0.0:
                scandots = scandots + self.np_random.normal(0.0, self.observation_noise_std, size=scandots.shape).astype(np.float32)
        if self.observation_mode == "teacher_dict":
            obs = {
                "proprio": proprio,
                "command": command,
            }
            if scandots is not None:
                obs["scandots"] = scandots
            if self.include_privileged_obs:
                obs["privileged"] = self._get_privileged_obs()
            return obs
        obs_parts = [
            proprio.astype(np.float64),
            command.astype(np.float64),
        ]
        if scandots is not None:
            obs_parts.append(scandots.astype(np.float64))
        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)

    def _sample_command(self) -> tuple[float, float]:
        linear_velocity = float(self.np_random.uniform(self.config.command_min, self.config.command_max))
        yaw_rate = float(self.np_random.uniform(self.config.command_yaw_min, self.config.command_yaw_max))
        return linear_velocity, yaw_rate

    def _initial_base_position(self) -> np.ndarray:
        base_pos = np.array([0.0, 0.0, 0.22], dtype=np.float64)
        if self.current_terrain_type == "precision_stepping_stones":
            platform_center_x = float(self._terrain_param("precision_stone_platform_center_x", 0.38))
            platform_top_z = 2.0 * float(self._terrain_param("precision_stone_platform_height", 0.065))
            spawn_x = float(self._terrain_param("precision_stone_spawn_x", platform_center_x - 0.08))
            spawn_y = float(self._terrain_param("precision_stone_spawn_y", 0.0))
            spawn_clearance_z = float(self._terrain_param("precision_stone_spawn_clearance_z", 0.22))
            base_pos[:] = np.array([spawn_x, spawn_y, platform_top_z + spawn_clearance_z], dtype=np.float64)
        return base_pos

    def _reset_state(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = self._initial_base_position()
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        noise = self.np_random.uniform(-self.config.reset_noise_scale, self.config.reset_noise_scale, size=self.default_pose.shape)
        self.data.qpos[self.joint_qpos_adr] = self.default_pose + noise
        mujoco.mj_forward(self.model, self.data)
        self.previous_foot_positions = self.data.site_xpos[self.foot_site_ids].copy()
        self.previous_foot_contact_forces[:] = 0.0
        self.episode_max_x = float(self.data.qpos[0])
        self._episode_start_x = self.episode_max_x
        self._progress_reference_x = self.episode_max_x
        self._stuck_steps = 0
        self._passed_stair_edge_count = 0
        self._rear_foot_highest_step[:] = -1
        self._rear_lag_steps = 0
        self._previous_rear_mean_x = float(np.mean(self.data.site_xpos[self.foot_site_ids[self.rear_foot_indices], 0]))
        self._previous_rear_mean_lift = 0.0

    def _sample_scale(self, key: str) -> float:
        spec = self.privileged_randomization.get(key)
        if spec is None:
            return 1.0
        if isinstance(spec, (int, float)):
            return float(spec)
        low, high = spec
        return float(self.np_random.uniform(low, high))

    def _apply_domain_randomization(self) -> None:
        self.model.body_mass[:] = self._default_body_mass
        self.model.geom_friction[:] = self._default_geom_friction
        self.model.dof_damping[:] = self._default_dof_damping

        body_mass_scale = self._sample_scale("body_mass_range")
        friction_scale = self._sample_scale("friction_range")
        joint_damping_scale = self._sample_scale("joint_damping_range")

        self.model.body_mass[1:] = self._default_body_mass[1:] * body_mass_scale
        friction_geom_ids = [self.ground_geom_id, *sorted(self.bump_geom_ids)]
        for geom_id in friction_geom_ids:
            if geom_id >= 0:
                self.model.geom_friction[geom_id] = self._default_geom_friction[geom_id] * friction_scale
        self.model.dof_damping[self.joint_qvel_adr] = self._default_dof_damping[self.joint_qvel_adr] * joint_damping_scale

        self._current_randomization = {
            "body_mass_scale": body_mass_scale,
            "friction_scale": friction_scale,
            "joint_damping_scale": joint_damping_scale,
        }

    def _terrain_level_scale(self) -> float:
        level_scale_overrides = self.terrain_generation_cfg.get("level_scale_overrides", {})
        if isinstance(level_scale_overrides, dict):
            override = level_scale_overrides.get(str(self.curriculum_level), level_scale_overrides.get(self.curriculum_level))
            if override is not None:
                override_level = int(override)
                if self.curriculum_max_level <= self.curriculum_min_level:
                    return 0.0
                return float(np.clip(override_level, self.curriculum_min_level, self.curriculum_max_level) - self.curriculum_min_level) / float(
                    self.curriculum_max_level - self.curriculum_min_level
                )
        if self.curriculum_max_level <= self.curriculum_min_level:
            return 0.0
        return float(self.curriculum_level - self.curriculum_min_level) / float(
            self.curriculum_max_level - self.curriculum_min_level
        )

    def _normalize_terrain_type(self, terrain_type: str | None) -> str:
        terrain_type = "rough_flat" if terrain_type is None else str(terrain_type)
        aliases = {
            "bumps": "rough_flat",
            "slopes": "slope_up_down",
            "random_slopes": "random_slope_up_down",
            "steps": "stairs",
            "stairs_curbs": "stairs",
            "blocks": "discrete_obstacles",
            "stones": "stepping_stones",
            "precision_stones": "precision_stepping_stones",
            "mixed": "mixed_course",
        }
        return aliases.get(terrain_type, terrain_type)

    def _terrain_param(self, key: str, default):
        return self.terrain_generation_cfg.get(key, default)

    def _stairs_height_scale_factor(self) -> float:
        difficulty_mode = str(self._terrain_param("stairs_difficulty_mode", "default")).lower()
        mode_scale = {
            "default": 1.0,
            "higher_steps": 1.30,
        }.get(difficulty_mode, 1.0)
        std_scale = float(self._terrain_param("stairs_std_scale", 1.0))
        return mode_scale * std_scale

    def _segment_tile_counts(self) -> dict[str, int]:
        defaults = {
            "rough_flat": 10,
            "slope_up": 2,
            "slope_down": 2,
            "slope_up_down": 3,
            "stepping_stones": 3,
            "stairs": 2,
            "discrete_obstacles": 2,
        }
        counts = dict(defaults)
        counts.update(self._terrain_param("segment_tile_counts", {}))
        return {str(name): max(1, int(value)) for name, value in counts.items()}

    def _set_bump_geom(self, geom_id: int, x: float, y: float, half_length: float, half_width: float, height: float) -> None:
        self.model.geom_type[geom_id] = self._default_geom_type[geom_id]
        self.model.geom_pos[geom_id] = np.array([x, y, height], dtype=np.float64)
        self.model.geom_size[geom_id] = np.array([half_length, half_width, height], dtype=np.float64)
        self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_cylinder_geom(self, geom_id: int, x: float, y: float, radius: float, height: float) -> None:
        self.model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
        self.model.geom_pos[geom_id] = np.array([x, y, height], dtype=np.float64)
        self.model.geom_size[geom_id] = np.array([radius, height, 0.0], dtype=np.float64)
        self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_ramp_geom(
        self,
        geom_id: int,
        x: float,
        y: float,
        half_length: float,
        half_width: float,
        half_height: float,
        pitch_rad: float,
        center_z: float | None = None,
    ) -> None:
        self.model.geom_type[geom_id] = self._default_geom_type[geom_id]
        z = half_height if center_z is None else float(center_z)
        self.model.geom_pos[geom_id] = np.array([x, y, z], dtype=np.float64)
        self.model.geom_size[geom_id] = np.array([half_length, half_width, half_height], dtype=np.float64)
        half_angle = 0.5 * float(pitch_rad)
        self.model.geom_quat[geom_id] = np.array(
            [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
            dtype=np.float64,
        )

    def _hide_bump_geom(self, geom_id: int) -> None:
        self.model.geom_type[geom_id] = self._default_geom_type[geom_id]
        self.model.geom_pos[geom_id] = np.array([8.0, 0.0, -0.25], dtype=np.float64)
        self.model.geom_size[geom_id] = np.array([0.02, 0.02, 0.01], dtype=np.float64)
        self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_precision_mocap_pose(self, index: int, x: float, y: float, z_center: float) -> None:
        mocap_id = self._precision_mocap_ids[index]
        self.data.mocap_pos[mocap_id] = np.array([x, y, z_center], dtype=np.float64)
        self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _hide_precision_geom(self, index: int) -> None:
        self._set_precision_mocap_pose(index, 8.0, 0.0, -1.0)

    def _hide_all_precision_geoms(self) -> None:
        for index in range(len(self._precision_mocap_ids)):
            self._hide_precision_geom(index)

    def _sync_precision_geom_sizes_from_config(self) -> None:
        platform_half_length = float(self._terrain_param("precision_stone_platform_half_length", 0.26))
        platform_half_width = float(self._terrain_param("precision_stone_platform_half_width", 0.24))
        platform_height = float(self._terrain_param("precision_stone_platform_height", 0.10))
        self.model.geom_size[self._precision_platform_geom_id] = np.array(
            [platform_half_length, platform_half_width, platform_height],
            dtype=np.float64,
        )

        radius = float(
            self._terrain_param(
                "precision_stone_radius_easy",
                self._terrain_param("precision_stone_radius", 0.09),
            )
        )
        stone_height = float(
            self._terrain_param(
                "precision_stone_height_easy",
                self._terrain_param("precision_stone_height", platform_height),
            )
        )
        for geom_id in self._precision_stone_geom_ids:
            self.model.geom_size[geom_id] = np.array([radius, stone_height, 0.0], dtype=np.float64)

    def _set_stair_mocap_pose(self, index: int, x: float, y: float, z_center: float) -> None:
        mocap_id = self._stair_mocap_ids[index]
        self.data.mocap_pos[mocap_id] = np.array([x, y, z_center], dtype=np.float64)
        self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_stair_mocap_ramp_pose(self, index: int, x: float, y: float, z_center: float, pitch_rad: float) -> None:
        mocap_id = self._stair_mocap_ids[index]
        half_angle = 0.5 * float(pitch_rad)
        self.data.mocap_pos[mocap_id] = np.array([x, y, z_center], dtype=np.float64)
        self.data.mocap_quat[mocap_id] = np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0], dtype=np.float64)

    def _hide_stair_geom(self, index: int) -> None:
        self._set_stair_mocap_pose(index, 8.0, 0.0, -1.0)

    def _hide_all_stair_geoms(self) -> None:
        for index in range(len(self._stair_mocap_ids)):
            self._hide_stair_geom(index)

    def _update_corridor_walls(self) -> None:
        if not self._wall_geom_ids:
            return
        if self.corridor_half_width is None:
            for geom_id in self._wall_geom_ids:
                self.model.geom_pos[geom_id] = np.array([8.0, 0.0, -0.50], dtype=np.float64)
                self.model.geom_size[geom_id] = np.array([0.02, 0.02, 0.01], dtype=np.float64)
                self.model.geom_rgba[geom_id] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return
        for geom_id in self._wall_geom_ids:
            self.model.geom_pos[geom_id] = self._default_geom_pos[geom_id]
            self.model.geom_size[geom_id] = self._default_geom_size[geom_id]
            self.model.geom_rgba[geom_id] = self._default_geom_rgba[geom_id]

    def _generate_rough_flat_layout(self, level_scale: float) -> list[tuple[float, float, float, float, float]]:
        """Continuous full-width undulations that approximate rough flat terrain."""
        layout = []
        x = 0.42
        half_width = 0.255
        half_length = 0.15
        base_amp = 0.008 + 0.020 * level_scale
        for idx in range(len(self.bump_geom_order)):
            phase = idx * 0.85
            height = 0.004 + base_amp * (0.45 + 0.55 * np.sin(phase))
            height += 0.004 * level_scale * ((idx % 2) - 0.5)
            layout.append((x, 0.0, half_width, max(height, 0.004), half_length))
            x += 2.0 * half_length - 0.01
        return layout

    def _generate_slope_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
        direction: str,
    ) -> list[tuple[float, float, float, float, float, float]]:
        layout = []
        half_width = float(self._terrain_param("slope_half_width", 0.255))
        half_length = float(self._terrain_param("slope_half_length", 0.20))
        stride = float(self._terrain_param("slope_stride", 0.22))
        rise = float(self._terrain_param("slope_rise_base", 0.030)) + float(
            self._terrain_param("slope_rise_scale", 0.070)
        ) * level_scale
        thickness = float(self._terrain_param("slope_thickness", 0.012))
        step_count = max(1, tile_count)
        x = x_start
        max_rise = min(rise, 2.0 * half_length * 0.95)
        pitch_mag = float(np.arcsin(max(max_rise / max(2.0 * half_length, 1e-6), 0.0)))
        if direction == "up":
            pitch = -pitch_mag
        elif direction == "down":
            pitch = pitch_mag
        else:
            raise ValueError(f"Unsupported slope direction: {direction}")
        for _ in range(step_count):
            layout.append((x, 0.0, half_width, thickness, half_length, pitch))
            x += stride
        return layout

    def _generate_slope_up_down_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
    ) -> list[tuple[float, float, float, float, float, float]]:
        ascent = max(1, tile_count // 2)
        descent = max(1, tile_count - ascent)
        layout = self._generate_slope_segment(level_scale, x_start, ascent, direction="up")
        if layout:
            last_x, _, _, _, last_half_length = layout[-1]
            next_x = float(last_x + last_half_length + 0.15)
        else:
            next_x = x_start
        layout.extend(self._generate_slope_segment(level_scale, next_x, descent, direction="down"))
        return layout

    def _generate_random_slope_up_down_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
    ) -> list[tuple[float, float, float, float, float, float, float]]:
        """Piecewise planar terrain: initial uphill, then frequent random slope changes."""
        layout = []
        if tile_count <= 0:
            return layout
        half_width = float(self._terrain_param("random_slope_half_width", 0.255))
        thickness = float(self._terrain_param("random_slope_thickness", 0.035))
        overlap = float(self._terrain_param("random_slope_overlap", 0.015))
        lateral_shift_fraction = float(self._terrain_param("random_slope_lateral_shift_max_fraction", 0.0))
        lateral_shift_max = max(0.0, min(1.0, lateral_shift_fraction)) * half_width
        lateral_center_limit_fraction = float(self._terrain_param("random_slope_lateral_center_limit_fraction", 0.0))
        lateral_center_limit = max(0.0, lateral_center_limit_fraction) * half_width
        min_angle_deg = float(self._terrain_param("random_slope_min_angle_deg", 5.0))
        max_angle_deg = float(self._terrain_param("random_slope_max_angle_deg", 14.0))
        level_angle_scale = float(self._terrain_param("random_slope_level_angle_scale", 1.0))
        angle_high = min_angle_deg + (max_angle_deg - min_angle_deg) * float(np.clip(level_scale * level_angle_scale, 0.0, 1.0))
        angle_low = min(min_angle_deg, angle_high)
        initial_uphill_run = float(self.np_random.uniform(
            float(self._terrain_param("random_slope_initial_uphill_run_min", 1.35)),
            float(self._terrain_param("random_slope_initial_uphill_run_max", 1.65)),
        ))
        segment_run_min = float(self._terrain_param("random_slope_segment_run_min", 0.35))
        segment_run_max = float(self._terrain_param("random_slope_segment_run_max", 0.55))
        first_segment_count = int(self._terrain_param("random_slope_initial_segments", 3))
        first_segment_count = max(1, min(first_segment_count, tile_count))
        first_angle_factors = [
            float(value)
            for value in self._terrain_param("random_slope_initial_angle_factors", [0.55, 0.80, 1.0])
        ]
        if not first_angle_factors:
            first_angle_factors = [1.0]
        base_initial_angle = float(self.np_random.uniform(angle_low, angle_high))
        first_run = initial_uphill_run / float(first_segment_count)
        segments: list[tuple[float, float]] = []
        current_top_z = 0.0
        for idx in range(first_segment_count):
            factor = first_angle_factors[min(idx, len(first_angle_factors) - 1)]
            angle = float(np.deg2rad(base_initial_angle * factor))
            pitch = -angle
            segments.append((first_run, pitch))
            current_top_z += first_run * float(np.tan(-pitch))

        max_top_z = float(self._terrain_param("random_slope_max_top_height", 0.42))
        min_top_z = float(self._terrain_param("random_slope_min_top_height", 0.0))
        turn_probability = float(self._terrain_param("random_slope_turn_probability", 0.55))
        same_direction_probability = float(self._terrain_param("random_slope_same_direction_probability", 0.35))
        max_delta_deg = float(self._terrain_param("random_slope_max_angle_delta_deg", 3.5))
        direction = int(self.np_random.choice([-1, 1]))
        current_angle_deg = float(self.np_random.uniform(angle_low, angle_high))

        for _ in range(first_segment_count, tile_count):
            run = float(self.np_random.uniform(segment_run_min, segment_run_max))
            if current_top_z <= min_top_z + 0.025:
                direction = 1
            elif current_top_z >= max_top_z - 0.025:
                direction = -1
            elif float(self.np_random.uniform()) < turn_probability:
                if float(self.np_random.uniform()) > same_direction_probability:
                    direction *= -1

            current_angle_deg += float(self.np_random.uniform(-max_delta_deg, max_delta_deg))
            current_angle_deg = float(np.clip(current_angle_deg, angle_low, angle_high))
            pitch = -float(np.deg2rad(current_angle_deg)) if direction > 0 else float(np.deg2rad(current_angle_deg))
            height_delta = run * float(np.tan(-pitch))
            if current_top_z + height_delta < min_top_z:
                direction = 1
                pitch = -abs(pitch)
                height_delta = run * float(np.tan(-pitch))
            elif current_top_z + height_delta > max_top_z:
                direction = -1
                pitch = abs(pitch)
                height_delta = run * float(np.tan(-pitch))
            segments.append((run, pitch))
            current_top_z += height_delta

        start_edge_x = float(x_start)
        start_top_z = 0.0
        center_y = 0.0
        for index, (run, pitch) in enumerate(segments[:tile_count]):
            if index > 0 and lateral_shift_max > 0.0:
                center_y += float(self.np_random.uniform(-lateral_shift_max, lateral_shift_max))
                if lateral_center_limit > 0.0:
                    center_y = float(np.clip(center_y, -lateral_center_limit, lateral_center_limit))
            run = float(run) + overlap
            half_length = max(0.02, 0.5 * run / max(float(np.cos(pitch)), 1e-6))
            cos_pitch = float(np.cos(pitch))
            sin_pitch = float(np.sin(pitch))
            x = start_edge_x + cos_pitch * half_length - sin_pitch * thickness
            center_z = start_top_z - sin_pitch * half_length - cos_pitch * thickness
            layout.append((x, center_y, half_width, thickness, half_length, pitch, center_z))
            end_edge_x = x + cos_pitch * half_length + sin_pitch * thickness
            end_top_z = start_top_z + (run - overlap) * float(np.tan(-pitch))
            start_top_z = 0.0 if abs(end_top_z) < 1e-9 else float(end_top_z)
            start_edge_x = float(end_edge_x - overlap)
        return layout

    def _generate_stepping_stones_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
    ) -> list[tuple[float, float, float, float, float]]:
        layout = []
        x = x_start
        lateral_choices = list(self._terrain_param("stepping_stone_lateral_choices", [-0.12, 0.0, 0.12]))
        base_height = float(self._terrain_param("stepping_stone_height_base", 0.018))
        height_scale = float(self._terrain_param("stepping_stone_height_scale", 0.030))
        gap_min = float(self._terrain_param("stepping_stone_gap_min", 0.11))
        gap_max = float(self._terrain_param("stepping_stone_gap_max", 0.18))
        for idx in range(tile_count):
            y = 0.0 if idx == 0 else float(self.np_random.choice(lateral_choices))
            half_width = float(self.np_random.uniform(
                float(self._terrain_param("stepping_stone_half_width_min", 0.05)),
                float(self._terrain_param("stepping_stone_half_width_max", 0.08)),
            ))
            half_length = float(self.np_random.uniform(
                float(self._terrain_param("stepping_stone_half_length_min", 0.07)),
                float(self._terrain_param("stepping_stone_half_length_max", 0.11)),
            ))
            height = base_height + height_scale * level_scale + 0.008 * float(self.np_random.uniform())
            layout.append((x, y, half_width, height, half_length))
            x += 2.0 * half_length + float(self.np_random.uniform(gap_min, gap_max))
        return layout

    def _generate_precision_stepping_stones_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
    ) -> list[dict[str, float | str]]:
        layout: list[dict[str, float | str]] = []
        platform_height = float(self._terrain_param("precision_stone_platform_height", 0.065))
        platform_half_length = float(self._terrain_param("precision_stone_platform_half_length", 0.24))
        platform_half_width = float(self._terrain_param("precision_stone_platform_half_width", 0.24))
        platform_center_x = float(self._terrain_param("precision_stone_platform_center_x", x_start))
        layout.append(
            {
                "shape": "box",
                "x": platform_center_x,
                "y": 0.0,
                "half_width": platform_half_width,
                "height": platform_height,
                "half_length": platform_half_length,
            }
        )
        radius = float(self._terrain_param("precision_stone_radius_easy", self._terrain_param("precision_stone_radius", 0.085))) + level_scale * float(
            self._terrain_param("precision_stone_radius_scale", -0.010)
        )
        stone_height = float(self._terrain_param("precision_stone_height_easy", platform_height)) + level_scale * float(
            self._terrain_param("precision_stone_height_scale", 0.010)
        )
        stride = float(self._terrain_param("precision_stone_stride_easy", self._terrain_param("precision_stone_stride", 0.22))) + level_scale * float(
            self._terrain_param("precision_stone_stride_scale", 0.05)
        )
        first_center_x = float(self._terrain_param("precision_stone_first_center_x", platform_center_x + platform_half_length + 0.17))
        pair_gap = float(self._terrain_param("precision_stone_pair_gap_easy", self._terrain_param("precision_stone_pair_gap", 0.11))) + level_scale * float(
            self._terrain_param("precision_stone_pair_gap_scale", 0.03)
        )
        pair_rows = max(1, int(self._terrain_param("precision_stone_pair_rows", max(1, (tile_count - 1) // 2))))
        row_lateral_bias = [float(v) for v in self._terrain_param("precision_stone_row_lateral_bias", [0.0, 0.025, -0.025, 0.015, -0.015])]
        row_x_jitter = float(self._terrain_param("precision_stone_row_x_jitter_easy", self._terrain_param("precision_stone_row_x_jitter", 0.0))) + level_scale * float(
            self._terrain_param("precision_stone_row_x_jitter_scale", 0.015)
        )
        stone_y_jitter = float(self._terrain_param("precision_stone_y_jitter_easy", self._terrain_param("precision_stone_y_jitter", 0.0))) + level_scale * float(
            self._terrain_param("precision_stone_y_jitter_scale", 0.010)
        )
        for row_idx in range(pair_rows):
            x = first_center_x + stride * row_idx + float(self.np_random.uniform(-row_x_jitter, row_x_jitter))
            row_bias = row_lateral_bias[row_idx % len(row_lateral_bias)]
            for y in (-0.5 * pair_gap + row_bias, 0.5 * pair_gap + row_bias):
                y += float(self.np_random.uniform(-stone_y_jitter, stone_y_jitter))
                layout.append(
                    {
                        "shape": "cylinder",
                        "x": x,
                        "y": y,
                        "radius": radius,
                        "height": stone_height,
                    }
                )
        return layout[:tile_count]

    def _generate_stairs_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
    ) -> list[tuple[float, float, float, float, float]]:
        layout = []
        x = x_start
        half_width = float(self._terrain_param("stairs_half_width", 0.255))
        stair_height = float(self._terrain_param("stairs_height_base", 0.025)) + float(
            self._terrain_param("stairs_height_scale", 0.035)
        ) * level_scale
        stair_height *= self._stairs_height_scale_factor()
        half_length = float(self._terrain_param("stairs_half_length", 0.10))
        stride = float(self._terrain_param("stairs_stride", 0.20))
        min_half_height = float(self._terrain_param("stairs_min_half_height", 0.01))
        warmup_progression = list(self._terrain_param("stairs_warmup_progression", []))
        for idx in range(tile_count):
            warmup_scale = 1.0
            if idx < len(warmup_progression):
                warmup_scale = float(warmup_progression[idx])
            height = stair_height * float(idx + 1) * warmup_scale
            layout.append((x, 0.0, half_width, max(height, min_half_height), half_length))
            x += stride
        return layout

    def _generate_discrete_obstacles_segment(
        self,
        level_scale: float,
        x_start: float,
        tile_count: int,
    ) -> list[tuple[float, float, float, float, float]]:
        """Sparse lateralized blocks that force cleaner placement and obstacle avoidance."""
        layout = []
        x = x_start
        lateral_choices = list(self._terrain_param("discrete_lateral_choices", [-0.14, -0.08, 0.0, 0.08, 0.14]))
        for idx in range(tile_count):
            y = float(self.np_random.choice(lateral_choices))
            if idx == 0:
                y = 0.0
            half_width = 0.06 + 0.02 * self.np_random.uniform()
            half_length = 0.08 + 0.03 * self.np_random.uniform()
            height = 0.018 + 0.030 * level_scale + 0.008 * self.np_random.uniform()
            layout.append((x, y, half_width, height, half_length))
            x += 0.34 + 0.08 * self.np_random.uniform()
        return layout

    def _generate_mixed_course_layout(self, level_scale: float) -> list[tuple[float, float, float, float, float] | tuple[float, float, float, float, float, float]]:
        layout = []
        x_cursor = float(self._terrain_param("mixed_course_x_start", 0.52))
        gap = float(self._terrain_param("mixed_course_segment_gap", 0.18))
        first_segment_scale = float(self._terrain_param("mixed_first_segment_scale", 0.65))
        counts = self._segment_tile_counts()
        sequence = [self._normalize_terrain_type(name) for name in self._terrain_param(
            "mixed_course_segments",
            ["slope_up", "slope_down", "stepping_stones", "stairs", "discrete_obstacles"],
        )]
        generators = {
            "rough_flat": lambda x, n: self._generate_rough_flat_layout(level_scale)[:n],
            "slope_up": lambda x, n: self._generate_slope_segment(level_scale, x, n, direction="up"),
            "slope_down": lambda x, n: self._generate_slope_segment(level_scale, x, n, direction="down"),
            "slope_up_down": lambda x, n: self._generate_slope_up_down_segment(level_scale, x, n),
            "stepping_stones": lambda x, n: self._generate_stepping_stones_segment(level_scale, x, n),
            "stairs": lambda x, n: self._generate_stairs_segment(level_scale, x, n),
            "discrete_obstacles": lambda x, n: self._generate_discrete_obstacles_segment(level_scale, x, n),
        }
        remaining = len(self.bump_geom_order)
        for segment_idx, segment_name in enumerate(sequence):
            if remaining <= 0:
                break
            tile_count = min(remaining, counts.get(segment_name, 2))
            local_scale = level_scale * first_segment_scale if segment_idx == 0 else level_scale
            if segment_name == "slope_up":
                segment_layout = self._generate_slope_segment(local_scale, x_cursor, tile_count, direction="up")
            elif segment_name == "slope_down":
                segment_layout = self._generate_slope_segment(local_scale, x_cursor, tile_count, direction="down")
            elif segment_name == "slope_up_down":
                segment_layout = self._generate_slope_up_down_segment(local_scale, x_cursor, tile_count)
            elif segment_name == "stepping_stones":
                segment_layout = self._generate_stepping_stones_segment(local_scale, x_cursor, tile_count)
            elif segment_name == "stairs":
                segment_layout = self._generate_stairs_segment(local_scale, x_cursor, tile_count)
            elif segment_name == "discrete_obstacles":
                segment_layout = self._generate_discrete_obstacles_segment(local_scale, x_cursor, tile_count)
            else:
                segment_layout = generators[segment_name](x_cursor, tile_count)
            if not segment_layout:
                continue
            layout.extend(segment_layout[:tile_count])
            remaining -= len(segment_layout[:tile_count])
            last_half_length = float(layout[-1][4])
            last_x = float(layout[-1][0])
            x_cursor = float(last_x + last_half_length + gap)
        return layout[: len(self.bump_geom_order)]

    def _apply_terrain_layout(self, terrain_type: str | None = None) -> None:
        level_scale = self._terrain_level_scale() if self.curriculum_enabled else 0.0
        if terrain_type is not None:
            self.current_terrain_type = self._normalize_terrain_type(terrain_type)
        elif self.fixed_terrain_type is not None:
            self.current_terrain_type = self._normalize_terrain_type(self.fixed_terrain_type)
        elif self.curriculum_enabled:
            available_terrain_types = self._available_terrain_types()
            self.current_terrain_type = str(self.np_random.choice(available_terrain_types))
        else:
            self.current_terrain_type = "rough_flat"

        terrain_type = self._normalize_terrain_type(self.current_terrain_type)
        self.current_terrain_type = terrain_type

        if terrain_type == "rough_flat":
            layout = self._generate_rough_flat_layout(level_scale)
        elif terrain_type == "slope_up":
            layout = self._generate_slope_segment(level_scale, x_start=0.48, tile_count=len(self.bump_geom_order), direction="up")
        elif terrain_type == "slope_down":
            layout = self._generate_slope_segment(level_scale, x_start=0.48, tile_count=len(self.bump_geom_order), direction="down")
        elif terrain_type == "slope_up_down":
            layout = self._generate_slope_up_down_segment(level_scale, x_start=0.48, tile_count=len(self.bump_geom_order))
        elif terrain_type == "random_slope_up_down":
            layout = self._generate_random_slope_up_down_segment(
                level_scale,
                x_start=float(self._terrain_param("random_slope_x_start", 0.52)),
                tile_count=len(self._stair_geom_order),
            )
        elif terrain_type == "stepping_stones":
            layout = self._generate_stepping_stones_segment(level_scale, x_start=0.58, tile_count=len(self.bump_geom_order))
        elif terrain_type == "precision_stepping_stones":
            layout = self._generate_precision_stepping_stones_segment(
                level_scale,
                x_start=float(self._terrain_param("precision_stone_platform_center_x", 0.38)),
                tile_count=len(self.bump_geom_order),
            )
        elif terrain_type == "stairs":
            layout = self._generate_stairs_segment(
                level_scale,
                x_start=float(self._terrain_param("stairs_x_start", 0.56)),
                tile_count=len(self._stair_geom_order),
            )
        elif terrain_type == "discrete_obstacles":
            layout = self._generate_discrete_obstacles_segment(level_scale, x_start=0.62, tile_count=len(self.bump_geom_order))
        elif terrain_type == "mixed_course":
            layout = self._generate_mixed_course_layout(level_scale)
        else:
            layout = self._generate_rough_flat_layout(level_scale)

        self.bump_regions = []
        self._active_stair_layout = []
        if terrain_type == "precision_stepping_stones":
            self._active_bump_geom_order = self._precision_geom_order[: len(layout)]
            self._sync_precision_geom_sizes_from_config()
            for geom_id in self._default_bump_geom_order:
                self._hide_bump_geom(geom_id)
            self._hide_all_stair_geoms()
            for index, spec in enumerate(layout):
                shape = str(spec.get("shape", "box"))
                x = float(spec["x"])
                y = float(spec.get("y", 0.0))
                height = float(spec["height"])
                self._set_precision_mocap_pose(index, x, y, height)
                if shape == "cylinder":
                    radius = float(self.model.geom_size[self._precision_geom_order[index]][0])
                    self.bump_regions.append((x - radius, x + radius, y - radius, y + radius, 2.0 * height))
                else:
                    half_length = float(self.model.geom_size[self._precision_geom_order[index]][0])
                    half_width = float(self.model.geom_size[self._precision_geom_order[index]][1])
                    self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, 2.0 * height))
            for index in range(len(layout), len(self._precision_geom_order)):
                self._hide_precision_geom(index)
        elif terrain_type == "random_slope_up_down":
            self._active_bump_geom_order = self._stair_geom_order[: len(layout)]
            for geom_id in self._default_bump_geom_order:
                self._hide_bump_geom(geom_id)
            self._hide_all_precision_geoms()
            for index, spec in enumerate(layout):
                x, y, half_width, height, half_length, pitch, center_z = spec
                geom_id = self._stair_geom_order[index]
                self.model.geom_size[geom_id] = np.array([half_length, half_width, height], dtype=np.float64)
                self._set_stair_mocap_ramp_pose(index, float(x), float(y), float(center_z), float(pitch))
                top_z = float(center_z + abs(np.sin(float(pitch))) * half_length + np.cos(float(pitch)) * height)
                self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, top_z))
            for index in range(len(layout), len(self._stair_geom_order)):
                self._hide_stair_geom(index)
        elif terrain_type == "stairs":
            self._active_bump_geom_order = self._stair_geom_order[: len(layout)]
            self._active_stair_layout = list(layout)
            for geom_id in self._default_bump_geom_order:
                self._hide_bump_geom(geom_id)
            self._hide_all_precision_geoms()
            for index, spec in enumerate(layout):
                x, y, half_width, height, half_length = spec
                geom_id = self._stair_geom_order[index]
                self.model.geom_size[geom_id] = np.array([half_length, half_width, height], dtype=np.float64)
                self._set_stair_mocap_pose(index, float(x), float(y), float(height))
                self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, 2.0 * height))
            for index in range(len(layout), len(self._stair_geom_order)):
                self._hide_stair_geom(index)
        else:
            self._active_bump_geom_order = self._default_bump_geom_order
            self._hide_all_precision_geoms()
            self._hide_all_stair_geoms()
            for geom_id, spec in zip(self._default_bump_geom_order, layout):
                if isinstance(spec, dict):
                    shape = str(spec.get("shape", "box"))
                    x = float(spec["x"])
                    y = float(spec.get("y", 0.0))
                    height = float(spec["height"])
                    if shape == "cylinder":
                        radius = float(spec["radius"])
                        self._set_cylinder_geom(geom_id, x=x, y=y, radius=radius, height=height)
                        self.bump_regions.append((x - radius, x + radius, y - radius, y + radius, 2.0 * height))
                    else:
                        half_width = float(spec["half_width"])
                        half_length = float(spec["half_length"])
                        self._set_bump_geom(geom_id, x=x, y=y, half_length=half_length, half_width=half_width, height=height)
                        self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, 2.0 * height))
                elif len(spec) == 7:
                    x, y, half_width, height, half_length, pitch, center_z = spec
                    self._set_ramp_geom(
                        geom_id,
                        x=x,
                        y=y,
                        half_length=half_length,
                        half_width=half_width,
                        half_height=height,
                        pitch_rad=pitch,
                        center_z=center_z,
                    )
                    top_z = float(center_z + abs(np.sin(float(pitch))) * half_length + np.cos(float(pitch)) * height)
                    self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, top_z))
                elif len(spec) == 6:
                    x, y, half_width, height, half_length, pitch = spec
                    self._set_ramp_geom(
                        geom_id,
                        x=x,
                        y=y,
                        half_length=half_length,
                        half_width=half_width,
                        half_height=height,
                        pitch_rad=pitch,
                    )
                    top_z = float(height + abs(np.sin(float(pitch))) * half_length + np.cos(float(pitch)) * height)
                    self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, top_z))
                else:
                    x, y, half_width, height, half_length = spec
                    self._set_bump_geom(geom_id, x=x, y=y, half_length=half_length, half_width=half_width, height=height)
                    self.bump_regions.append((x - half_length, x + half_length, y - half_width, y + half_width, 2.0 * height))
            for geom_id in self._default_bump_geom_order[len(layout) :]:
                self._hide_bump_geom(geom_id)

        self._update_corridor_walls()
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _available_terrain_types(self) -> list[str]:
        terrain_types = [self._normalize_terrain_type(name) for name in self.curriculum_terrain_types] or ["rough_flat"]
        if not self.curriculum_enabled:
            return terrain_types
        if terrain_types == ["mixed_course"]:
            return terrain_types
        if self.curriculum_level <= 1:
            return terrain_types[:1]
        if self.curriculum_level <= 3:
            return terrain_types[: min(2, len(terrain_types))]
        if self.curriculum_level <= 5:
            return terrain_types[: min(3, len(terrain_types))]
        return terrain_types

    def _update_curriculum(self) -> None:
        if not self.curriculum_enabled or not self.last_episode_recorded:
            return
        if self.last_episode_success:
            score = 1.0
        elif self.last_episode_terminated or self.last_episode_success_progress <= 0.0:
            score = 0.0
        else:
            score = float(np.clip(self.last_episode_success_progress, 0.0, 1.0))
        self.curriculum_history.append(score)
        if len(self.curriculum_history) < self.curriculum_window:
            self.last_episode_recorded = False
            return
        mean_score = float(np.mean(self.curriculum_history))
        if mean_score >= self.curriculum_promotion_threshold and self.curriculum_level < self.curriculum_max_level:
            self.curriculum_level += 1
            self.curriculum_history.clear()
        elif mean_score <= self.curriculum_demotion_threshold and self.curriculum_level > self.curriculum_min_level:
            self.curriculum_level -= 1
            self.curriculum_history.clear()
        self.last_episode_recorded = False

    def set_curriculum_level(self, level: int) -> None:
        self.curriculum_level = int(np.clip(level, self.curriculum_min_level, self.curriculum_max_level))

    def get_curriculum_level(self) -> int:
        return int(self.curriculum_level)

    def set_fixed_terrain_type(self, terrain_type: str | None) -> None:
        self.fixed_terrain_type = terrain_type

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        if "terrain_level" in options:
            self.set_curriculum_level(int(options["terrain_level"]))
        if "terrain_type" in options:
            self.set_fixed_terrain_type(None if options["terrain_type"] is None else str(options["terrain_type"]))
        self._update_curriculum()
        self._apply_domain_randomization()
        self._apply_terrain_layout(terrain_type=self.fixed_terrain_type)
        self._reset_state()
        self.command_velocity, self.command_yaw_rate = self._sample_command()
        self.previous_action[:] = 0.0
        self.step_count = 0
        return self._get_obs(), {
            "command_velocity": self.command_velocity,
            "command_yaw_rate": self.command_yaw_rate,
            "terrain_type": self.current_terrain_type,
            "terrain_level": float(self.curriculum_level),
            "terrain_family_count": float(len(self._available_terrain_types())),
        }

    def _is_unhealthy(self) -> bool:
        torso_z = float(self.data.qpos[2])
        torso_up = float(self._base_rotation()[2, 2])
        return torso_z < self.config.healthy_z_min or torso_up < self.config.healthy_torso_up_min

    def _hit_corridor_boundary(self) -> bool:
        if self.corridor_half_width is None:
            return False
        return abs(float(self.data.qpos[1])) > float(self.corridor_half_width)

    def _terrain_height_at_world_xy(self, world_x: float, world_y: float) -> float:
        terrain_height = 0.0
        for geom_id in self._active_bump_geom_order:
            geom_pos = self.data.geom_xpos[geom_id]
            geom_size = self.model.geom_size[geom_id]
            geom_type = int(self.model.geom_type[geom_id])
            rot = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                top_half_height = float(geom_size[1])
            else:
                top_half_height = float(geom_size[2])
            top_point = geom_pos + rot @ np.array([0.0, 0.0, top_half_height], dtype=np.float64)
            top_normal = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(top_normal[2])) < 1e-6:
                continue
            plane_z = float(
                top_point[2]
                - (
                    float(top_normal[0]) * (world_x - float(top_point[0]))
                    + float(top_normal[1]) * (world_y - float(top_point[1]))
                )
                / float(top_normal[2])
            )
            candidate_world = np.array([world_x, world_y, plane_z], dtype=np.float64)
            local = rot.T @ (candidate_world - geom_pos)
            on_top_surface = abs(float(local[2]) - top_half_height) <= 1e-4
            if not on_top_surface:
                continue
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                # Cylinder top support should only exist on the circular disk,
                # not across the square AABB used by generic box tests.
                radial_sq = float(local[0] * local[0] + local[1] * local[1])
                inside_support = radial_sq <= float(geom_size[0] * geom_size[0]) + 1e-6
            else:
                inside_support = (
                    abs(float(local[0])) <= float(geom_size[0]) + 1e-6
                    and abs(float(local[1])) <= float(geom_size[1]) + 1e-6
                )
            if inside_support:
                terrain_height = max(terrain_height, plane_z)
        return terrain_height

    def _terrain_slope_deg_at_world_xy(self, world_x: float, world_y: float) -> float:
        best_height = -np.inf
        best_slope = 0.0
        for geom_id in self._active_bump_geom_order:
            geom_pos = self.data.geom_xpos[geom_id]
            geom_size = self.model.geom_size[geom_id]
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                continue
            rot = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            top_half_height = float(geom_size[2])
            top_point = geom_pos + rot @ np.array([0.0, 0.0, top_half_height], dtype=np.float64)
            top_normal = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(top_normal[2])) < 1e-6:
                continue
            plane_z = float(top_point[2] - float(top_normal[0]) * (world_x - float(top_point[0])) / float(top_normal[2]))
            candidate_world = np.array([world_x, world_y, plane_z], dtype=np.float64)
            local = rot.T @ (candidate_world - geom_pos)
            if (
                abs(float(local[2]) - top_half_height) <= 1e-4
                and abs(float(local[0])) <= float(geom_size[0]) + 1e-6
                and abs(float(local[1])) <= float(geom_size[1]) + 1e-6
                and plane_z > best_height
            ):
                dz_dx = -float(top_normal[0]) / float(top_normal[2])
                best_height = plane_z
                best_slope = float(np.rad2deg(np.arctan(dz_dx)))
        return best_slope

    def _query_scandots(self) -> tuple[np.ndarray, np.ndarray]:
        base_pos = self.data.qpos[0:3].copy()
        if self.terrain_obs_mode == "stable_world_patch":
            rot = self._heading_rotation()
            center_offset = rot @ np.array(
                [self.terrain_patch_center_offset[0], self.terrain_patch_center_offset[1], 0.0],
                dtype=np.float64,
            )
            anchor = base_pos + center_offset
            reference_height = base_pos[2]
            if self.terrain_obs_height_reference in {"center_height", "local_ground", "anchor_height"}:
                reference_height = self._terrain_height_at_world_xy(float(anchor[0]), float(anchor[1]))
        else:
            rot = self._base_rotation()
            anchor = base_pos
            reference_height = base_pos[2]
        samples = []
        hit_points = []
        for x_offset, y_offset in self.scandot_points:
            local_offset = np.array([x_offset, y_offset, 0.0], dtype=np.float64)
            world_offset = rot @ local_offset
            world_x = float(anchor[0] + world_offset[0])
            world_y = float(anchor[1] + world_offset[1])
            if self.corridor_half_width is not None and abs(world_y) > float(self.corridor_half_width):
                terrain_height = 0.0
                hit_point = np.array([world_x, world_y, np.nan], dtype=np.float64)
                samples.append(0.0 - reference_height)
                hit_points.append(hit_point)
                continue
            terrain_height = self._terrain_height_at_world_xy(world_x, world_y)
            hit_point = np.array([world_x, world_y, terrain_height], dtype=np.float64)
            samples.append(terrain_height - reference_height)
            hit_points.append(hit_point)
        return np.asarray(samples, dtype=np.float64), np.asarray(hit_points, dtype=np.float64)

    def _scandot_height_samples(self) -> np.ndarray:
        samples, _ = self._query_scandots()
        return samples

    def _terrain_height_samples(self) -> np.ndarray:
        return self._scandot_height_samples()

    def _contact_force_world(self, contact_idx: int) -> np.ndarray:
        contact_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, contact_idx, contact_force)
        contact = self.data.contact[contact_idx]
        contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        return contact_frame @ contact_force[:3]

    def _stair_riser_contact_counts(self) -> tuple[int, int]:
        if self.current_terrain_type != "stairs" or not self._active_stair_layout:
            return 0, 0

        riser_margin = float(self.reward_weights.get("stair_riser_contact_margin", 0.015))
        top_clearance = float(self.reward_weights.get("stair_riser_top_clearance", 0.012))
        total_contacts = 0
        rear_contacts = 0
        active_stair_geom_ids = set(self._stair_geom_order[: len(self._active_stair_layout)])

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in active_stair_geom_ids and geom2 not in active_stair_geom_ids:
                continue

            if geom1 in active_stair_geom_ids:
                stair_geom = geom1
                robot_geom = geom2
            else:
                stair_geom = geom2
                robot_geom = geom1

            robot_body = int(self.model.geom_bodyid[robot_geom])
            if robot_body not in self.robot_body_ids:
                continue

            geom_pos = np.asarray(self.data.geom_xpos[stair_geom], dtype=np.float64)
            geom_rot = np.asarray(self.data.geom_xmat[stair_geom], dtype=np.float64).reshape(3, 3)
            local_contact = geom_rot.T @ (np.asarray(contact.pos, dtype=np.float64) - geom_pos)
            half_length, half_width, half_height = self.model.geom_size[stair_geom]
            touches_front_riser = (
                abs(float(local_contact[0] + half_length)) <= riser_margin
                and abs(float(local_contact[1])) <= float(half_width) + riser_margin
                and abs(float(local_contact[2])) <= max(float(half_height) - top_clearance, 0.0) + riser_margin
            )
            if not touches_front_riser:
                continue

            total_contacts += 1
            if robot_body in self.rear_leg_body_ids:
                rear_contacts += 1

        return total_contacts, rear_contacts

    def _compute_stuck_penalty(self, x_position: float) -> float:
        progress_threshold = float(self.reward_weights.get("stuck_progress_threshold", 0.05))
        patience_s = float(self.reward_weights.get("stuck_patience_s", 2.0))
        ramp_s = float(self.reward_weights.get("stuck_ramp_s", 3.0))
        base_weight = float(self.reward_weights.get("stuck", 0.0))
        if base_weight <= 0.0:
            return 0.0

        self.episode_max_x = max(self.episode_max_x, x_position)
        if self.episode_max_x >= self._progress_reference_x + progress_threshold:
            self._progress_reference_x = self.episode_max_x
            self._stuck_steps = 0
            return 0.0

        self._stuck_steps += 1
        patience_steps = max(1, int(round(patience_s / max(self.config.control_dt, 1e-6))))
        if self._stuck_steps <= patience_steps:
            return 0.0

        ramp_steps = max(1, int(round(ramp_s / max(self.config.control_dt, 1e-6))))
        ramp_progress = float(min(1.0, (self._stuck_steps - patience_steps) / ramp_steps))
        scale = 0.25 + 0.75 * ramp_progress * ramp_progress
        return base_weight * scale

    def _compute_low_speed_penalty(self, x_position: float, forward_velocity: float) -> tuple[float, float, float]:
        avg_weight = float(self.reward_weights.get("avg_speed_penalty", 0.0))
        min_weight = float(self.reward_weights.get("min_speed_penalty", 0.0))
        elapsed_s = max(self.step_count * self.config.control_dt, self.config.control_dt)
        episode_avg_speed = max(0.0, (float(x_position) - self._episode_start_x) / elapsed_s)
        avg_target = float(self.reward_weights.get("target_avg_forward_velocity", 0.0))
        min_target = float(self.reward_weights.get("min_forward_velocity", 0.0))
        avg_penalty = avg_weight * max(0.0, avg_target - episode_avg_speed)
        min_penalty = min_weight * max(0.0, min_target - max(0.0, float(forward_velocity)))
        return float(avg_penalty + min_penalty), float(episode_avg_speed), float(max(0.0, forward_velocity))

    def _compute_step_completion_bonus(self, x_position: float) -> tuple[float, int]:
        if self.current_terrain_type != "stairs" or not self._active_stair_layout:
            return 0.0, self._passed_stair_edge_count
        bonus_weight = float(self.reward_weights.get("step_completion_bonus", 0.0))
        if bonus_weight <= 0.0:
            return 0.0, self._passed_stair_edge_count

        base_margin = float(self.reward_weights.get("step_completion_base_margin", 0.015))
        passed_count = 0
        for x_center, _y, _half_width, _height, half_length in self._active_stair_layout:
            riser_x = float(x_center - half_length)
            if x_position >= riser_x + base_margin:
                passed_count += 1
        gained_steps = max(0, passed_count - self._passed_stair_edge_count)
        self._passed_stair_edge_count = max(self._passed_stair_edge_count, passed_count)
        return bonus_weight * gained_steps, passed_count

    def _highest_step_under_foot(self, foot_pos: np.ndarray) -> int:
        if self.current_terrain_type != "stairs" or not self._active_stair_layout:
            return -1
        top_margin_x = float(self.reward_weights.get("rear_foot_step_margin_x", 0.02))
        top_margin_y = float(self.reward_weights.get("rear_foot_step_margin_y", 0.02))
        top_height_tol = float(self.reward_weights.get("rear_foot_step_height_tol", 0.03))
        best_index = -1
        best_top_z = -np.inf
        for idx, (x_center, y_center, half_width, height, half_length) in enumerate(self._active_stair_layout):
            x_min = float(x_center - half_length + top_margin_x)
            x_max = float(x_center + half_length - top_margin_x)
            y_min = float(y_center - half_width + top_margin_y)
            y_max = float(y_center + half_width - top_margin_y)
            top_z = float(2.0 * height)
            if (
                x_min <= float(foot_pos[0]) <= x_max
                and y_min <= float(foot_pos[1]) <= y_max
                and float(foot_pos[2]) >= top_z - top_height_tol
                and top_z > best_top_z
            ):
                best_top_z = top_z
                best_index = idx
        return best_index

    def _foot_step_levels(self) -> np.ndarray:
        levels = np.full(len(self.foot_site_ids), -1, dtype=np.int32)
        if self.current_terrain_type != "stairs" or not self._active_stair_layout:
            return levels
        for foot_idx, site_id in enumerate(self.foot_site_ids.tolist()):
            levels[foot_idx] = self._highest_step_under_foot(self.data.site_xpos[site_id])
        return levels

    def _support_and_airborne_shaping(self, foot_contact_forces: np.ndarray) -> tuple[float, float, int, int]:
        contact_threshold = float(self.reward_weights.get("contact_force_threshold", 1.0))
        contact_mask = foot_contact_forces[:, 2] >= contact_threshold
        grounded_count = int(np.sum(contact_mask))
        airborne_count = int(len(contact_mask) - grounded_count)

        support_bonus = 0.0
        support_weight = float(self.reward_weights.get("support_bonus", 0.0))
        if support_weight > 0.0 and airborne_count >= 1:
            min_support_feet = int(self.reward_weights.get("support_min_grounded_feet", 2))
            if grounded_count >= min_support_feet:
                support_bonus = support_weight * float(grounded_count - min_support_feet + 1)

        multi_air_penalty = 0.0
        multi_air_weight = float(self.reward_weights.get("multi_air_penalty", 0.0))
        free_air_limit = int(self.reward_weights.get("multi_air_max_feet", 1))
        if multi_air_weight > 0.0 and airborne_count > free_air_limit:
            multi_air_penalty = multi_air_weight * float(airborne_count - free_air_limit)

        return support_bonus, multi_air_penalty, grounded_count, airborne_count

    def _compute_rear_follow_shaping(self, foot_step_levels: np.ndarray, foot_contact_forces: np.ndarray) -> tuple[float, float, int, int]:
        if self.current_terrain_type != "stairs" or not self._active_stair_layout:
            self._rear_lag_steps = 0
            self._previous_rear_mean_lift = 0.0
            return 0.0, 0.0, -1, -1

        front_highest_step = int(np.max(foot_step_levels[self.front_foot_indices]))
        rear_highest_step = int(np.max(foot_step_levels[self.rear_foot_indices]))
        step_gap = front_highest_step - rear_highest_step
        if front_highest_step < 0 or step_gap <= 0:
            self._rear_lag_steps = 0
            self._previous_rear_mean_x = float(np.mean(self.data.site_xpos[self.foot_site_ids[self.rear_foot_indices], 0]))
            self._previous_rear_mean_lift = 0.0
            return 0.0, 0.0, front_highest_step, rear_highest_step

        rear_mean_x = float(np.mean(self.data.site_xpos[self.foot_site_ids[self.rear_foot_indices], 0]))
        rear_delta_x = max(0.0, rear_mean_x - self._previous_rear_mean_x)
        self._previous_rear_mean_x = rear_mean_x
        rear_contact_threshold = float(self.reward_weights.get("contact_force_threshold", 1.0))
        rear_airborne = foot_contact_forces[self.rear_foot_indices, 2] < rear_contact_threshold
        target_step = max(0, front_highest_step)
        target_top_z = float(2.0 * self._active_stair_layout[target_step][3])
        rear_lifts = []
        for foot_idx in self.rear_foot_indices.tolist():
            foot_pos = self.data.site_xpos[self.foot_site_ids[foot_idx]]
            rear_lifts.append(max(0.0, float(foot_pos[2]) - target_top_z))
        rear_mean_lift = float(np.mean(rear_lifts))
        rear_delta_lift = max(0.0, rear_mean_lift - self._previous_rear_mean_lift)
        self._previous_rear_mean_lift = rear_mean_lift

        forward_scale = float(self.reward_weights.get("rear_follow_bonus", 0.0))
        follow_clip = float(self.reward_weights.get("rear_follow_progress_clip", 0.03))
        lift_scale = float(self.reward_weights.get("rear_follow_lift_bonus", 0.0))
        lift_clip = float(self.reward_weights.get("rear_follow_lift_clip", 0.025))
        airborne_scale = float(self.reward_weights.get("rear_follow_airborne_multiplier", 1.25)) if np.any(rear_airborne) else 1.0
        rear_follow_bonus = airborne_scale * (
            forward_scale * min(rear_delta_x, follow_clip)
            + lift_scale * min(rear_delta_lift, lift_clip)
        )

        self._rear_lag_steps += 1
        grace_s = float(self.reward_weights.get("rear_lag_grace_s", 0.6))
        grace_steps = max(1, int(round(grace_s / max(self.config.control_dt, 1e-6))))
        rear_lag_penalty = 0.0
        if self._rear_lag_steps > grace_steps:
            lag_scale = float(self.reward_weights.get("rear_lag_penalty", 0.0))
            lag_multiplier = 1.0 + float(max(0, step_gap - 1))
            rear_lag_penalty = lag_scale * lag_multiplier

        return rear_follow_bonus, rear_lag_penalty, front_highest_step, rear_highest_step

    def _compute_rear_foot_step_bonus(self, foot_contact_forces: np.ndarray) -> tuple[float, int]:
        if self.current_terrain_type != "stairs" or not self._active_stair_layout:
            return 0.0, int(np.max(self._rear_foot_highest_step))
        bonus_weight = float(self.reward_weights.get("rear_foot_step_bonus", 0.0))
        if bonus_weight <= 0.0:
            return 0.0, int(np.max(self._rear_foot_highest_step))

        contact_threshold = float(self.reward_weights.get("contact_force_threshold", 1.0))
        bonus = 0.0
        for rear_slot, foot_idx in enumerate(self.rear_foot_indices.tolist()):
            if float(foot_contact_forces[foot_idx, 2]) < contact_threshold:
                continue
            highest_step = self._highest_step_under_foot(self.data.site_xpos[self.foot_site_ids[foot_idx]])
            if highest_step > int(self._rear_foot_highest_step[rear_slot]):
                bonus += bonus_weight * float(highest_step - int(self._rear_foot_highest_step[rear_slot]))
                self._rear_foot_highest_step[rear_slot] = highest_step
        return bonus, int(np.max(self._rear_foot_highest_step))

    def _contact_force_metrics(self) -> tuple[int, int, np.ndarray, np.ndarray]:
        torso_count = 0
        leg_count = 0
        foot_contact_forces = np.zeros((len(self.foot_body_ids), 3), dtype=np.float64)
        collision_forces = np.zeros((len(self.collision_body_ids), 3), dtype=np.float64)
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            pair = {geom1, geom2}
            touches_bump = bool(pair & self.bump_geom_ids)
            is_self_collision = body1 in self.robot_body_ids and body2 in self.robot_body_ids
            world_force = self._contact_force_world(contact_idx)
            if not is_self_collision:
                if body1 in self.foot_body_id_to_index:
                    foot_contact_forces[self.foot_body_id_to_index[body1]] -= world_force
                if body2 in self.foot_body_id_to_index:
                    foot_contact_forces[self.foot_body_id_to_index[body2]] += world_force
                if body1 in self.collision_body_id_to_index:
                    collision_forces[self.collision_body_id_to_index[body1]] -= world_force
                if body2 in self.collision_body_id_to_index:
                    collision_forces[self.collision_body_id_to_index[body2]] += world_force
            if touches_bump:
                robot_geoms = pair - self.bump_geom_ids
                if robot_geoms & self.torso_geom_ids:
                    torso_count += 1
                elif robot_geoms & self.leg_geom_ids:
                    leg_count += 1
        return torso_count, leg_count, foot_contact_forces, collision_forces

    def _obstacle_clearance_score(self) -> float:
        if not self.bump_regions:
            return 0.0
        margin = float(self.reward_weights["clearance_margin"])
        target_height = float(self.reward_weights["clearance_height"])
        scores = []
        for site_id in self.foot_site_ids:
            foot_pos = self.data.site_xpos[site_id]
            foot_x = float(foot_pos[0])
            foot_z = float(foot_pos[2])
            foot_y = float(foot_pos[1])
            for x_min, x_max, y_min, y_max, bump_top in self.bump_regions:
                if x_min - margin <= foot_x <= x_max + margin and y_min - margin <= foot_y <= y_max + margin:
                    clearance = foot_z - bump_top
                    scores.append(np.clip(clearance / target_height, 0.0, 1.0))
                    break
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _obstacle_path_end_x(self) -> float:
        if not self.bump_regions:
            return 0.0
        return float(max(region[1] for region in self.bump_regions))

    def _success_target_x(self) -> float:
        path_end_x = self._obstacle_path_end_x()
        if path_end_x <= 0.0:
            return float(self.curriculum_success_distance)
        return float(self.curriculum_success_x_fraction * path_end_x)

    def _success_progress(self, x_position: float) -> float:
        target_x = self._success_target_x()
        if target_x <= 1e-6:
            return 0.0
        return float(np.clip(x_position / target_x, 0.0, 1.0))

    def _episode_success(self, x_position: float, terminated: bool) -> bool:
        return bool((x_position >= self._success_target_x()) and not terminated)

    def _foot_horizontal_speeds(self) -> np.ndarray:
        current_positions = self.data.site_xpos[self.foot_site_ids].copy()
        foot_velocities = (current_positions - self.previous_foot_positions) / max(self.config.control_dt, 1e-6)
        self.previous_foot_positions = current_positions
        return np.linalg.norm(foot_velocities[:, :2], axis=1)

    def get_metrics(self) -> dict:
        base_rot = self._base_rotation()
        x_position = float(self.data.qpos[0])
        success_target_x = self._success_target_x()
        return {
            "x_position": x_position,
            "y_position": float(self.data.qpos[1]),
            "torso_height": float(self.data.qpos[2]),
            "forward_velocity": float(self.data.qvel[0]),
            "yaw_rate": float(self.data.qvel[5]),
            "torso_up": float(base_rot[2, 2]),
            "command_velocity": float(self.command_velocity),
            "command_yaw_rate": float(self.command_yaw_rate),
            "body_mass_scale": float(self._current_randomization["body_mass_scale"]),
            "friction_scale": float(self._current_randomization["friction_scale"]),
            "joint_damping_scale": float(self._current_randomization["joint_damping_scale"]),
            "terrain_level": float(self.curriculum_level),
            "terrain_type": self.current_terrain_type,
            "terrain_family_count": float(len(self._available_terrain_types())),
            "terrain_slope_deg": self._terrain_slope_deg_at_world_xy(x_position, float(self.data.qpos[1])),
            "success_target_x": success_target_x,
            "success_progress": self._success_progress(x_position),
            "obstacle_path_end_x": self._obstacle_path_end_x(),
            "episode_max_x": float(self.episode_max_x),
            "stair_steps_cleared": float(self._passed_stair_edge_count),
            "rear_foot_highest_step": float(np.max(self._rear_foot_highest_step)),
        }

    def _make_tracking_camera(self) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.data.qpos[0:3]
        camera.lookat[2] = max(camera.lookat[2], 0.16)
        camera.distance = 1.15
        camera.azimuth = 135.0
        camera.elevation = -18.0
        return camera

    def _make_terrain_overview_camera(self) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        if self.current_terrain_type == "precision_stepping_stones" and self.bump_regions:
            path_end_x = self._obstacle_path_end_x()
            camera.lookat[:] = np.array([0.5 * path_end_x, 0.0, 0.12], dtype=np.float64)
            camera.distance = max(4.5, 0.62 * path_end_x)
        else:
            camera.lookat[:] = np.array([2.35, 0.0, 0.08], dtype=np.float64)
            camera.distance = 3.95
        camera.azimuth = 90.0
        camera.elevation = -28.0
        return camera

    def _scan_window_world_bounds(self) -> tuple[float, float, float, float] | None:
        if not self.use_privileged_terrain or self.scandot_points.size == 0:
            return None
        base_pos = self.data.qpos[0:3].copy()
        rot = self._base_rotation()
        points = []
        for x_offset, y_offset in self.scandot_points:
            local_offset = np.array([x_offset, y_offset, 0.0], dtype=np.float64)
            world_offset = rot @ local_offset
            points.append(base_pos[:2] + world_offset[:2])
        if not points:
            return None
        points = np.asarray(points, dtype=np.float64)
        padding_x = 0.06
        padding_y = 0.04
        return (
            float(np.min(points[:, 0]) - padding_x),
            float(np.max(points[:, 0]) + padding_x),
            float(np.min(points[:, 1]) - padding_y),
            float(np.max(points[:, 1]) + padding_y),
        )

    def _geom_intersects_scan_window(self, geom_id: int, bounds: tuple[float, float, float, float]) -> bool:
        x_min, x_max, y_min, y_max = bounds
        geom_pos = self.data.geom_xpos[geom_id]
        geom_size = self.model.geom_size[geom_id]
        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            geom_x_min = float(geom_pos[0] - geom_size[0])
            geom_x_max = float(geom_pos[0] + geom_size[0])
            geom_y_min = float(geom_pos[1] - geom_size[0])
            geom_y_max = float(geom_pos[1] + geom_size[0])
        else:
            geom_x_min = float(geom_pos[0] - geom_size[0])
            geom_x_max = float(geom_pos[0] + geom_size[0])
            geom_y_min = float(geom_pos[1] - geom_size[1])
            geom_y_max = float(geom_pos[1] + geom_size[1])
        return geom_x_max >= x_min and geom_x_min <= x_max and geom_y_max >= y_min and geom_y_min <= y_max

    def _refresh_preview_highlight(self) -> None:
        if not self._highlight_geom_ids:
            return
        for geom_id, rgba in self._highlight_original_rgba.items():
            self.model.geom_rgba[geom_id] = rgba
        if not self._preview_highlight_enabled:
            return

        # A soft warm tint keeps preview obstacles visible without overpowering the video.
        base_preview_rgba = np.array([0.95, 0.62, 0.28, 0.78], dtype=np.float32)
        for geom_id in self._highlight_geom_ids:
            self.model.geom_rgba[geom_id] = base_preview_rgba

        scan_bounds = self._scan_window_world_bounds()
        if not self._privileged_scan_highlight_enabled or scan_bounds is None:
            return

        # The privileged terrain window gets a cooler tint so the sensed region is easy to spot.
        scan_rgba = np.array([0.55, 0.82, 0.92, 0.88], dtype=np.float32)
        for geom_id in self._highlight_geom_ids:
            if self._geom_intersects_scan_window(geom_id, scan_bounds):
                self.model.geom_rgba[geom_id] = scan_rgba

    def set_preview_highlight(self, enabled: bool) -> None:
        self._preview_highlight_enabled = bool(enabled)
        self._refresh_preview_highlight()

    def set_scandot_overlay(self, enabled: bool) -> None:
        self._scandot_overlay_enabled = bool(enabled)

    def set_stair_edge_overlay(self, enabled: bool) -> None:
        self._stair_edge_overlay_enabled = bool(enabled)

    def _append_scandot_markers(self) -> None:
        if not self._scandot_overlay_enabled or not self.use_privileged_terrain or self.scandot_points.size == 0:
            return
        if self.renderer is None:
            return
        scene = self.renderer.scene
        _, hit_points = self._query_scandots()
        marker_rgba = np.array([0.10, 0.85, 0.95, 0.95], dtype=np.float32)
        identity_mat = np.eye(3, dtype=np.float32).reshape(-1)
        for hit_point in hit_points:
            if np.isnan(hit_point[2]):
                continue
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            marker_pos = np.array(hit_point, dtype=np.float32)
            marker_pos[2] += 0.012
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.012, 0.012], dtype=np.float32),
                marker_pos,
                identity_mat,
                marker_rgba,
            )

    def _append_stair_edge_markers(self) -> None:
        if not self._stair_edge_overlay_enabled or self.current_terrain_type not in {"stairs", "precision_stepping_stones", "random_slope_up_down"}:
            return
        if self.renderer is None:
            return
        scene = self.renderer.scene
        identity_mat = np.eye(3, dtype=np.float32).reshape(-1)
        edge_rgba = np.array([0.98, 0.98, 0.98, 0.92], dtype=np.float32)
        z_lift = 0.003
        cylinder_segments = 16
        cylinder_ring_radius = 0.006

        def append_capsule(p0: np.ndarray, p1: np.ndarray, radius: float = 0.006) -> bool:
            if scene.ngeom >= scene.maxgeom:
                return False
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([radius, 0.0, 0.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                identity_mat,
                edge_rgba,
            )
            connector_fn = getattr(mujoco, "mjv_makeConnector", None)
            if connector_fn is None:
                connector_fn = getattr(mujoco, "mjv_connector")
            if getattr(mujoco, "mjv_makeConnector", None) is not None:
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
                connector_fn(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p0.astype(np.float64), p1.astype(np.float64))
            return True

        for geom_id, (x_min, x_max, y_min, y_max, bump_top) in zip(self._active_bump_geom_order, self.bump_regions):
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                radius = float(self.model.geom_size[geom_id][0])
                z_center = float(bump_top + z_lift)
                if scene.ngeom + cylinder_segments >= scene.maxgeom:
                    break
                for seg_idx in range(cylinder_segments):
                    theta0 = 2.0 * np.pi * seg_idx / cylinder_segments
                    theta1 = 2.0 * np.pi * (seg_idx + 1) / cylinder_segments
                    p0 = np.array(
                        [self.data.geom_xpos[geom_id][0] + radius * np.cos(theta0), self.data.geom_xpos[geom_id][1] + radius * np.sin(theta0), z_center],
                        dtype=np.float32,
                    )
                    p1 = np.array(
                        [self.data.geom_xpos[geom_id][0] + radius * np.cos(theta1), self.data.geom_xpos[geom_id][1] + radius * np.sin(theta1), z_center],
                        dtype=np.float32,
                    )
                    append_capsule(p0, p1, cylinder_ring_radius)
                continue
            if scene.ngeom + 4 >= scene.maxgeom:
                break
            geom_pos = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
            geom_rot = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            sx, sy, sz = [float(value) for value in self.model.geom_size[geom_id]]
            top_normal = geom_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            local_vertices = [
                np.array([-sx, -sy, sz], dtype=np.float64),
                np.array([sx, -sy, sz], dtype=np.float64),
                np.array([sx, sy, sz], dtype=np.float64),
                np.array([-sx, sy, sz], dtype=np.float64),
            ]
            vertices = [geom_pos + geom_rot @ vertex + z_lift * top_normal for vertex in local_vertices]
            for start_idx, end_idx in ((0, 1), (1, 2), (2, 3), (3, 0)):
                if not append_capsule(vertices[start_idx].astype(np.float32), vertices[end_idx].astype(np.float32)):
                    break

    def render_frame(self, width: int = 640, height: int = 480, camera: str = "tracking") -> np.ndarray:
        if self.renderer is None or self.renderer_width != width or self.renderer_height != height:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.renderer_width = width
            self.renderer_height = height
        self._refresh_preview_highlight()
        camera_spec: str | mujoco.MjvCamera
        if camera == "tracking":
            camera_spec = self._make_tracking_camera()
        elif camera == "terrain_overview":
            camera_spec = self._make_terrain_overview_camera()
        else:
            camera_spec = camera
        self.renderer.update_scene(self.data, camera=camera_spec)
        self._append_stair_edge_markers()
        self._append_scandot_markers()
        return self.renderer.render()

    def render_depth_image(
        self,
        width: int = 87,
        height: int = 58,
        camera: str = "front_camera",
        near: float = 0.05,
        far: float = 2.0,
        normalize: bool = True,
    ) -> np.ndarray:
        if self.depth_renderer is None or self.depth_renderer_width != width or self.depth_renderer_height != height:
            if self.depth_renderer is not None:
                self.depth_renderer.close()
            self.depth_renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.depth_renderer_width = width
            self.depth_renderer_height = height
        self.depth_renderer.enable_depth_rendering()
        try:
            self.depth_renderer.update_scene(self.data, camera=camera)
            depth = np.asarray(self.depth_renderer.render(), dtype=np.float32)
        finally:
            self.depth_renderer.disable_depth_rendering()
        far = float(far)
        near = float(near)
        depth = np.nan_to_num(depth, nan=far, posinf=far, neginf=far)
        depth = np.clip(depth, near, far)
        if normalize:
            depth = (depth - near) / max(far - near, 1e-6)
        return depth[None, :, :].astype(np.float32)

    def save_video(self, frames: list[np.ndarray], output_path: str | Path, fps: int = 30) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        target = self.default_pose + self.config.action_scale * action
        self.data.ctrl[:] = target
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        hit_corridor = self._hit_corridor_boundary()
        terminated = self._is_unhealthy() or hit_corridor
        truncated = self.step_count >= self.max_steps

        forward_velocity = float(self.data.qvel[0])
        yaw_rate = float(self.data.qvel[5])
        torso_contacts, leg_contacts, foot_contact_forces, collision_forces = self._contact_force_metrics()
        stair_riser_contact_count, rear_riser_contact_count = self._stair_riser_contact_counts()
        foot_horizontal_speeds = self._foot_horizontal_speeds()
        foot_force_delta = foot_contact_forces - self.previous_foot_contact_forces
        foot_jerk_metric = float(np.sum(np.linalg.norm(foot_force_delta, axis=1)))
        foot_drag_metric = float(
            np.sum(
                foot_horizontal_speeds[
                    foot_contact_forces[:, 2] >= float(self.reward_weights["contact_force_threshold"])
                ]
            )
        )
        collision_metric = float(
            np.sum(
                np.linalg.norm(collision_forces, axis=1) >= float(self.reward_weights["collision_force_threshold"])
            )
        )
        x_position = float(self.data.qpos[0])
        foot_step_levels = self._foot_step_levels()
        support_bonus, multi_air_penalty, grounded_count, airborne_count = self._support_and_airborne_shaping(
            foot_contact_forces
        )
        rear_follow_bonus, rear_lag_penalty, front_highest_step, rear_highest_step = self._compute_rear_follow_shaping(
            foot_step_levels,
            foot_contact_forces,
        )
        stuck_penalty = self._compute_stuck_penalty(x_position)
        low_speed_penalty, episode_avg_forward_velocity, clipped_forward_velocity = self._compute_low_speed_penalty(
            x_position,
            forward_velocity,
        )
        step_completion_bonus, passed_step_count = self._compute_step_completion_bonus(x_position)
        rear_foot_step_bonus, rear_foot_highest_step = self._compute_rear_foot_step_bonus(foot_contact_forces)
        stair_riser_collision_penalty = float(self.reward_weights.get("stair_riser_collision", 0.0)) * float(
            stair_riser_contact_count
        )
        rear_leg_riser_collision_penalty = float(
            self.reward_weights.get("rear_leg_riser_collision", 0.0)
        ) * float(rear_riser_contact_count)
        if front_highest_step >= 0 and rear_highest_step >= 0 and front_highest_step > rear_highest_step:
            rear_leg_riser_collision_penalty *= float(self.reward_weights.get("rear_riser_lag_multiplier", 1.5))
        reward_terms = compute_teacher_walk_reward(
            command_velocity=self.command_velocity,
            command_yaw_rate=self.command_yaw_rate,
            forward_velocity=forward_velocity,
            yaw_rate=yaw_rate,
            actuator_torque=self.data.qfrc_actuator[self.joint_qvel_adr],
            joint_velocity=self.data.qvel[self.joint_qvel_adr],
            foot_contact_forces=foot_contact_forces,
            previous_foot_contact_forces=self.previous_foot_contact_forces,
            foot_horizontal_speeds=foot_horizontal_speeds,
            collision_forces=collision_forces,
            terminated=terminated,
            weights=self.reward_weights,
            stuck_penalty=stuck_penalty,
            stair_riser_collision_penalty=stair_riser_collision_penalty,
            rear_leg_riser_collision_penalty=rear_leg_riser_collision_penalty,
            step_completion_bonus=step_completion_bonus,
            rear_foot_step_bonus=rear_foot_step_bonus,
            rear_follow_bonus=rear_follow_bonus,
            rear_lag_penalty=rear_lag_penalty,
            support_bonus=support_bonus,
            multi_air_penalty=multi_air_penalty,
            low_speed_penalty=low_speed_penalty,
        )
        self.previous_foot_contact_forces = foot_contact_forces.copy()
        self.previous_action = action.copy()

        metrics = self.get_metrics()
        episode_success = self._episode_success(metrics["x_position"], bool(terminated))
        info = {
            **metrics,
            "reward_survival_bonus": reward_terms.survival_bonus,
            "reward_command_tracking": reward_terms.command_tracking_reward,
            "reward_absolute_work_penalty": reward_terms.absolute_work_penalty,
            "reward_foot_drag_penalty": reward_terms.foot_drag_penalty,
            "reward_foot_jerk_penalty": reward_terms.foot_jerk_penalty,
            "reward_collision_penalty": reward_terms.collision_penalty,
            "reward_stuck_penalty": reward_terms.stuck_penalty,
            "reward_stair_riser_collision_penalty": reward_terms.stair_riser_collision_penalty,
            "reward_rear_leg_riser_collision_penalty": reward_terms.rear_leg_riser_collision_penalty,
            "reward_step_completion_bonus": reward_terms.step_completion_bonus,
            "reward_rear_foot_step_bonus": reward_terms.rear_foot_step_bonus,
            "reward_rear_follow_bonus": reward_terms.rear_follow_bonus,
            "reward_rear_lag_penalty": reward_terms.rear_lag_penalty,
            "reward_support_bonus": reward_terms.support_bonus,
            "reward_multi_air_penalty": reward_terms.multi_air_penalty,
            "reward_low_speed_penalty": reward_terms.low_speed_penalty,
            "reward_fall_penalty": reward_terms.fall_penalty,
            "torso_obstacle_contacts": float(torso_contacts),
            "leg_obstacle_contacts": float(leg_contacts),
            "foot_drag_metric": foot_drag_metric,
            "foot_jerk_metric": foot_jerk_metric,
            "collision_metric": collision_metric,
            "stair_riser_contact_count": float(stair_riser_contact_count),
            "rear_riser_contact_count": float(rear_riser_contact_count),
            "stuck_steps": float(self._stuck_steps),
            "step_completion_count": float(passed_step_count),
            "rear_foot_step_level": float(rear_foot_highest_step),
            "front_foot_step_level": float(front_highest_step),
            "rear_follow_step_gap": float(front_highest_step - rear_highest_step),
            "rear_lag_steps": float(self._rear_lag_steps),
            "grounded_foot_count": float(grounded_count),
            "airborne_foot_count": float(airborne_count),
            "episode_avg_forward_velocity": float(episode_avg_forward_velocity),
            "clipped_forward_velocity": float(clipped_forward_velocity),
            "hit_corridor_boundary": hit_corridor,
            "episode_success": float(episode_success),
        }
        if terminated or truncated:
            self.last_episode_distance = metrics["x_position"]
            self.last_episode_success_x = metrics["success_target_x"]
            self.last_episode_success_progress = metrics["success_progress"]
            self.last_episode_success = bool(episode_success)
            self.last_episode_terminated = bool(terminated)
            self.last_episode_truncated = bool(truncated)
            self.last_episode_recorded = True
        return self._get_obs(), reward_terms.total, terminated, truncated, info

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.renderer_width = None
            self.renderer_height = None
        if self.depth_renderer is not None:
            self.depth_renderer.close()
            self.depth_renderer = None
            self.depth_renderer_width = None
            self.depth_renderer_height = None


def make_env(config_dict: dict) -> PupperLikeEnv:
    model_path = Path(config_dict["model_path"])
    if not model_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        model_path = repo_root / model_path
    env_config = EnvConfig(
        model_path=str(model_path),
        control_dt=float(config_dict.get("control_dt", 0.02)),
        episode_length_s=float(config_dict.get("episode_length_s", 10.0)),
        action_scale=float(config_dict.get("action_scale", 0.35)),
        reset_noise_scale=float(config_dict.get("reset_noise_scale", 0.03)),
        command_min=float(config_dict.get("command_min", 0.0)),
        command_max=float(config_dict.get("command_max", 0.6)),
        command_yaw_min=float(config_dict.get("command_yaw_min", 0.0)),
        command_yaw_max=float(config_dict.get("command_yaw_max", 0.0)),
        healthy_z_min=float(config_dict.get("healthy_z_min", 0.09)),
        healthy_torso_up_min=float(config_dict.get("healthy_torso_up_min", 0.45)),
        reward_weights=config_dict.get("reward_weights"),
        seed=int(config_dict.get("seed", 0)),
        use_privileged_terrain=bool(config_dict.get("use_privileged_terrain", False)),
        terrain_obs_mode=str(config_dict.get("terrain_obs_mode", "legacy")),
        terrain_obs_height_reference=str(config_dict.get("terrain_obs_height_reference", "base_z")),
        terrain_patch_rows=int(config_dict.get("terrain_patch_rows", 0)),
        terrain_patch_cols=int(config_dict.get("terrain_patch_cols", 0)),
        terrain_patch_dx=float(config_dict.get("terrain_patch_dx", 0.10)),
        terrain_patch_dy=float(config_dict.get("terrain_patch_dy", 0.10)),
        terrain_patch_center_offset_x=float(config_dict.get("terrain_patch_center_offset_x", 0.0)),
        terrain_patch_center_offset_y=float(config_dict.get("terrain_patch_center_offset_y", 0.0)),
        scandot_layout=str(
            config_dict.get(
                "scandot_layout",
                "custom" if config_dict.get("scandot_points") else "grid",
            )
        ),
        scandot_points=tuple(tuple(point) for point in config_dict.get("scandot_points", [])),
        footstep_scandot_points=tuple(tuple(point) for point in config_dict.get("footstep_scandot_points", [])),
        scandot_min_forward_offset=float(config_dict.get("scandot_min_forward_offset", 0.18)),
        terrain_scan_x=tuple(config_dict.get("terrain_scan_x", [0.18, 0.32, 0.48, 0.66])),
        terrain_scan_y=tuple(config_dict.get("terrain_scan_y", [-0.10, -0.04, 0.04, 0.10])),
        terrain_scan_origin_z=float(config_dict.get("terrain_scan_origin_z", 0.35)),
        terrain_scan_max_dist=float(config_dict.get("terrain_scan_max_dist", 1.2)),
        corridor_half_width=(
            None if config_dict.get("corridor_half_width") is None else float(config_dict.get("corridor_half_width"))
        ),
        observation_mode=str(config_dict.get("observation_mode", "flat")),
        include_privileged_obs=bool(config_dict.get("include_privileged_obs", False)),
        observation_noise_std=float(config_dict.get("observation_noise_std", 0.0)),
        privileged_randomization=config_dict.get("privileged_randomization"),
        curriculum=config_dict.get("curriculum"),
        terrain_generation=config_dict.get("terrain_generation"),
    )
    return PupperLikeEnv(env_config)
