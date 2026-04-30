from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.train_teacher import build_model, is_recurrent_config, load_config, make_single_env


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StudentObs:
    depth: torch.Tensor
    proprio: torch.Tensor
    command: torch.Tensor


class DepthConvNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], feature_dim: int):
        super().__init__()
        channels, height, width = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flat_dim = int(self.conv(dummy).shape[1])
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, int(feature_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(depth))


class PaperLikeStudentPolicy(nn.Module):
    def __init__(
        self,
        *,
        proprio_dim: int,
        command_dim: int,
        action_dim: int,
        depth_shape: tuple[int, int, int],
        depth_feature_dim: int = 128,
        gru_hidden_size: int = 256,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.command_dim = int(command_dim)
        self.action_dim = int(action_dim)
        self.depth_shape = tuple(int(v) for v in depth_shape)
        self.depth_feature_dim = int(depth_feature_dim)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_layers = int(gru_layers)
        self.depth_encoder = DepthConvNet(self.depth_shape, self.depth_feature_dim)
        self.gru = nn.GRU(
            self.proprio_dim + self.depth_feature_dim + self.command_dim,
            self.gru_hidden_size,
            num_layers=self.gru_layers,
        )
        self.action_head = nn.Linear(self.gru_hidden_size, self.action_dim)

    def encode_step(self, depth: torch.Tensor, proprio: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        z_depth = self.depth_encoder(depth)
        return torch.cat([proprio, z_depth, command], dim=-1)

    def forward_sequence(
        self,
        depth: torch.Tensor,
        proprio: torch.Tensor,
        command: torch.Tensor,
        episode_starts: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = depth.shape[:2]
        flat_features = self.encode_step(
            depth.reshape(batch_size * seq_len, *depth.shape[2:]),
            proprio.reshape(batch_size * seq_len, -1),
            command.reshape(batch_size * seq_len, -1),
        )
        features = flat_features.reshape(batch_size, seq_len, -1).transpose(0, 1)
        if hidden is None:
            hidden = torch.zeros(self.gru_layers, batch_size, self.gru_hidden_size, device=depth.device)
        outputs = []
        if episode_starts is None:
            gru_out, hidden = self.gru(features, hidden)
            actions = self.action_head(gru_out.transpose(0, 1))
            return actions, hidden
        starts = episode_starts.transpose(0, 1).float()
        for step_features, step_starts in zip(features, starts, strict=True):
            hidden = hidden * (1.0 - step_starts).view(1, batch_size, 1)
            out, hidden = self.gru(step_features.unsqueeze(0), hidden)
            outputs.append(out)
        gru_out = torch.cat(outputs, dim=0)
        actions = self.action_head(gru_out.transpose(0, 1))
        return actions, hidden

    @torch.no_grad()
    def predict(
        self,
        depth: np.ndarray,
        proprio: np.ndarray,
        command: np.ndarray,
        hidden: torch.Tensor | None,
        episode_start: bool,
        device: torch.device,
    ) -> tuple[np.ndarray, torch.Tensor]:
        self.eval()
        depth_t = torch.as_tensor(depth[None, None], dtype=torch.float32, device=device)
        proprio_t = torch.as_tensor(proprio[None, None], dtype=torch.float32, device=device)
        command_t = torch.as_tensor(command[None, None], dtype=torch.float32, device=device)
        starts = torch.as_tensor([[episode_start]], dtype=torch.float32, device=device)
        action, hidden = self.forward_sequence(depth_t, proprio_t, command_t, starts, hidden)
        action_np = action[0, 0].detach().cpu().numpy()
        return np.clip(action_np, -1.0, 1.0), hidden


IMU_VARIANT_SLICES = {
    "imu_clean": (27, 33),
    "base_ang_vel_only": (27, 30),
    "projected_gravity_only": (30, 33),
}


class ResidualGatedStudentPolicy(PaperLikeStudentPolicy):
    def __init__(
        self,
        *,
        proprio_dim: int,
        command_dim: int,
        action_dim: int,
        depth_shape: tuple[int, int, int],
        depth_feature_dim: int = 128,
        gru_hidden_size: int = 256,
        gru_layers: int = 1,
        imu_variant: str = "imu_clean",
        imu_hidden_dim: int = 128,
        residual_beta_init: float = 0.01,
    ):
        super().__init__(
            proprio_dim=proprio_dim,
            command_dim=command_dim,
            action_dim=action_dim,
            depth_shape=depth_shape,
            depth_feature_dim=depth_feature_dim,
            gru_hidden_size=gru_hidden_size,
            gru_layers=gru_layers,
        )
        if imu_variant not in IMU_VARIANT_SLICES:
            raise ValueError(f"unknown imu_variant={imu_variant!r}; valid values are {sorted(IMU_VARIANT_SLICES)}")
        self.imu_variant = str(imu_variant)
        self.imu_slice = IMU_VARIANT_SLICES[self.imu_variant]
        self.imu_dim = int(self.imu_slice[1] - self.imu_slice[0])
        self.imu_encoder = nn.Sequential(
            nn.Linear(self.imu_dim, int(imu_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(imu_hidden_dim), self.depth_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.depth_feature_dim, self.depth_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.depth_feature_dim, self.depth_feature_dim),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(float(residual_beta_init), dtype=torch.float32))
        self._last_gate_stats: dict[str, float] = {}

    def _imu_from_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        start, end = self.imu_slice
        if proprio.shape[-1] < end:
            raise ValueError(f"proprio dim {proprio.shape[-1]} is too small for {self.imu_variant} slice {start}:{end}")
        return proprio[..., start:end]

    def encode_step(self, depth: torch.Tensor, proprio: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        z_vis = self.depth_encoder(depth)
        z_imu = self.imu_encoder(self._imu_from_proprio(proprio))
        alpha = self.gate_mlp(torch.cat([z_vis, z_imu], dim=-1))
        z_gate = alpha * z_vis + (1.0 - alpha) * z_imu
        z_new = z_vis + self.beta * (z_gate - z_vis)
        if not self.training:
            self._last_gate_stats = _tensor_gate_stats(alpha, self.beta)
        return torch.cat([proprio, z_new, command], dim=-1)

    @torch.no_grad()
    def gate_stats_for_batch(self, depth: torch.Tensor, proprio: torch.Tensor) -> dict[str, float]:
        was_training = self.training
        self.eval()
        z_vis = self.depth_encoder(depth)
        z_imu = self.imu_encoder(self._imu_from_proprio(proprio))
        alpha = self.gate_mlp(torch.cat([z_vis, z_imu], dim=-1))
        stats = _tensor_gate_stats(alpha, self.beta)
        if was_training:
            self.train()
        return stats

    def latest_gate_stats(self) -> dict[str, float]:
        return dict(self._last_gate_stats) if self._last_gate_stats else _empty_gate_stats(self.beta)


class ResidualBilinearStudentPolicy(PaperLikeStudentPolicy):
    def __init__(
        self,
        *,
        proprio_dim: int,
        command_dim: int,
        action_dim: int,
        depth_shape: tuple[int, int, int],
        depth_feature_dim: int = 128,
        gru_hidden_size: int = 256,
        gru_layers: int = 1,
        imu_variant: str = "imu_clean",
        imu_hidden_dim: int = 64,
        bilinear_rank: int = 64,
        residual_beta_init: float = 0.01,
    ):
        super().__init__(
            proprio_dim=proprio_dim,
            command_dim=command_dim,
            action_dim=action_dim,
            depth_shape=depth_shape,
            depth_feature_dim=depth_feature_dim,
            gru_hidden_size=gru_hidden_size,
            gru_layers=gru_layers,
        )
        if imu_variant != "imu_clean":
            raise ValueError("ResidualBilinearStudentPolicy currently supports only imu_variant='imu_clean'")
        self.imu_variant = str(imu_variant)
        self.imu_slice = IMU_VARIANT_SLICES[self.imu_variant]
        self.imu_dim = int(self.imu_slice[1] - self.imu_slice[0])
        self.bilinear_rank = int(bilinear_rank)
        self.imu_encoder = nn.Sequential(
            nn.Linear(self.imu_dim, int(imu_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(imu_hidden_dim), self.depth_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.vis_bilinear_proj = nn.Linear(self.depth_feature_dim, self.bilinear_rank)
        self.imu_bilinear_proj = nn.Linear(self.depth_feature_dim, self.bilinear_rank)
        self.bilinear_out_proj = nn.Linear(self.bilinear_rank, self.depth_feature_dim)
        self.beta = nn.Parameter(torch.tensor(float(residual_beta_init), dtype=torch.float32))
        self._last_bilinear_stats: dict[str, float] = {}

    def _imu_from_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        start, end = self.imu_slice
        if proprio.shape[-1] < end:
            raise ValueError(f"proprio dim {proprio.shape[-1]} is too small for {self.imu_variant} slice {start}:{end}")
        return proprio[..., start:end]

    def _bilinear_features(self, depth: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_vis = self.depth_encoder(depth)
        z_imu = self.imu_encoder(self._imu_from_proprio(proprio))
        h = self.vis_bilinear_proj(z_vis) * self.imu_bilinear_proj(z_imu)
        delta = self.bilinear_out_proj(h)
        z_new = z_vis + self.beta * delta
        return z_vis, h, delta, z_new

    def encode_step(self, depth: torch.Tensor, proprio: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        z_vis, h, delta, z_new = self._bilinear_features(depth, proprio)
        if not self.training:
            self._last_bilinear_stats = _tensor_bilinear_stats(z_vis, h, delta, self.beta)
        return torch.cat([proprio, z_new, command], dim=-1)

    @torch.no_grad()
    def bilinear_stats_for_batch(self, depth: torch.Tensor, proprio: torch.Tensor) -> dict[str, float]:
        was_training = self.training
        self.eval()
        z_vis, h, delta, _z_new = self._bilinear_features(depth, proprio)
        stats = _tensor_bilinear_stats(z_vis, h, delta, self.beta)
        if was_training:
            self.train()
        return stats

    def latest_bilinear_stats(self) -> dict[str, float]:
        return dict(self._last_bilinear_stats) if self._last_bilinear_stats else _empty_bilinear_stats(self.beta)


def _tensor_gate_stats(alpha: torch.Tensor, beta: torch.Tensor) -> dict[str, float]:
    alpha_detached = alpha.detach()
    return {
        "beta_value": float(beta.detach().cpu()),
        "alpha_mean": float(alpha_detached.mean().cpu()),
        "alpha_std": float(alpha_detached.std(unbiased=False).cpu()),
        "alpha_min": float(alpha_detached.min().cpu()),
        "alpha_max": float(alpha_detached.max().cpu()),
    }


def _empty_gate_stats(beta: torch.Tensor | None = None) -> dict[str, float]:
    beta_value = 0.0 if beta is None else float(beta.detach().cpu())
    return {"beta_value": beta_value, "alpha_mean": 0.0, "alpha_std": 0.0, "alpha_min": 0.0, "alpha_max": 0.0}


def _tensor_bilinear_stats(z_vis: torch.Tensor, h: torch.Tensor, delta: torch.Tensor, beta: torch.Tensor) -> dict[str, float]:
    z_vis_detached = z_vis.detach()
    h_detached = h.detach()
    delta_detached = delta.detach()
    beta_detached = beta.detach()
    delta_norm = float(delta_detached.norm(dim=-1).mean().cpu())
    z_vis_norm = float(z_vis_detached.norm(dim=-1).mean().cpu())
    scaled_delta_norm = float((beta_detached * delta_detached).norm(dim=-1).mean().cpu())
    return {
        "beta_value": float(beta_detached.cpu()),
        "bilinear_delta_norm": delta_norm,
        "bilinear_h_mean": float(h_detached.mean().cpu()),
        "bilinear_h_std": float(h_detached.std(unbiased=False).cpu()),
        "z_vis_norm": z_vis_norm,
        "delta_to_vis_ratio": float(scaled_delta_norm / (z_vis_norm + 1e-8)),
    }


def _empty_bilinear_stats(beta: torch.Tensor | None = None) -> dict[str, float]:
    beta_value = 0.0 if beta is None else float(beta.detach().cpu())
    return {
        "beta_value": beta_value,
        "bilinear_delta_norm": 0.0,
        "bilinear_h_mean": 0.0,
        "bilinear_h_std": 0.0,
        "z_vis_norm": 0.0,
        "delta_to_vis_ratio": 0.0,
    }


def get_student_gate_stats(student, depth: torch.Tensor | None = None, proprio: torch.Tensor | None = None) -> dict[str, float]:
    if isinstance(student, ResidualGatedStudentPolicy):
        if depth is not None and proprio is not None:
            flat_depth = depth.reshape(-1, *depth.shape[2:]) if depth.ndim == 5 else depth
            flat_proprio = proprio.reshape(-1, proprio.shape[-1]) if proprio.ndim == 3 else proprio
            return student.gate_stats_for_batch(flat_depth, flat_proprio)
        return student.latest_gate_stats()
    return _empty_gate_stats(None)


def get_student_bilinear_stats(student, depth: torch.Tensor | None = None, proprio: torch.Tensor | None = None) -> dict[str, float]:
    if isinstance(student, ResidualBilinearStudentPolicy):
        if depth is not None and proprio is not None:
            flat_depth = depth.reshape(-1, *depth.shape[2:]) if depth.ndim == 5 else depth
            flat_proprio = proprio.reshape(-1, proprio.shape[-1]) if proprio.ndim == 3 else proprio
            return student.bilinear_stats_for_batch(flat_depth, flat_proprio)
        return student.latest_bilinear_stats()
    return _empty_bilinear_stats(None)


def build_student_from_config(config: dict, observation_space: gym.spaces.Dict, action_space: gym.spaces.Box) -> PaperLikeStudentPolicy:
    student_cfg = config["student"]
    depth_cfg = student_cfg["depth"]
    common_kwargs = dict(
        proprio_dim=int(observation_space.spaces["proprio"].shape[0]),
        command_dim=int(observation_space.spaces["command"].shape[0]),
        action_dim=int(action_space.shape[0]),
        depth_shape=(1, int(depth_cfg["height"]), int(depth_cfg["width"])),
        depth_feature_dim=int(student_cfg.get("depth_feature_dim", 128)),
        gru_hidden_size=int(student_cfg.get("gru_hidden_size", 256)),
        gru_layers=int(student_cfg.get("gru_layers", 1)),
    )
    architecture = str(student_cfg.get("architecture", "paper_like"))
    if architecture == "paper_like":
        return PaperLikeStudentPolicy(**common_kwargs)
    if architecture == "residual_gated":
        return ResidualGatedStudentPolicy(
            **common_kwargs,
            imu_variant=str(student_cfg.get("imu_variant", "imu_clean")),
            imu_hidden_dim=int(student_cfg.get("imu_hidden_dim", student_cfg.get("depth_feature_dim", 128))),
            residual_beta_init=float(student_cfg.get("residual_beta_init", 0.01)),
        )
    if architecture == "residual_bilinear":
        return ResidualBilinearStudentPolicy(
            **common_kwargs,
            imu_variant=str(student_cfg.get("imu_variant", "imu_clean")),
            imu_hidden_dim=int(student_cfg.get("imu_hidden_dim", 64)),
            bilinear_rank=int(student_cfg.get("bilinear_rank", 64)),
            residual_beta_init=float(student_cfg.get("residual_beta_init", 0.01)),
        )
    raise ValueError(f"unknown student architecture={architecture!r}")


def load_paper_like_weights_into_gated_student(paper_like_checkpoint: str | Path, gated_student: ResidualGatedStudentPolicy, device: torch.device | str = "cpu") -> dict[str, list[str]]:
    return _load_paper_like_shared_weights(paper_like_checkpoint, gated_student, device=device, target_name="gated student")


def load_paper_like_weights_into_bilinear_student(
    paper_like_checkpoint: str | Path,
    bilinear_student: ResidualBilinearStudentPolicy,
    device: torch.device | str = "cpu",
) -> dict[str, list[str]]:
    return _load_paper_like_shared_weights(paper_like_checkpoint, bilinear_student, device=device, target_name="bilinear student")


def _load_paper_like_shared_weights(
    paper_like_checkpoint: str | Path,
    target_student: PaperLikeStudentPolicy,
    *,
    device: torch.device | str = "cpu",
    target_name: str,
) -> dict[str, list[str]]:
    checkpoint_path = resolve_path(paper_like_checkpoint)
    payload = torch.load(checkpoint_path, map_location=device)
    source_state = payload.get("model_state_dict", payload)
    target_state = target_student.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    allowed_prefixes = ("depth_encoder.", "gru.", "action_head.")
    with torch.no_grad():
        for key, source_value in source_state.items():
            if not key.startswith(allowed_prefixes):
                skipped.append(f"{key}: not a paper-like shared module")
                continue
            if key not in target_state:
                skipped.append(f"{key}: missing in {target_name}")
                continue
            if tuple(source_value.shape) != tuple(target_state[key].shape):
                skipped.append(f"{key}: shape {tuple(source_value.shape)} != {tuple(target_state[key].shape)}")
                continue
            target_state[key].copy_(source_value.to(device=target_state[key].device, dtype=target_state[key].dtype))
            copied.append(key)
    return {"copied": copied, "skipped": skipped, "checkpoint": [str(checkpoint_path)]}


def copy_teacher_weights_to_student(teacher_model, student: PaperLikeStudentPolicy) -> dict[str, list[str]]:
    teacher_state = teacher_model.policy.state_dict()
    student_state = student.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    mapping = {
        "lstm_actor.weight_ih_l0": "gru.weight_ih_l0",
        "lstm_actor.weight_hh_l0": "gru.weight_hh_l0",
        "lstm_actor.bias_ih_l0": "gru.bias_ih_l0",
        "lstm_actor.bias_hh_l0": "gru.bias_hh_l0",
        "action_net.weight": "action_head.weight",
        "action_net.bias": "action_head.bias",
    }
    with torch.no_grad():
        for teacher_key, student_key in mapping.items():
            if teacher_key not in teacher_state or student_key not in student_state:
                skipped.append(f"{teacher_key} -> {student_key}: missing")
                continue
            if tuple(teacher_state[teacher_key].shape) != tuple(student_state[student_key].shape):
                skipped.append(
                    f"{teacher_key} -> {student_key}: shape {tuple(teacher_state[teacher_key].shape)} != {tuple(student_state[student_key].shape)}"
                )
                continue
            student_state[student_key].copy_(teacher_state[teacher_key])
            copied.append(f"{teacher_key} -> {student_key}")
    return {"copied": copied, "skipped": skipped}


def save_student_checkpoint(
    path: Path,
    student: PaperLikeStudentPolicy,
    config: dict,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": student.state_dict(),
        "student_config": config["student"],
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_student_checkpoint(
    path: Path,
    config: dict,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Box,
    device: torch.device,
) -> PaperLikeStudentPolicy:
    student = build_student_from_config(config, observation_space, action_space).to(device)
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload)
    student.load_state_dict(state)
    return student


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_teacher_for_student(config: dict, device_name: str, seed: int = 0):
    teacher_cfg = config["teacher"]
    teacher_config = load_config(resolve_path(teacher_cfg["config"]))
    checkpoint_path = resolve_path(teacher_cfg["checkpoint"])
    vecnormalize_path = resolve_path(teacher_cfg["vecnormalize"])
    env_cfg = {**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0}
    vec_env = DummyVecEnv([make_single_env(env_cfg)])
    vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    teacher_model = build_model(teacher_config, vec_env, Path(tempfile.mkdtemp(prefix="student_teacher_model_")), device_name, int(seed))
    _, params, _ = load_from_zip_file(str(checkpoint_path), device=device_name, print_system_info=False)
    if params is None:
        raise FileNotFoundError(f"Could not load teacher params from {checkpoint_path}")
    teacher_model.set_parameters(params, exact_match=True, device=device_name)
    return teacher_model, vec_env, teacher_config


def normalize_teacher_obs(vec_env: VecNormalize, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batched = {key: value[None] for key, value in obs.items()}
    normalized = vec_env.normalize_obs(batched)
    return {key: np.asarray(value[0], dtype=np.float32) for key, value in normalized.items()}


def render_student_depth(env, depth_cfg: dict) -> np.ndarray:
    return env.render_depth_image(
        width=int(depth_cfg["width"]),
        height=int(depth_cfg["height"]),
        camera=str(depth_cfg.get("camera", "front_camera")),
        near=float(depth_cfg.get("near", 0.05)),
        far=float(depth_cfg.get("far", 2.0)),
        normalize=bool(depth_cfg.get("normalize", True)),
    )
