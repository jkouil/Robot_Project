from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import types

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.student_dataset import RecurrentTransitionDataset, load_transition_dataset, split_train_val
from rl.student_policy import (
    ResidualBilinearStudentPolicy,
    build_student_from_config,
    get_student_bilinear_stats,
    load_paper_like_weights_into_bilinear_student,
    resolve_path,
    save_student_checkpoint,
)
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces
from scripts.eval_teacher_bc_dagger_old_shift import _eval_student


TASK_TO_CONFIG_KEY = {
    "random_slope_shift": ("shift_config", 100_000),
    "random_slope_shift_hard_heldout": ("hard_heldout_config", 200_000),
}


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _loader(arrays: dict, bptt_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = RecurrentTransitionDataset(arrays, bptt_len=int(bptt_len))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


def _maybe_debug_arrays(arrays: dict[str, np.ndarray], debug: bool) -> dict[str, np.ndarray]:
    if not debug:
        return arrays
    max_transitions = min(int(arrays["action"].shape[0]), 1024)
    debug_arrays = {key: value[:max_transitions].copy() for key, value in arrays.items()}
    debug_arrays["episode_start"][0] = True
    return debug_arrays


def _set_trainable(student: ResidualBilinearStudentPolicy, stage: int) -> None:
    for parameter in student.parameters():
        parameter.requires_grad = stage != 1
    if stage == 1:
        for module in (student.imu_encoder, student.vis_bilinear_proj, student.imu_bilinear_proj, student.bilinear_out_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True
        student.beta.requires_grad = True


def _make_optimizer(student: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))


def _train_one_epoch(
    *,
    student: ResidualBilinearStudentPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    loss_fn = nn.MSELoss()
    student.train()
    train_losses: list[float] = []
    for batch in train_loader:
        depth = batch["depth"].to(device)
        proprio = batch["proprio"].to(device)
        pred, _ = student.forward_sequence(
            depth,
            proprio,
            batch["command"].to(device),
            batch["episode_start"].to(device),
        )
        loss = loss_fn(pred, batch["action"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    student.eval()
    val_losses: list[float] = []
    bilinear_stats: dict[str, float] | None = None
    with torch.no_grad():
        for batch in val_loader:
            depth = batch["depth"].to(device)
            proprio = batch["proprio"].to(device)
            pred, _ = student.forward_sequence(
                depth,
                proprio,
                batch["command"].to(device),
                batch["episode_start"].to(device),
            )
            val_loss = loss_fn(pred, batch["action"].to(device))
            val_losses.append(float(val_loss.detach().cpu()))
            if bilinear_stats is None:
                bilinear_stats = get_student_bilinear_stats(student, depth, proprio)
    return _mean(train_losses), _mean(val_losses), bilinear_stats or get_student_bilinear_stats(student)


def _run_eval(
    *,
    student: ResidualBilinearStudentPolicy,
    config: dict,
    save_dir: Path,
    epoch: int,
    stage: int,
    device: torch.device,
    episodes: int,
    seed: int,
    tasks: list[str],
) -> tuple[float, list[dict]]:
    rows = []
    scores = []
    teacher_vecnormalize_path = resolve_path(config["teacher"]["vecnormalize"])
    for task in tasks:
        if task not in TASK_TO_CONFIG_KEY:
            raise ValueError(f"unsupported eval task={task!r}; valid values are {sorted(TASK_TO_CONFIG_KEY)}")
        config_key, seed_offset = TASK_TO_CONFIG_KEY[task]
        metrics = _eval_student(
            student=student,
            student_config=config,
            teacher_config_path=resolve_path(config["eval"][config_key]),
            teacher_vecnormalize_path=teacher_vecnormalize_path,
            episodes=int(episodes),
            seed=int(seed) + int(seed_offset),
            device=device,
            video_path=None,
        )
        stats = get_student_bilinear_stats(student)
        score = float(metrics["fall_rate"] - 0.05 * metrics["avg_distance"])
        scores.append(score)
        row = {
            "epoch": int(epoch),
            "stage": int(stage),
            "task": task,
            "fall_rate": float(metrics["fall_rate"]),
            "avg_distance": float(metrics["avg_distance"]),
            "avg_forward_velocity": float(metrics["avg_forward_velocity"]),
            "avg_return": float(metrics["avg_return"]),
            "eval_score": score,
            **stats,
        }
        rows.append(row)
        _append_csv(save_dir / "eval_during_training.csv", row)
        print(
            f"[bilinear-eval] epoch={epoch} stage={stage} task={task} fall_rate={metrics['fall_rate']:.4f} "
            f"avg_distance={metrics['avg_distance']:.4f} avg_speed={metrics['avg_forward_velocity']:.4f} score={score:.4f}",
            flush=True,
        )
    return _mean(scores), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_residual_bilinear_imu_clean.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_student_config(args.config)
    config.setdefault("wandb", {})["enabled"] = False
    device = device_from_config(config, args.device)
    bilinear_cfg = dict(config["bilinear_bc"])
    if args.debug:
        bilinear_cfg["stage1_epochs"] = 1
        bilinear_cfg["stage2_epochs"] = 1
        bilinear_cfg["batch_size"] = min(int(bilinear_cfg.get("batch_size", 64)), 8)
        bilinear_cfg["eval_episodes"] = min(int(bilinear_cfg.get("eval_episodes", 10)), 2)
        bilinear_cfg["eval_every_epochs"] = 1

    save_dir = resolve_path(bilinear_cfg["save_dir"])
    if args.debug:
        save_dir = save_dir / "debug"
    save_dir.mkdir(parents=True, exist_ok=True)

    arrays, _metadata = load_transition_dataset(resolve_path(bilinear_cfg.get("dataset_path", config["bc"]["dataset_path"])))
    arrays = _maybe_debug_arrays(arrays, args.debug)
    train_arrays, val_arrays = split_train_val(arrays, val_fraction=float(bilinear_cfg.get("val_fraction", config["bc"].get("val_fraction", 0.1))))
    train_loader = _loader(train_arrays, int(bilinear_cfg.get("bptt_len", 24)), int(bilinear_cfg.get("batch_size", 64)), shuffle=True)
    val_loader = _loader(val_arrays, int(bilinear_cfg.get("bptt_len", 24)), int(bilinear_cfg.get("batch_size", 64)), shuffle=False)

    observation_space, action_space = make_student_env_spaces(config, seed=int(config["eval"].get("seed", 12345)))
    student = build_student_from_config(config, observation_space, action_space).to(device)
    if not isinstance(student, ResidualBilinearStudentPolicy):
        raise TypeError("train_student_bilinear_bc.py requires student.architecture: residual_bilinear")

    init_checkpoint = resolve_path(bilinear_cfg.get("init_checkpoint", config["student"]["init_from_paper_like"]))
    copy_log = load_paper_like_weights_into_bilinear_student(init_checkpoint, student, device=device)
    with open(save_dir / "weight_copy_log.json", "w", encoding="utf-8") as handle:
        json.dump(copy_log, handle, indent=2, sort_keys=True)
    print(f"[bilinear-init] copied={len(copy_log['copied'])} skipped={len(copy_log['skipped'])} checkpoint={init_checkpoint}", flush=True)
    for key in copy_log["copied"]:
        print(f"[bilinear-init] copied {key}", flush=True)
    for item in copy_log["skipped"][:50]:
        print(f"[bilinear-init] skipped {item}", flush=True)

    best_val_loss = float("inf")
    best_eval_score = float("inf")
    global_epoch = 0
    eval_enabled = bool(bilinear_cfg.get("eval_during_training", True))
    eval_every = max(1, int(bilinear_cfg.get("eval_every_epochs", 1)))
    eval_episodes = int(bilinear_cfg.get("eval_episodes", 10))
    eval_tasks = [str(task) for task in bilinear_cfg.get("eval_tasks", ["random_slope_shift", "random_slope_shift_hard_heldout"])]
    seed = int(config["eval"].get("seed", 12345))
    stages = [
        (1, int(bilinear_cfg.get("stage1_epochs", 5)), float(bilinear_cfg.get("stage1_learning_rate", 3e-4))),
        (2, int(bilinear_cfg.get("stage2_epochs", 10)), float(bilinear_cfg.get("stage2_learning_rate", 5e-5))),
    ]
    for stage, epochs, learning_rate in stages:
        _set_trainable(student, stage)
        optimizer = _make_optimizer(student, learning_rate, float(bilinear_cfg.get("weight_decay", 0.0)))
        for _ in range(int(epochs)):
            global_epoch += 1
            train_loss, val_loss, stats = _train_one_epoch(
                student=student,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
            )
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                **stats,
            }
            _append_csv(save_dir / "train_loss.csv", row)
            _append_csv(save_dir / "val_loss.csv", row)
            save_student_checkpoint(save_dir / "latest_student.pt", student, config, extra=row)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_student_checkpoint(save_dir / "best_by_val_loss.pt", student, config, extra=row)
            print(
                f"[bilinear-bc] epoch={global_epoch} stage={stage} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} beta={stats['beta_value']:.6f} "
                f"delta_to_vis_ratio={stats['delta_to_vis_ratio']:.6f} "
                f"h_mean={stats['bilinear_h_mean']:.4f} h_std={stats['bilinear_h_std']:.4f}",
                flush=True,
            )
            if eval_enabled and global_epoch % eval_every == 0:
                eval_score, eval_rows = _run_eval(
                    student=student,
                    config=config,
                    save_dir=save_dir,
                    epoch=global_epoch,
                    stage=stage,
                    device=device,
                    episodes=eval_episodes,
                    seed=seed,
                    tasks=eval_tasks,
                )
                if eval_score < best_eval_score:
                    best_eval_score = eval_score
                    save_student_checkpoint(
                        save_dir / "best_by_eval.pt",
                        student,
                        config,
                        extra={"epoch": global_epoch, "eval_score": eval_score, "eval_rows": eval_rows, **stats},
                    )
    if not (save_dir / "best_by_eval.pt").exists():
        save_student_checkpoint(save_dir / "best_by_eval.pt", student, config, extra={"epoch": global_epoch, "eval_score": None})
    print(
        f"[bilinear-bc] done latest={save_dir / 'latest_student.pt'} "
        f"best_by_val_loss={save_dir / 'best_by_val_loss.pt'} best_by_eval={save_dir / 'best_by_eval.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
