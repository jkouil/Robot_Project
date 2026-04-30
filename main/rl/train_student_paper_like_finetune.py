from __future__ import annotations

import argparse
import csv
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
from rl.student_policy import build_student_from_config, load_student_checkpoint, resolve_path, save_student_checkpoint
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


def _make_optimizer(student: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    for parameter in student.parameters():
        parameter.requires_grad = True
    return torch.optim.AdamW(student.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))


def _train_one_epoch(*, student, train_loader: DataLoader, val_loader: DataLoader, optimizer, device: torch.device) -> tuple[float, float]:
    loss_fn = nn.MSELoss()
    student.train()
    train_losses: list[float] = []
    for batch in train_loader:
        pred, _ = student.forward_sequence(
            batch["depth"].to(device),
            batch["proprio"].to(device),
            batch["command"].to(device),
            batch["episode_start"].to(device),
        )
        loss = loss_fn(pred, batch["action"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    student.eval()
    val_losses: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            pred, _ = student.forward_sequence(
                batch["depth"].to(device),
                batch["proprio"].to(device),
                batch["command"].to(device),
                batch["episode_start"].to(device),
            )
            val_loss = loss_fn(pred, batch["action"].to(device))
            val_losses.append(float(val_loss.detach().cpu()))
    return _mean(train_losses), _mean(val_losses)


def _run_eval(*, student, config: dict, save_dir: Path, epoch: int, stage: int, device: torch.device, episodes: int, seed: int, tasks: list[str]) -> tuple[float, list[dict]]:
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
        }
        rows.append(row)
        _append_csv(save_dir / "eval_during_training.csv", row)
        print(
            f"[paper-ft-eval] epoch={epoch} stage={stage} task={task} "
            f"fall_rate={metrics['fall_rate']:.4f} avg_distance={metrics['avg_distance']:.4f} "
            f"avg_speed={metrics['avg_forward_velocity']:.4f} score={score:.4f}",
            flush=True,
        )
    return _mean(scores), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like_finetune_same_budget.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_student_config(args.config)
    config.setdefault("wandb", {})["enabled"] = False
    if str(config.get("student", {}).get("architecture", "paper_like")) != "paper_like":
        raise ValueError("train_student_paper_like_finetune.py requires student.architecture: paper_like")
    device = device_from_config(config, args.device)
    ft_cfg = dict(config["paper_like_finetune"])
    if args.debug:
        ft_cfg["stage1_epochs"] = 1
        ft_cfg["stage2_epochs"] = 1
        ft_cfg["batch_size"] = min(int(ft_cfg.get("batch_size", 64)), 8)
        ft_cfg["eval_episodes"] = min(int(ft_cfg.get("eval_episodes", 10)), 2)
        ft_cfg["eval_every_epochs"] = 1

    save_dir = resolve_path(ft_cfg["save_dir"])
    if args.debug:
        save_dir = save_dir / "debug"
    save_dir.mkdir(parents=True, exist_ok=True)

    arrays, _metadata = load_transition_dataset(resolve_path(ft_cfg["dataset_path"]))
    arrays = _maybe_debug_arrays(arrays, args.debug)
    train_arrays, val_arrays = split_train_val(arrays, val_fraction=float(ft_cfg.get("val_fraction", config.get("bc", {}).get("val_fraction", 0.1))))
    train_loader = _loader(train_arrays, int(ft_cfg.get("bptt_len", 24)), int(ft_cfg.get("batch_size", 64)), shuffle=True)
    val_loader = _loader(val_arrays, int(ft_cfg.get("bptt_len", 24)), int(ft_cfg.get("batch_size", 64)), shuffle=False)

    observation_space, action_space = make_student_env_spaces(config, seed=int(config["eval"].get("seed", 12345)))
    init_checkpoint = resolve_path(ft_cfg["init_checkpoint"])
    student = load_student_checkpoint(init_checkpoint, config, observation_space, action_space, device)
    print(f"[paper-ft-init] checkpoint={init_checkpoint}", flush=True)

    best_val_loss = float("inf")
    best_eval_score = float("inf")
    global_epoch = 0
    eval_enabled = bool(ft_cfg.get("eval_during_training", True))
    eval_every = max(1, int(ft_cfg.get("eval_every_epochs", 1)))
    eval_episodes = int(ft_cfg.get("eval_episodes", 10))
    eval_tasks = [str(task) for task in ft_cfg.get("eval_tasks", ["random_slope_shift", "random_slope_shift_hard_heldout"])]
    seed = int(config["eval"].get("seed", 12345))
    stages = [
        (1, int(ft_cfg.get("stage1_epochs", 5)), float(ft_cfg.get("stage1_learning_rate", 3e-4))),
        (2, int(ft_cfg.get("stage2_epochs", 10)), float(ft_cfg.get("stage2_learning_rate", 5e-5))),
    ]
    for stage, epochs, learning_rate in stages:
        optimizer = _make_optimizer(student, learning_rate, float(ft_cfg.get("weight_decay", 0.0)))
        for _ in range(int(epochs)):
            global_epoch += 1
            train_loss, val_loss = _train_one_epoch(
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
            }
            _append_csv(save_dir / "training_loss.csv", row)
            save_student_checkpoint(save_dir / "latest_student.pt", student, config, extra=row)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_student_checkpoint(save_dir / "best_by_val_loss.pt", student, config, extra=row)
            print(
                f"[paper-ft] epoch={global_epoch} stage={stage} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} lr={learning_rate:.6g}",
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
                        extra={"epoch": global_epoch, "eval_score": eval_score, "eval_rows": eval_rows, **row},
                    )
    if not (save_dir / "best_by_eval.pt").exists():
        save_student_checkpoint(save_dir / "best_by_eval.pt", student, config, extra={"epoch": global_epoch, "eval_score": None})
    print(
        f"[paper-ft] done latest={save_dir / 'latest_student.pt'} "
        f"best_by_val_loss={save_dir / 'best_by_val_loss.pt'} best_by_eval={save_dir / 'best_by_eval.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
