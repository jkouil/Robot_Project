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

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wandb

from rl.env import make_env
from rl.student_dataset import RecurrentTransitionDataset, load_transition_dataset, split_train_val
from rl.student_policy import (
    build_student_from_config,
    copy_teacher_weights_to_student,
    load_teacher_for_student,
    resolve_path,
    save_student_checkpoint,
)


def load_student_config(path: Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def device_from_config(config: dict, requested: str | None = None) -> torch.device:
    name = requested or config.get("student", {}).get("device", "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("[student] requested CUDA but torch.cuda is unavailable; falling back to CPU", flush=True)
        name = "cpu"
    return torch.device(name)


def make_student_env_spaces(config: dict, seed: int = 0):
    teacher_config_path = resolve_path(config["teacher"]["config"])
    with open(teacher_config_path, "r", encoding="utf-8") as handle:
        teacher_config = yaml.safe_load(handle)
    env = make_env({**teacher_config["env"], "seed": int(seed), "reset_noise_scale": 0.0})
    try:
        observation_space = env.observation_space
        action_space = env.action_space
    finally:
        env.close()
    return observation_space, action_space


def train_student_epochs(
    *,
    student,
    train_arrays: dict,
    val_arrays: dict,
    config: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    bptt_len: int,
    learning_rate: float,
    weight_decay: float,
    save_dir: Path,
    prefix: str = "bc",
    start_epoch: int = 0,
) -> tuple[float, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = RecurrentTransitionDataset(train_arrays, bptt_len=bptt_len)
    val_dataset = RecurrentTransitionDataset(val_arrays, bptt_len=bptt_len)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    train_csv = save_dir / f"{prefix}_train_loss.csv"
    val_csv = save_dir / f"{prefix}_val_loss.csv"
    best_loss = float("inf")
    best_path = save_dir / "best_student.pt"

    def append_csv(path: Path, row: dict) -> None:
        new_file = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    for local_epoch in range(int(epochs)):
        epoch = int(start_epoch + local_epoch)
        student.train()
        train_losses = []
        for batch in train_loader:
            depth = batch["depth"].to(device)
            proprio = batch["proprio"].to(device)
            command = batch["command"].to(device)
            target = batch["action"].to(device)
            starts = batch["episode_start"].to(device)
            pred, _ = student.forward_sequence(depth, proprio, command, starts)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        student.eval()
        val_losses = []
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
        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = float(sum(val_losses) / max(len(val_losses), 1))
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        append_csv(train_csv, row)
        append_csv(val_csv, row)
        if wandb.run is not None:
            wandb.log({f"{prefix}/epoch": epoch, f"{prefix}/train_loss": train_loss, f"{prefix}/val_loss": val_loss})
        print(f"[student-{prefix}] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        save_student_checkpoint(save_dir / "latest_student.pt", student, config, extra={"epoch": epoch, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            save_student_checkpoint(best_path, student, config, extra={"epoch": epoch, "val_loss": val_loss})
    return best_loss, best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_student_config(args.config)
    device = device_from_config(config, args.device)
    bc_cfg = config["bc"]
    if args.debug:
        bc_cfg = dict(bc_cfg)
        bc_cfg["epochs"] = 1
        bc_cfg["batch_size"] = min(int(bc_cfg.get("batch_size", 64)), 8)
        config.setdefault("wandb", {})["enabled"] = False

    save_dir = resolve_path(bc_cfg.get("save_dir", "artifacts/student_paper_like_bc"))
    save_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata = load_transition_dataset(resolve_path(bc_cfg["dataset_path"]))
    train_arrays, val_arrays = split_train_val(arrays, val_fraction=float(bc_cfg.get("val_fraction", 0.1)))
    observation_space, action_space = make_student_env_spaces(config, seed=int(config["eval"].get("seed", 12345)))
    student = build_student_from_config(config, observation_space, action_space).to(device)

    teacher_model, teacher_vecnormalize, _teacher_config = load_teacher_for_student(config, str(device), seed=int(config["eval"].get("seed", 12345)))
    copy_log = copy_teacher_weights_to_student(teacher_model, student)
    with open(save_dir / "teacher_weight_copy_log.json", "w", encoding="utf-8") as handle:
        json.dump(copy_log, handle, indent=2, sort_keys=True)
    print(f"[student-bc] copied={len(copy_log['copied'])} skipped={len(copy_log['skipped'])}", flush=True)
    for item in copy_log["copied"]:
        print(f"[student-bc] copied {item}", flush=True)
    for item in copy_log["skipped"]:
        print(f"[student-bc] skipped {item}", flush=True)

    wandb_cfg = config.get("wandb", {})
    run = None
    if bool(wandb_cfg.get("enabled", False)):
        run = wandb.init(
            project=wandb_cfg.get("project", "quadruped-student"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name", "student-paper-like-bc"),
            group=wandb_cfg.get("group", "student-paper-like"),
            tags=wandb_cfg.get("tags", []),
            dir=str(save_dir),
            config=config,
            mode=wandb_cfg.get("mode", "online"),
        )

    try:
        best_loss, best_path = train_student_epochs(
            student=student,
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            config=config,
            device=device,
            epochs=int(bc_cfg.get("epochs", 20)),
            batch_size=int(bc_cfg.get("batch_size", 64)),
            bptt_len=int(bc_cfg.get("bptt_len", 24)),
            learning_rate=float(bc_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(bc_cfg.get("weight_decay", 0.0)),
            save_dir=save_dir,
            prefix="bc",
        )
        print(f"[student-bc] best_student={best_path} best_val_loss={best_loss:.6f}", flush=True)
    finally:
        teacher_vecnormalize.close()
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
