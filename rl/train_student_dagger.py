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

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wandb

from rl.student_dataset import append_transition_datasets, load_transition_dataset, save_transition_dataset, split_train_val
from rl.student_policy import load_student_checkpoint, load_teacher_for_student, resolve_path
from rl.student_rollout import collect_dagger_rollout
from rl.train_student_bc import device_from_config, load_student_config, make_student_env_spaces, train_student_epochs


def _append_round_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_student_config(args.config)
    device = device_from_config(config, args.device)
    dagger_cfg = dict(config["dagger"])
    if args.init_checkpoint is not None:
        dagger_cfg["init_checkpoint"] = str(args.init_checkpoint)
    if args.save_dir is not None:
        dagger_cfg["save_dir"] = str(args.save_dir)
    if args.rounds is not None:
        dagger_cfg["rounds"] = int(args.rounds)
    if args.debug:
        dagger_cfg["rounds"] = 1
        dagger_cfg["rollout_steps_per_round"] = min(int(dagger_cfg.get("rollout_steps_per_round", 20000)), 1000)
        dagger_cfg["train_epochs_per_round"] = 1
        dagger_cfg["batch_size"] = min(int(dagger_cfg.get("batch_size", 64)), 8)
        config.setdefault("wandb", {})["enabled"] = False

    save_dir = resolve_path(dagger_cfg.get("save_dir", "artifacts/student_paper_like_dagger"))
    save_dir.mkdir(parents=True, exist_ok=True)
    observation_space, action_space = make_student_env_spaces(config, seed=int(config["eval"].get("seed", 12345)))
    init_checkpoint = resolve_path(dagger_cfg.get("init_checkpoint", config["bc"].get("best_checkpoint", "artifacts/student_paper_like_bc/best_student.pt")))
    student = load_student_checkpoint(init_checkpoint, config, observation_space, action_space, device)
    teacher_model, teacher_vecnormalize, teacher_config = load_teacher_for_student(config, str(device), seed=int(config["eval"].get("seed", 12345)))

    replay_arrays = None
    initial_dataset = resolve_path(config["bc"]["dataset_path"])
    if initial_dataset.exists() and bool(dagger_cfg.get("include_bc_dataset", True)):
        replay_arrays, _ = load_transition_dataset(initial_dataset)

    wandb_cfg = config.get("wandb", {})
    run = None
    if bool(wandb_cfg.get("enabled", False)):
        run = wandb.init(
            project=wandb_cfg.get("project", "quadruped-student"),
            entity=wandb_cfg.get("entity"),
            name=str(wandb_cfg.get("run_name", "student-paper-like")) + "-dagger",
            group=wandb_cfg.get("group", "student-paper-like"),
            tags=[*wandb_cfg.get("tags", []), "dagger"],
            dir=str(save_dir),
            config=config,
            mode=wandb_cfg.get("mode", "online"),
        )

    try:
        for round_idx in range(1, int(dagger_cfg.get("rounds", 5)) + 1):
            arrays = collect_dagger_rollout(
                student=student,
                teacher_model=teacher_model,
                teacher_vecnormalize=teacher_vecnormalize,
                teacher_config=teacher_config,
                student_config=config,
                num_steps=int(dagger_cfg.get("rollout_steps_per_round", 20000)),
                seed=int(config["eval"].get("seed", 12345)) + 10_000 * round_idx,
                device=device,
                terrain_type="random_slope_up_down",
            )
            replay_arrays = append_transition_datasets(
                replay_arrays,
                arrays,
                max_transitions=int(dagger_cfg.get("max_buffer_transitions", 300000)),
            )
            if bool(dagger_cfg.get("save_replay_dataset", False)):
                replay_path = save_dir / "dagger_replay_dataset.pt"
                save_transition_dataset(replay_path, replay_arrays, metadata={"round": round_idx})
            train_arrays, val_arrays = split_train_val(replay_arrays, val_fraction=float(dagger_cfg.get("val_fraction", 0.1)))
            best_loss, best_path = train_student_epochs(
                student=student,
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                config=config,
                device=device,
                epochs=int(dagger_cfg.get("train_epochs_per_round", 5)),
                batch_size=int(dagger_cfg.get("batch_size", 64)),
                bptt_len=int(dagger_cfg.get("bptt_len", 24)),
                learning_rate=float(dagger_cfg.get("learning_rate", 1e-4)),
                weight_decay=float(dagger_cfg.get("weight_decay", 0.0)),
                save_dir=save_dir,
                prefix=f"dagger_round_{round_idx}",
                start_epoch=0,
            )
            row = {
                "round": round_idx,
                "collected_transitions": int(arrays["action"].shape[0]),
                "buffer_transitions": int(replay_arrays["action"].shape[0]),
                "best_val_loss": best_loss,
                "best_path": str(best_path),
            }
            _append_round_csv(save_dir / "dagger_rounds.csv", row)
            if wandb.run is not None:
                wandb.log(
                    {
                        "dagger/round": round_idx,
                        "dagger/collected_transitions": row["collected_transitions"],
                        "dagger/buffer_transitions": row["buffer_transitions"],
                        "dagger/best_val_loss": best_loss,
                    }
                )
            print(
                f"[student-dagger] round={round_idx} collected={row['collected_transitions']} "
                f"buffer={row['buffer_transitions']} best_val_loss={best_loss:.6f}",
                flush=True,
            )
    finally:
        teacher_vecnormalize.close()
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
