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

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.student_dataset import save_transition_dataset
from rl.student_policy import load_teacher_for_student, resolve_path
from rl.student_rollout import collect_teacher_bc_rollout


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _device_name(requested: str) -> str:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[student-bc-collect] requested CUDA but torch.cuda is unavailable; falling back to CPU", flush=True)
        return "cpu"
    return str(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "student_paper_like.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    config = _load_config(resolve_path(args.config))
    if args.device is not None:
        config.setdefault("student", {})["device"] = args.device
    device_name = _device_name(config["student"].get("device", "cpu"))
    num_steps = int(config["bc"].get("num_steps", 100000))
    if args.debug:
        num_steps = min(num_steps, 1000)

    teacher_model, teacher_vecnormalize, teacher_config = load_teacher_for_student(config, device_name, seed=int(config["eval"].get("seed", 12345)))
    arrays = collect_teacher_bc_rollout(
        teacher_model=teacher_model,
        teacher_vecnormalize=teacher_vecnormalize,
        teacher_config=teacher_config,
        student_config=config,
        num_steps=num_steps,
        seed=int(config["eval"].get("seed", 12345)),
        terrain_type="random_slope_up_down",
    )
    dataset_path = resolve_path(config["bc"]["dataset_path"])
    save_transition_dataset(
        dataset_path,
        arrays,
        metadata={
            "num_steps": int(num_steps),
            "teacher_checkpoint": str(resolve_path(config["teacher"]["checkpoint"])),
            "teacher_vecnormalize": str(resolve_path(config["teacher"]["vecnormalize"])),
        },
    )
    print(f"[student-bc-collect] saved_dataset={dataset_path} transitions={arrays['action'].shape[0]}", flush=True)
    teacher_vecnormalize.close()


if __name__ == "__main__":
    main()
