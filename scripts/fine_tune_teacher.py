from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


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

from rl.train_teacher import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_paper_stairs_worldpatch_higher_bptt24_finetune_conservative.yaml",
    )
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
