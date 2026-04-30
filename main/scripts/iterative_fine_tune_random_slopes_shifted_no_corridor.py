from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ARGS = [
    "--template-config",
    str(REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes_shifted_no_corridor.yaml"),
    "--output-root",
    str(REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_shifted_no_corridor_iterative_finetune"),
    "--start-model",
    str(REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_no_corridor_iterative_finetune" / "v12" / "final_teacher.zip"),
    "--start-vecnormalize",
    str(REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_no_corridor_iterative_finetune" / "v12" / "final_teacher_vecnormalize.pkl"),
    "--eval-patience",
    "50",
    "--max-rounds",
    "20",
    "--initial-best-fall-rate",
    "1.0",
    "--initial-best-avg-distance",
    "1000000000.0",
]


def main() -> None:
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    from scripts.iterative_fine_tune_random_slopes import main as iterative_main

    iterative_main()


if __name__ == "__main__":
    main()
