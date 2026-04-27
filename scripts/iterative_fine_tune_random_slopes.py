from __future__ import annotations

import argparse
import csv
import copy
import json
import os
from pathlib import Path
import re
import sys

import yaml


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


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _config_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _best_row_from_eval_csv(path: Path, baseline_fall_rate: float, baseline_avg_distance: float, fall_eps: float, distance_delta: float) -> dict | None:
    if not path.exists():
        return None
    best_row = None
    best_fall_rate = baseline_fall_rate
    best_avg_distance = baseline_avg_distance
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fall_rate = float(row["fall_rate"])
            avg_distance = float(row["avg_distance"])
            improved = False
            if fall_rate < best_fall_rate - fall_eps:
                improved = True
            elif abs(fall_rate - best_fall_rate) <= fall_eps and avg_distance > best_avg_distance + distance_delta:
                improved = True
            if improved:
                best_row = row
                best_fall_rate = fall_rate
                best_avg_distance = avg_distance
    return best_row


def _best_metrics_from_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _latest_completed_version(output_root: Path) -> tuple[int, Path, Path, dict] | None:
    if not output_root.exists():
        return None
    best = None
    pattern = re.compile(r"^v(\d+)$")
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        version = int(match.group(1))
        model_path = child / "best_teacher.zip"
        vecnormalize_path = child / "best_teacher_vecnormalize.pkl"
        metrics = _best_metrics_from_json(child / "best_teacher_metrics.json")
        if not model_path.exists() or not vecnormalize_path.exists() or metrics is None:
            continue
        if best is None or version > best[0]:
            best = (version, model_path, vecnormalize_path, metrics)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes_finetune_conservative.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_iterative_finetune",
    )
    parser.add_argument(
        "--start-model",
        type=Path,
        default=REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes" / "best_teacher.zip",
    )
    parser.add_argument(
        "--start-vecnormalize",
        type=Path,
        default=REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes" / "best_teacher_vecnormalize.pkl",
    )
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-patience", type=int, default=50)
    parser.add_argument("--initial-best-fall-rate", type=float, default=1.0)
    parser.add_argument("--initial-best-avg-distance", type=float, default=1.0e9)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing vN folders and start again from --start-model.")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    template_path = args.template_config if args.template_config.is_absolute() else REPO_ROOT / args.template_config
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    current_model = args.start_model if args.start_model.is_absolute() else REPO_ROOT / args.start_model
    current_vecnormalize = args.start_vecnormalize if args.start_vecnormalize.is_absolute() else REPO_ROOT / args.start_vecnormalize
    if not current_model.exists():
        raise FileNotFoundError(f"start model not found: {current_model}")
    if not current_vecnormalize.exists():
        raise FileNotFoundError(f"start vecnormalize not found: {current_vecnormalize}")
    if args.fresh and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"--fresh was requested but output root is not empty: {output_root}. "
            "Use a new --output-root for a clean run, or omit --fresh to resume."
        )

    template = _load_yaml(template_path)
    eval_template = template.setdefault("eval", {})
    fall_eps = float(eval_template.get("best_fall_rate_epsilon", 0.05))
    distance_delta = float(eval_template.get("early_stop_min_distance_delta", 0.02))
    best_fall_rate = float(args.initial_best_fall_rate)
    best_avg_distance = float(args.initial_best_avg_distance)

    output_root.mkdir(parents=True, exist_ok=True)
    start_round = 1
    if not args.fresh:
        latest = _latest_completed_version(output_root)
        if latest is not None:
            latest_version, latest_model, latest_vecnormalize, latest_metrics = latest
            current_model = latest_model
            current_vecnormalize = latest_vecnormalize
            best_fall_rate = float(latest_metrics["fall_rate"])
            best_avg_distance = float(latest_metrics["avg_distance"])
            start_round = latest_version + 1
            print(
                f"[iterative-ft] resumed_from=v{latest_version} model={current_model} "
                f"baseline_fall_rate={best_fall_rate:.3f} baseline_avg_distance={best_avg_distance:.3f}",
                flush=True,
            )
    print(
        f"[iterative-ft] output_root={output_root} start_model={current_model} "
        f"baseline_fall_rate={best_fall_rate:.3f} baseline_avg_distance={best_avg_distance:.3f}",
        flush=True,
    )

    end_round = start_round + int(args.max_rounds) - 1
    for round_idx in range(start_round, end_round + 1):
        version_name = f"v{round_idx}"
        round_dir = output_root / version_name
        config = copy.deepcopy(template)
        config.setdefault("train", {})
        config.setdefault("eval", {})
        config.setdefault("wandb", {})

        config["train"]["output_dir"] = _config_path(round_dir)
        config["train"]["finetune_model_path"] = _config_path(current_model)
        config["train"]["finetune_vecnormalize_path"] = _config_path(current_vecnormalize)

        config["eval"]["best_metric_mode"] = "fall_rate_then_distance"
        config["eval"]["early_stop_no_improvement_evals"] = int(args.eval_patience)
        config["eval"]["stop_on_best_improvement"] = True
        config["eval"]["initial_best_fall_rate"] = float(best_fall_rate)
        config["eval"]["initial_best_avg_distance"] = float(best_avg_distance)
        config["eval"]["initial_best_no_fall_distance"] = float(best_avg_distance)

        run_name = str(config["wandb"].get("run_name", "teacher-preview-gru-random-slopes-iterative"))
        group = str(config["wandb"].get("group", "preview-gru-random-slopes-iterative"))
        config["wandb"]["run_name"] = f"{run_name}-{version_name}"
        config["wandb"]["group"] = group
        tags = list(config["wandb"].get("tags", []))
        if version_name not in tags:
            tags.append(version_name)
        if "iterative-finetune" not in tags:
            tags.append("iterative-finetune")
        config["wandb"]["tags"] = tags

        round_config_path = round_dir / "config.yaml"
        _write_yaml(round_config_path, config)
        print(
            f"[iterative-ft] starting {version_name} baseline_fall_rate={best_fall_rate:.3f} "
            f"baseline_avg_distance={best_avg_distance:.3f} config={round_config_path}",
            flush=True,
        )

        train(load_config(round_config_path))

        best_model = round_dir / "best_teacher.zip"
        best_vecnormalize = round_dir / "best_teacher_vecnormalize.pkl"
        best_metrics = _best_metrics_from_json(round_dir / "best_teacher_metrics.json")
        if best_metrics is None or not best_model.exists() or not best_vecnormalize.exists():
            print(
                f"[iterative-ft] no new best after {args.eval_patience} evals in {version_name}; stopping.",
                flush=True,
            )
            break

        best_fall_rate = float(best_metrics["fall_rate"])
        best_avg_distance = float(best_metrics["avg_distance"])
        current_model = best_model
        current_vecnormalize = best_vecnormalize
        print(
            f"[iterative-ft] accepted {version_name} new_best_fall_rate={best_fall_rate:.3f} "
            f"new_best_avg_distance={best_avg_distance:.3f} model={current_model}",
            flush=True,
        )
    else:
        print(f"[iterative-ft] reached max_rounds={args.max_rounds}", flush=True)


if __name__ == "__main__":
    main()
