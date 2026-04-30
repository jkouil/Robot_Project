from __future__ import annotations

import argparse
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


def _prepare_pretrain_config(template: dict, output_dir: Path, total_timesteps: int) -> dict:
    config = copy.deepcopy(template)
    config.setdefault("train", {})
    config.setdefault("eval", {})
    config.setdefault("wandb", {})

    config["train"]["output_dir"] = _config_path(output_dir)
    config["train"]["total_timesteps"] = int(total_timesteps)
    config["train"].pop("finetune_model_path", None)
    config["train"].pop("finetune_vecnormalize_path", None)

    config["eval"]["eval_episodes"] = 8
    config["eval"]["best_metric_mode"] = "fall_rate_then_distance"
    config["eval"]["best_fall_rate_epsilon"] = float(config["eval"].get("best_fall_rate_epsilon", 0.05))
    config["eval"]["early_stop_min_distance_delta"] = float(config["eval"].get("early_stop_min_distance_delta", 0.02))
    config["eval"].pop("candidate_eval_additional_episodes", None)
    config["eval"]["candidate_eval_episodes"] = 8
    config["eval"].pop("early_stop_no_improvement_evals", None)
    config["eval"]["early_stop_require_no_fall"] = False
    config["eval"]["record_best_video"] = True
    config["eval"]["record_recent_video"] = True

    config["wandb"]["run_name"] = str(config["wandb"].get("run_name", "teacher-preview-gru-random-slopes")) + "-pretrain-3m"
    config["wandb"]["group"] = str(config["wandb"].get("group", "preview-gru-random-slopes")) + "-two-stage"
    tags = list(config["wandb"].get("tags", []))
    for tag in ("pretrain-3m", "fall-rate-best", "two-stage"):
        if tag not in tags:
            tags.append(tag)
    config["wandb"]["tags"] = tags
    return config


def _prepare_finetune_config(
    template: dict,
    output_dir: Path,
    model_path: Path,
    vecnormalize_path: Path,
    baseline_metrics: dict,
    eval_patience: int,
) -> dict:
    config = copy.deepcopy(template)
    config.setdefault("train", {})
    config.setdefault("eval", {})
    config.setdefault("wandb", {})

    config["train"]["output_dir"] = _config_path(output_dir)
    config["train"]["finetune_model_path"] = _config_path(model_path)
    config["train"]["finetune_vecnormalize_path"] = _config_path(vecnormalize_path)
    config["train"]["learning_rate"] = min(float(config["train"].get("learning_rate", 5e-5)), 5e-5)
    config["train"]["clip_coef"] = min(float(config["train"].get("clip_coef", 0.1)), 0.1)
    config["train"]["entropy_coef"] = max(float(config["train"].get("entropy_coef", 0.012)), 0.012)
    config["train"]["target_kl"] = min(float(config["train"].get("target_kl", 0.015)), 0.015)

    config["eval"]["eval_episodes"] = 8
    config["eval"]["best_metric_mode"] = "fall_rate_then_distance"
    config["eval"]["candidate_eval_trigger_fall_rate_below"] = 1.0
    config["eval"]["candidate_eval_additional_episodes"] = 12
    config["eval"]["candidate_eval_episodes"] = 20
    config["eval"]["early_stop_no_improvement_evals"] = int(eval_patience)
    config["eval"]["early_stop_require_no_fall"] = False
    config["eval"]["stop_on_best_improvement"] = True
    config["eval"]["initial_best_fall_rate"] = float(baseline_metrics["fall_rate"])
    config["eval"]["initial_best_avg_distance"] = float(baseline_metrics["avg_distance"])
    config["eval"]["initial_best_no_fall_distance"] = float(baseline_metrics["avg_distance"])
    config["eval"]["record_best_video"] = True
    config["eval"]["record_recent_video"] = True

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes.yaml",
    )
    parser.add_argument(
        "--finetune-config",
        type=Path,
        default=REPO_ROOT / "configs" / "teacher_walk_preview_gru_random_slopes_finetune_conservative.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "teacher_walk_preview_gru_random_slopes_two_stage",
    )
    parser.add_argument("--pretrain-timesteps", type=int, default=3_000_000)
    parser.add_argument("--eval-patience", type=int, default=50)
    parser.add_argument("--max-finetune-rounds", type=int, default=20)
    parser.add_argument("--fresh", action="store_true", help="Run pretrain again and ignore existing vN folders.")
    parser.add_argument("--skip-pretrain", action="store_true", help="Use existing pretrain best_teacher as the fine-tune seed.")
    parser.add_argument("--mujoco-gl", type=str, choices=["egl", "osmesa", "glx"], default=None)
    args = parser.parse_args()

    pretrain_template_path = args.pretrain_config if args.pretrain_config.is_absolute() else REPO_ROOT / args.pretrain_config
    finetune_template_path = args.finetune_config if args.finetune_config.is_absolute() else REPO_ROOT / args.finetune_config
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    pretrain_dir = output_root / "pretrain_3m"
    finetune_root = output_root / "iterative_finetune"
    if args.fresh and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"--fresh was requested but output root is not empty: {output_root}. "
            "Use a new --output-root for a clean long run, or omit --fresh to resume."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    pretrain_model = pretrain_dir / "best_teacher.zip"
    pretrain_vecnormalize = pretrain_dir / "best_teacher_vecnormalize.pkl"
    pretrain_metrics_path = pretrain_dir / "best_teacher_metrics.json"

    if not args.skip_pretrain and (args.fresh or not (pretrain_model.exists() and pretrain_vecnormalize.exists() and pretrain_metrics_path.exists())):
        pretrain_config = _prepare_pretrain_config(
            _load_yaml(pretrain_template_path),
            pretrain_dir,
            int(args.pretrain_timesteps),
        )
        pretrain_config_path = pretrain_dir / "config.yaml"
        _write_yaml(pretrain_config_path, pretrain_config)
        print(
            f"[two-stage] starting_pretrain timesteps={args.pretrain_timesteps} config={pretrain_config_path}",
            flush=True,
        )
        train(load_config(pretrain_config_path))
    else:
        print(f"[two-stage] using_existing_pretrain={pretrain_model}", flush=True)

    baseline_metrics = _best_metrics_from_json(pretrain_metrics_path)
    if baseline_metrics is None or not pretrain_model.exists() or not pretrain_vecnormalize.exists():
        raise FileNotFoundError(f"pretrain best checkpoint is incomplete under {pretrain_dir}")

    current_model = pretrain_model
    current_vecnormalize = pretrain_vecnormalize
    start_round = 1
    if not args.fresh:
        latest = _latest_completed_version(finetune_root)
        if latest is not None:
            latest_version, latest_model, latest_vecnormalize, latest_metrics = latest
            current_model = latest_model
            current_vecnormalize = latest_vecnormalize
            baseline_metrics = latest_metrics
            start_round = latest_version + 1
            print(
                f"[two-stage] resumed_from=v{latest_version} "
                f"fall_rate={baseline_metrics['fall_rate']:.3f} avg_distance={baseline_metrics['avg_distance']:.3f}",
                flush=True,
            )

    finetune_template = _load_yaml(finetune_template_path)
    end_round = start_round + int(args.max_finetune_rounds) - 1
    for round_idx in range(start_round, end_round + 1):
        version_name = f"v{round_idx}"
        round_dir = finetune_root / version_name
        config = _prepare_finetune_config(
            finetune_template,
            round_dir,
            current_model,
            current_vecnormalize,
            baseline_metrics,
            int(args.eval_patience),
        )
        run_name = str(config["wandb"].get("run_name", "teacher-preview-gru-random-slopes-finetune"))
        group = str(config["wandb"].get("group", "preview-gru-random-slopes-finetune"))
        config["wandb"]["run_name"] = f"{run_name}-{version_name}"
        config["wandb"]["group"] = f"{group}-two-stage"
        tags = list(config["wandb"].get("tags", []))
        for tag in ("two-stage", "iterative-finetune", version_name):
            if tag not in tags:
                tags.append(tag)
        config["wandb"]["tags"] = tags

        round_config_path = round_dir / "config.yaml"
        _write_yaml(round_config_path, config)
        print(
            f"[two-stage] starting_finetune {version_name} "
            f"baseline_fall_rate={baseline_metrics['fall_rate']:.3f} "
            f"baseline_avg_distance={baseline_metrics['avg_distance']:.3f} config={round_config_path}",
            flush=True,
        )
        train(load_config(round_config_path))

        best_model = round_dir / "best_teacher.zip"
        best_vecnormalize = round_dir / "best_teacher_vecnormalize.pkl"
        best_metrics = _best_metrics_from_json(round_dir / "best_teacher_metrics.json")
        if best_metrics is None or not best_model.exists() or not best_vecnormalize.exists():
            print(
                f"[two-stage] no_new_best_after_{args.eval_patience}_evals in {version_name}; stopping.",
                flush=True,
            )
            break

        current_model = best_model
        current_vecnormalize = best_vecnormalize
        baseline_metrics = best_metrics
        print(
            f"[two-stage] accepted {version_name} "
            f"new_best_fall_rate={baseline_metrics['fall_rate']:.3f} "
            f"new_best_avg_distance={baseline_metrics['avg_distance']:.3f} model={current_model}",
            flush=True,
        )
    else:
        print(f"[two-stage] reached max_finetune_rounds={args.max_finetune_rounds}", flush=True)


if __name__ == "__main__":
    main()
