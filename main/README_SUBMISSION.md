# Submission Notes

This folder is the cleaned code package for my robot learning project. I kept the source files, configs, MuJoCo XML files, and final CSV result, and left out checkpoints, datasets, videos, W&B folders, and cache files.

## What Is Included

- `models/`: custom Pupper-like XML files.
- `rl/`: environment, reward, PPO teacher, student models, BC training, and fusion fine-tuning code.
- `scripts/`: data collection, PPO fine-tuning, evaluation, and rendering scripts.
- `configs/`: teacher random-slope configs and student ablation configs.
- `notes/`: short note on the Pupper-like model parameters.
- `results/`: final multi-seed comparison CSV.
- `diff_robot_project.txt`: required diff file. The project was written from scratch, so the file only states that there was no starter-code diff.

## Main Code Pointers

- Robot/environment:
  - `models/pupper_like_preview_terrain.xml`
  - `rl/env.py`
  - `rl/rewards.py`

- Teacher PPO:
  - `rl/train_teacher.py`
  - `rl/gru_policy.py`
  - `rl/paper_bptt.py`
  - `scripts/train_random_slopes_two_stage.py`
  - `scripts/iterative_fine_tune_random_slopes_no_corridor.py`
  - `scripts/iterative_fine_tune_random_slopes_shifted_no_corridor.py`

- Student BC and fusion models:
  - `rl/student_policy.py`
  - `rl/student_dataset.py`
  - `rl/student_rollout.py`
  - `rl/train_student_bc.py`
  - `rl/train_student_paper_like_finetune.py`
  - `rl/train_student_gated_bc.py`
  - `rl/train_student_bilinear_bc.py`

- Evaluation:
  - `scripts/eval_student_paper_like.py`
  - `scripts/eval_final_four_models_multiseed.py`
  - `scripts/render_random_slope_train_eval_smoke.py`
  - `scripts/render_student_depth_eval_video.py`

## Training Pipeline

The teacher is a GRU PPO policy with proprioception, command, and privileged scandots. The terrain is generated online as random up/down slope segments in `rl/env.py`. Adjacent slope pieces overlap slightly so there are no gaps.

The teacher training stages are:

1. PPO pretraining on corridor random slopes.
2. Conservative PPO fine-tuning on the same corridor task.
3. Iterative no-corridor PPO fine-tuning.
4. Iterative shifted no-corridor PPO fine-tuning.

The iterative fine-tuning scripts select checkpoints by fall rate first and average distance second, and stop after 50 evaluations without improvement.

After the teacher is trained, I collect a BC dataset using teacher rollouts. The paper-like student uses:

```text
depth -> CNN -> z_vis
[proprio, z_vis, command] -> GRU -> action
```

Then I compare three student fine-tuning variants:

1. Same-budget paper-like fine-tune.
2. Residual gated IMU-depth fusion.
3. Residual low-rank bilinear IMU-depth fusion.

The gated and bilinear models both start from the same paper-like BC checkpoint and use the same 100k BC dataset.

## Useful Commands

Teacher no-corridor fine-tune:

```bash
python scripts/iterative_fine_tune_random_slopes_no_corridor.py --mujoco-gl osmesa
```

Shifted no-corridor fine-tune:

```bash
python scripts/iterative_fine_tune_random_slopes_shifted_no_corridor.py --mujoco-gl osmesa
```

Collect BC data:

```bash
python scripts/collect_student_bc_dataset.py \
  --config configs/student_paper_like.yaml \
  --device cuda \
  --mujoco-gl osmesa
```

Train paper-like BC:

```bash
python rl/train_student_bc.py \
  --config configs/student_paper_like.yaml \
  --device cuda
```

Train residual gated fusion:

```bash
python rl/train_student_gated_bc.py \
  --config configs/student_residual_gated_imu_clean.yaml \
  --device cuda
```

Train residual bilinear fusion:

```bash
python rl/train_student_bilinear_bc.py \
  --config configs/student_residual_bilinear_imu_clean.yaml \
  --device cuda
```

Final multi-seed evaluation:

```bash
python scripts/eval_final_four_models_multiseed.py \
  --device cuda:1 \
  --mujoco-gl osmesa \
  --output artifacts/final_model_comparison/teacher_paper_gated_bilinear_6seed_50eps.csv
```

## Notes

The code expects trained checkpoints and BC datasets under the original `artifacts/` paths. They are not included here because of file size. The final CSV result is included in `results/`.

The table 1 of the report is collected from wandb logs, so there's no generated csv result for that part in the submission.