#!/usr/bin/env bash
# End-to-end pipeline: prior -> D -> pi_D* -> standard PPO on D -> modified
# PPO on D -> gap report. Run from the project root after `pip install -e .`.
#
# This runs the FULL loop once for one "modified" config
# (configs/ppo_fixed_d_modified.yaml). To test additional hypotheses, add
# more --ppo-config / --out pairs to the "fixed-D PPO" block and list them
# all in the final --ppo-checkpoints call.
set -euo pipefail

ENV_CFG=configs/env_maze.yaml
PRIOR_CFG=configs/prior_training.yaml
REF_CFG=configs/reference.yaml
STANDARD_CFG=configs/ppo_fixed_d_standard.yaml
MODIFIED_CFG=configs/ppo_fixed_d_modified.yaml

echo "=== [1/5] Training online PPO prior to target success rate ==="
python scripts/01_train_prior.py \
    --env-config "$ENV_CFG" \
    --prior-config "$PRIOR_CFG" \
    --out results/prior_checkpoint.pt

echo "=== [2/5] Collecting fixed dataset D from the frozen prior ==="
python scripts/02_collect_dataset.py \
    --env-config "$ENV_CFG" \
    --checkpoint results/prior_checkpoint.pt \
    --n-episodes 2000 \
    --seed 1 \
    --out results/dataset_D.pkl

echo "=== [3/5] Computing pi_D* (empirical + true-restricted) via exact VI ==="
python scripts/03_compute_pi_d_star.py \
    --env-config "$ENV_CFG" \
    --reference-config "$REF_CFG" \
    --dataset results/dataset_D.pkl \
    --out-empirical results/pi_d_star_empirical.pkl \
    --out-true-restricted results/pi_d_star_true_restricted.pkl

echo "=== [4/5] Training standard PPO and modified PPO on D (identical D) ==="
python scripts/04_train_fixed_d_ppo.py \
    --dataset results/dataset_D.pkl \
    --ppo-config "$STANDARD_CFG" \
    --out results/ppo_standard_on_D.pt \
    --history-out results/ppo_standard_on_D_history.csv

python scripts/04_train_fixed_d_ppo.py \
    --dataset results/dataset_D.pkl \
    --ppo-config "$MODIFIED_CFG" \
    --out results/ppo_modified_on_D.pt \
    --history-out results/ppo_modified_on_D_history.csv

echo "=== [5/5] Evaluating everything under the same live-rollout protocol ==="
python scripts/05_evaluate_all.py \
    --env-config "$ENV_CFG" \
    --reference-config "$REF_CFG" \
    --prior-checkpoint results/prior_checkpoint.pt \
    --pi-d-star-empirical results/pi_d_star_empirical.pkl \
    --pi-d-star-true-restricted results/pi_d_star_true_restricted.pkl \
    --ppo-checkpoints standard=results/ppo_standard_on_D.pt modified=results/ppo_modified_on_D.pt \
    --n-episodes 500 \
    --eval-seed 999 \
    --out results/gap_report.csv

echo "=== Done. See results/gap_report.csv ==="
