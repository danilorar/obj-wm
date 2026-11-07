#!/usr/bin/env bash
# Train the official v2 world model (run "v2_d_rerun" in docs/EXPERIMENTS.md), from the repository root.
# Needs data/processed_v2 (scripts/build_processed_dataset.py --lane-tokens) and assets/anchors.npy
# (scripts/make_anchors.py). Usage: bash scripts/train_official_v2.sh [seed]   (default 42; reported seeds 42 123 2026)
set -euo pipefail
SEED="${1:-42}"
FLAGS="--model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2"
# --plan-drop 0.2 must be passed explicitly: the default (0.5) is kept for older runs.
.venv/bin/python scripts/train_model.py $FLAGS --assign predicted --epochs 60 --patience 15 --patience-start 30 \
    --batch-size 64 --lr 0.001 --seed "$SEED" --checkpoint-dir "checkpoints/v2_d_seed${SEED}"
# Validation only (test is scored only with --eval-test, for the final evaluation):
.venv/bin/python scripts/train_model.py $FLAGS --evaluate-only --checkpoint "checkpoints/v2_d_seed${SEED}/v2_seed${SEED}_best.pth"
