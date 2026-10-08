#!/usr/bin/env bash
# Run the official Hugging Face diffusion baseline on the CUDA Pod.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISODES="${1:-2}"
MODEL_DIR="$PROJECT_DIR/models/hf_baseline_migrated"

if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/eval_hf_baseline.sh [positive-number-of-episodes]"
  exit 1
fi

if [ ! -x "$PROJECT_DIR/.venv/bin/python" ]; then
  echo "Setup has not run yet. First run: bash scripts/setup_runpod.sh"
  exit 1
fi

if [ ! -f "$MODEL_DIR/policy_preprocessor.json" ]; then
  echo "Baseline model is not ready. First run: bash scripts/setup_runpod.sh"
  exit 1
fi

cd "$PROJECT_DIR"
time "$PROJECT_DIR/.venv/bin/python" evals/eval_hf_baseline.py \
  --episodes "$EPISODES" \
  --device cuda \
  --model-dir "$MODEL_DIR"
