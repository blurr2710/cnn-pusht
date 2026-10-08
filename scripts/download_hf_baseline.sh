#!/usr/bin/env bash
# Download and migrate the optional 1 GB Hugging Face diffusion baseline.
# This is deliberately separate from setup_runpod.sh: CNN-only work does not
# need this large model download.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SOURCE_MODEL_DIR="$PROJECT_DIR/models/hf_baseline_source"
MIGRATED_MODEL_DIR="$PROJECT_DIR/models/hf_baseline_migrated"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Run setup first: bash scripts/setup_runpod.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"
python -m pip install "huggingface_hub"

if [ ! -f "$MIGRATED_MODEL_DIR/policy_preprocessor.json" ]; then
  if [ ! -f "$SOURCE_MODEL_DIR/model.safetensors" ]; then
    echo "Downloading lerobot/diffusion_pusht (about 1 GB)..."
    hf download lerobot/diffusion_pusht --local-dir "$SOURCE_MODEL_DIR"
  fi

  echo "Converting the model to the LeRobot 0.4.4 processor format..."
  python -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path "$SOURCE_MODEL_DIR" \
    --output-dir "$MIGRATED_MODEL_DIR"
fi

echo "Hugging Face baseline ready: $MIGRATED_MODEL_DIR"
