#!/usr/bin/env bash
# Prepare this project for evaluation on a CUDA-enabled RunPod PyTorch Pod.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SOURCE_MODEL_DIR="$PROJECT_DIR/models/hf_baseline_source"
MIGRATED_MODEL_DIR="$PROJECT_DIR/models/hf_baseline_migrated"

if ! command -v python >/dev/null 2>&1; then
  echo "Python is not installed on this Pod. Use a RunPod PyTorch Pod image."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "lerobot==0.4.4" \
  "gymnasium==1.4.0" \
  "gym-pusht==0.1.6" \
  "pymunk==6.11.1" \
  "huggingface_hub"

python - <<'PY'
import pymunk
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. This script must run on a GPU Pod.")
if not hasattr(pymunk.Space(), "add_collision_handler"):
    raise SystemExit("Wrong Pymunk version installed; expected Pymunk 6.x.")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
print(f"Pymunk ready: {pymunk.version}")
PY

if [ ! -f "$MIGRATED_MODEL_DIR/policy_preprocessor.json" ]; then
  if [ ! -f "$SOURCE_MODEL_DIR/model.safetensors" ]; then
    echo "Downloading the Hugging Face baseline model (about 1 GB)..."
    hf download lerobot/diffusion_pusht --local-dir "$SOURCE_MODEL_DIR"
  fi

  echo "Converting the model to the LeRobot 0.4.4 format (one-time step)..."
  python -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path "$SOURCE_MODEL_DIR" \
    --output-dir "$MIGRATED_MODEL_DIR"
fi

echo
echo "Setup complete. Run:"
echo "  bash scripts/eval_hf_baseline.sh 2"
