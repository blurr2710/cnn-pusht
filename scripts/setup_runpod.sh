#!/usr/bin/env bash
# Prepare this project for evaluation on a CUDA-enabled RunPod PyTorch Pod.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

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
  "pymunk==6.11.1"

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

echo
echo "Setup complete. The CNN checkpoints come from Git."
echo "To evaluate the optional Hugging Face diffusion baseline later, run:"
echo "  bash scripts/download_hf_baseline.sh"
