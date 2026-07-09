#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-/home/sxq/.venvs/sd-diffusers}"
WHEEL_DIR="${WHEEL_DIR:-/home/sxq/.cache/sd_wheels}"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

bash "$(dirname "$0")/download_torch_cuda121_wheels.sh"

"$VENV/bin/python" -m pip install \
  "$WHEEL_DIR"/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl \
  "$WHEEL_DIR"/torchvision-0.20.1+cu121-cp310-cp310-linux_x86_64.whl \
  --find-links "$WHEEL_DIR" \
  --index-url https://download.pytorch.org/whl/cu121

"$VENV/bin/python" -m pip install \
  diffusers==0.31.0 \
  transformers==4.46.3 \
  accelerate==1.1.1 \
  safetensors==0.4.5 \
  huggingface_hub==0.26.5 \
  pillow \
  tqdm \
  opencv-python

"$VENV/bin/python" - <<'PY'
import torch
import diffusers

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("diffusers:", diffusers.__version__)
PY
