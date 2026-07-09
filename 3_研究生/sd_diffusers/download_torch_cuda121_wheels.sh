#!/usr/bin/env bash
set -euo pipefail

WHEEL_DIR="${WHEEL_DIR:-/home/sxq/.cache/sd_wheels}"
mkdir -p "$WHEEL_DIR"
cd "$WHEEL_DIR"

urls=(
  "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl"
  "https://download.pytorch.org/whl/cu121/nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl"
)

for url in "${urls[@]}"; do
  echo "Downloading: $url"
  wget -nv -c --tries=20 --timeout=60 "$url"
done

echo "Wheels saved in $WHEEL_DIR"
ls -lh "$WHEEL_DIR"
