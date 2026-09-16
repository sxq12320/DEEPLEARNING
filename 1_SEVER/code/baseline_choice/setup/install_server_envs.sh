#!/usr/bin/env bash
# Create the per-family conda environments for the citrus baseline suite on the
# Linux server. run_comparison_batch.py drives them via --python-<family>.
#
# Usage:
#   bash setup/install_server_envs.sh            # create every missing env
#   bash setup/install_server_envs.sh mmdet      # create only the mmdet env
#
# Override defaults through the environment, e.g.:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash setup/install_server_envs.sh mmdet
#
# Notes:
# - torch must match the server driver; the default cu124 wheels run on any
#   driver new enough for CUDA 12.4. Adjust TORCH_INDEX if needed.
# - The mmdet env pins torch 2.4.1 because OpenMMLab ships prebuilt mmcv wheels
#   only up to torch 2.4.x. A newer torch means compiling mmcv from source.
set -uo pipefail

SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$(cd "${SUITE_ROOT}/.." && pwd)"

# ----- adjustable defaults ---------------------------------------------------
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
ULTRALYTICS_ROOT="${ULTRALYTICS_ROOT:-${CODE_ROOT}/ultralytics-main-new}"
MMDET_ROOT="${MMDET_ROOT:-${CODE_ROOT}/mmdetection}"
ENV_YOLO="${ENV_YOLO:-citrus-bl-yolo}"
ENV_MMDET="${ENV_MMDET:-citrus-bl-mmdet}"
ENV_TORCHVISION="${ENV_TORCHVISION:-citrus-bl-mrcnn}"
ENV_RFDETR="${ENV_RFDETR:-citrus-bl-rfdetr}"
ENV_UNET="${ENV_UNET:-citrus-bl-unet}"

have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }
mk() { # mk <env> [extra conda args...]
  local env="$1"; shift
  if have_env "${env}"; then echo "== ${env} exists, reuse"; else conda create -n "${env}" python=3.10 -y "$@"; fi
}
pin() { conda run -n "$1" pip install "$@"; }

install_yolo() {
  echo "===== ${ENV_YOLO}: ultralytics fork + YOLO baselines ====="
  mk "${ENV_YOLO}"
  pin "${ENV_YOLO}" torch torchvision --index-url "${TORCH_INDEX}"
  if [ -d "${ULTRALYTICS_ROOT}" ]; then
    conda run -n "${ENV_YOLO}" pip install -e "${ULTRALYTICS_ROOT}"
  else
    echo "!! ${ULTRALYTICS_ROOT} missing; install the fork manually or pip install ultralytics>=8.4"
  fi
  pin "${ENV_YOLO}" -r "${SUITE_ROOT}/requirements-yolo.txt"
}

install_mmdet() {
  echo "===== ${ENV_MMDET}: MMDetection (RTMDet-Ins / SOLOv2 / Mask R-CNN) ====="
  mk "${ENV_MMDET}"
  # torch 2.4.1: newest line with prebuilt mmcv wheels (see header note).
  pin "${ENV_MMDET}" "torch==2.4.1" "torchvision==0.19.1" --index-url "${TORCH_INDEX}"
  pin "${ENV_MMDET}" -U openmim
  conda run -n "${ENV_MMDET}" mim install "mmcv==2.2.0" || \
    pin "${ENV_MMDET}" "mmcv==2.2.0" -f "https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html"
  conda run -n "${ENV_MMDET}" mim install "mmdet==3.3.0" || pin "${ENV_MMDET}" "mmdet==3.3.0"
  pin "${ENV_MMDET}" -r "${SUITE_ROOT}/requirements-mmdet.txt"
  if [ ! -d "${MMDET_ROOT}/configs" ]; then
    git clone -b v3.3.0 --depth 1 https://github.com/open-mmlab/mmdetection.git "${MMDET_ROOT}"
  fi
  echo "== official configs at ${MMDET_ROOT}; fetch COCO weights with:"
  echo "   conda run -n ${ENV_MMDET} python ${SUITE_ROOT}/setup/fetch_mmdet_checkpoints.py --mmdet-root ${MMDET_ROOT}"
}

install_torchvision() {
  echo "===== ${ENV_TORCHVISION}: Torchvision Mask R-CNN R50-FPN ====="
  mk "${ENV_TORCHVISION}"
  pin "${ENV_TORCHVISION}" torch torchvision --index-url "${TORCH_INDEX}"
  pin "${ENV_TORCHVISION}" -r "${SUITE_ROOT}/requirements-torchvision-maskrcnn.txt"
}

install_rfdetr() {
  echo "===== ${ENV_RFDETR}: RF-DETR Seg ====="
  mk "${ENV_RFDETR}"
  pin "${ENV_RFDETR}" torch torchvision --index-url "${TORCH_INDEX}"
  pin "${ENV_RFDETR}" -r "${SUITE_ROOT}/requirements-rfdetr.txt"
}

install_unet() {
  echo "===== ${ENV_UNET}: U-Net / DeepLabV3+ / SegFormer + watershed ====="
  mk "${ENV_UNET}"
  pin "${ENV_UNET}" torch torchvision --index-url "${TORCH_INDEX}"
  pin "${ENV_UNET}" -r "${SUITE_ROOT}/requirements-unet.txt"
}

selection=("$@")
[ ${#selection[@]} -eq 0 ] && selection=(yolo mmdet torchvision rfdetr unet)
for family in "${selection[@]}"; do
  case "${family}" in
    yolo) install_yolo ;;
    mmdet) install_mmdet ;;
    torchvision|maskrcnn) install_torchvision ;;
    rfdetr) install_rfdetr ;;
    unet) install_unet ;;
    *) echo "unknown family: ${family}"; exit 1 ;;
  esac
done

cat <<EOF

Done. Point the batch runner at the new interpreters, e.g.:
  python run_comparison_batch.py --suite smoke --device 0 \\
    --python-yolo        "$(conda run -n ${ENV_YOLO}        python -c 'import sys;print(sys.executable)' 2>/dev/null || echo <${ENV_YOLO}-python>)" \\
    --python-mmdet       "$(conda run -n ${ENV_MMDET}       python -c 'import sys;print(sys.executable)' 2>/dev/null || echo <${ENV_MMDET}-python>)" \\
    --python-torchvision "$(conda run -n ${ENV_TORCHVISION} python -c 'import sys;print(sys.executable)' 2>/dev/null || echo <${ENV_TORCHVISION}-python>)" \\
    --python-rfdetr      "$(conda run -n ${ENV_RFDETR}      python -c 'import sys;print(sys.executable)' 2>/dev/null || echo <${ENV_RFDETR}-python>)" \\
    --python-unet        "$(conda run -n ${ENV_UNET}        python -c 'import sys;print(sys.executable)' 2>/dev/null || echo <${ENV_UNET}-python>)"
Or export CITRUS_BL_PYTHON_<FAMILY> once in ~/.bashrc.
EOF
