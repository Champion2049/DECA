#!/usr/bin/env bash
set -euo pipefail

# Run this script inside WSL Ubuntu.
if ! grep -qi microsoft /proc/version; then
  echo "This script is intended for WSL2 Ubuntu." >&2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"

sudo apt-get update
sudo apt-get install -y build-essential cmake git curl wget unzip pkg-config libgl1 libglib2.0-0

if [[ ! -d "${HOME}/miniconda3" ]]; then
  echo "Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

ENV_NAME="deca-wsl"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

# Install a Blackwell-capable PyTorch build (RTX 50-series / sm_120).
pip install --upgrade pip setuptools wheel
pip uninstall -y torch torchvision torchaudio || true
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
pip install fvcore iopath ninja

# Build PyTorch3D against the installed torch when matching prebuilt wheels are unavailable.
export TORCH_CUDA_ARCH_LIST="12.0"
# Keep compile parallelism low; otherwise WSL can kill the build process due to RAM/swap pressure.
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# DECA runtime dependencies.
pip install yacs==0.1.8 kornia torchfile face-alignment scipy scikit-image opencv-python pyyaml loguru tensorboard tensorboardX gdown
pip install --no-build-isolation chumpy==0.70

# Build WSL-native train/val list files (Linux paths), not Windows C:\ paths.
export REPO_ROOT
python - <<'PY'
from pathlib import Path
import os
import random

repo = Path(os.environ["REPO_ROOT"])
img_dir = repo / "facescape_224"
if not img_dir.exists():
    raise SystemExit(f"Missing folder: {img_dir}")

images = sorted([p.resolve() for p in img_dir.glob("*.jpg")])
if not images:
    raise SystemExit("No .jpg images found under facescape_224")

random.seed(42)
random.shuffle(images)
cut = int(len(images) * 0.9)
train = images[:cut]
val = images[cut:]

(repo / "train_list_wsl.txt").write_text("\n".join(str(p) for p in train) + "\n", encoding="utf-8")
(repo / "val_list_wsl.txt").write_text("\n".join(str(p) for p in val) + "\n", encoding="utf-8")

print(f"Wrote {len(train)} train and {len(val)} val image paths")
PY

echo "Environment setup complete."
echo "Activate env with: conda activate ${ENV_NAME}"
echo "Run training from ${DECA_DIR} with:"
echo "python main_train.py --cfg configs/release_version/deca_facescape_finetune_wsl.yml"
