#!/usr/bin/env bash
set -eo pipefail

source /home/chirayu/miniconda3/etc/profile.d/conda.sh
conda activate deca-wsl

cd /mnt/c/Users/Chirayu/Documents/GitHub/2D-to-3D-image-reconstruction/DECA

SP="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# Collect all NVIDIA package include/lib directories (cusparse, cusolver, cublas, etc.).
INCLUDE_PATHS="$CONDA_PREFIX/include"
LIB_PATHS="$CONDA_PREFIX/lib"
for d in "$SP"/*; do
	if [ -d "$d/include" ]; then
		INCLUDE_PATHS="$INCLUDE_PATHS:$d/include"
	fi
	if [ -d "$d/lib" ]; then
		LIB_PATHS="$LIB_PATHS:$d/lib"
	fi
done

export CPLUS_INCLUDE_PATH="$INCLUDE_PATHS"
export CPATH="$INCLUDE_PATHS"
export LIBRARY_PATH="$LIB_PATHS"
export LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1

python -m pip uninstall -y pytorch3d || true
python -m pip install --no-build-isolation --verbose git+https://github.com/facebookresearch/pytorch3d.git

python - <<'PY'
import torch
import pytorch3d
from pytorch3d import _C
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('pytorch3d', pytorch3d.__version__)
print('has_cuda_ext', hasattr(_C, 'rasterize_meshes'))
PY
