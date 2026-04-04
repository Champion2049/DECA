#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"
CFG_PATH="${1:-configs/release_version/deca_facescape_finetune_wsl.yml}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate deca-wsl

cd "${DECA_DIR}"
python main_train.py --cfg "${CFG_PATH}"
