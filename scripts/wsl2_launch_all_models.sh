#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="${DECA_DIR}/logs/launcher"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate deca-wsl

SP="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SP}/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${SP}/cuda_runtime/lib:${SP}/cublas/lib:${SP}/cusparse/lib:${SP}/cusolver/lib:${LD_LIBRARY_PATH:-}"

cd "${DECA_DIR}"
mkdir -p "${LAUNCH_LOG_DIR}"

start_run() {
  local tag="$1"
  local cfg="$2"
  local out_log="${LAUNCH_LOG_DIR}/${tag}_${RUN_TS}.log"
  local out_pid="${LAUNCH_LOG_DIR}/${tag}_${RUN_TS}.pid"

  nohup python main_train.py --cfg "${cfg}" > "${out_log}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${out_pid}"
  echo "${tag}: pid=${pid}"
  echo "  cfg=${cfg}"
  echo "  log=${out_log}"
}

start_run "model_a" "configs/release_version/deca_facescape_finetune_model_a.yml"
start_run "model_b" "configs/release_version/deca_facescape_finetune_model_b.yml"
start_run "model_c" "configs/release_version/deca_facescape_finetune_model_c.yml"

echo "All three runs launched in background."
echo "Use: tail -f ${LAUNCH_LOG_DIR}/model_a_${RUN_TS}.log"
