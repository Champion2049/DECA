#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate deca-wsl

SP="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SP}/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${SP}/cuda_runtime/lib:${SP}/cublas/lib:${SP}/cusparse/lib:${SP}/cusolver/lib:${LD_LIBRARY_PATH:-}"

cd "${DECA_DIR}"

echo "[1/3] No-retune checkpoint recheck (175/200/225, max_samples=5000, guard=0.6)"
python scripts/recheck_checkpoints.py \
  --baseline ./data/deca_model.tar \
  --val_list ../val_list_wsl.txt \
  --checkpoint_dir ./logs/facescape_compare_model_a_lip_recover_v13_seed11_micro150_300_s11/models \
  --checkpoints 175,200,225 \
  --max_samples 5000 \
  --device cuda \
  --hybrid_threshold 0.6 \
  --tag seed11_micro150_300 \
  --out_csv ./logs/recheck_seed11_micro150_300_175_200_225_5000.csv

echo "[2/3] Loss-shape only trial (smooth-l1 beta=0.015; no teacher/hard-case weight changes)"
python scripts/run_multiseed_sweep.py \
  --base_cfg configs/release_version/deca_facescape_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015.yml \
  --seeds 11 \
  --final_max_samples 2000 \
  --sweep_max_samples 1000 \
  --results_csv ./logs/multiseed_v13_seed11_micro150_300_beta0015_results.csv

echo "[3/3] Guarded hybrid report for primary checkpoint 200 (threshold=0.6)"
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_lip_recover_v13_seed11_micro150_300_s11/models/00000200.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples 5000 \
  --device cuda \
  --hybrid_metric lmk_abs_max \
  --hybrid_auto_threshold 0.6 \
  --out ./logs/model_compare_report_seed11_primary200_guard060_5000.txt \
  --debug_dir ./logs/model_compare_debug_seed11_primary200_guard060_5000

echo "Done."
