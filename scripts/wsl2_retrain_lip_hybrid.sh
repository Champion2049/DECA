#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"
CFG_PATH="${1:-configs/release_version/deca_facescape_compare_model_a_lip_recover_v7_full20k.yml}"
MAX_SAMPLES="${2:-400}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate deca-wsl

SP="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SP}/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${SP}/cuda_runtime/lib:${SP}/cublas/lib:${SP}/cusparse/lib:${SP}/cusolver/lib:${LD_LIBRARY_PATH:-}"

cd "${DECA_DIR}"

echo "[1/5] Generate train landmarks (.npy)"
python scripts/generate_landmarks_68.py \
  --list ../train_list_wsl.txt \
  --device cuda \
  --skip_existing \
  --report ./logs/landmark_generation_report_train_full.csv \
  --failed_list ./logs/landmark_generation_failed_train_full.txt

echo "[2/5] Generate val landmarks (.npy)"
python scripts/generate_landmarks_68.py \
  --list ../val_list_wsl.txt \
  --device cuda \
  --skip_existing \
  --report ./logs/landmark_generation_report_val_full.csv \
  --failed_list ./logs/landmark_generation_failed_val_full.txt

echo "[3/5] Train lip-recovery checkpoint"
python main_train.py --cfg "${CFG_PATH}"

echo "[4/5] Compare baseline vs lip-recovery + hybrid"
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_lip_recover_v7_full20k/model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples "${MAX_SAMPLES}" \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 40 \
  --top_k_examples 40 \
  --hybrid_metric lmk_abs_max \
  --hybrid_cam_thresholds 0.6,0.7,0.8,0.9,1.0,1.1,1.2 \
  --out ./logs/model_compare_report_lip_recover_v7_full20k.txt \
  --debug_dir ./logs/model_compare_debug_lip_recover_v7_full20k

  echo "[4.5/5] Decide whether to promote finetuned checkpoint"
  python - <<'PY'
  import csv
  from pathlib import Path

  report_path = Path('./logs/model_compare_report_lip_recover_v3.txt')
  decision_path = Path('./logs/lip_recover_promotion_decision.txt')

  base_trim = ft_trim = None
  base_fail = ft_fail = None

  for raw in report_path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if line.startswith('LandmarkErrorTrimmedMean'):
      parts = [p.strip() for p in line.split(',')]
      if len(parts) >= 3:
        base_trim = float(parts[1])
        ft_trim = float(parts[2])
    if line.startswith('LandmarkFailureRate(>2.0)'):
      parts = [p.strip() for p in line.split(',')]
      if len(parts) >= 3:
        base_fail = float(parts[1])
        ft_fail = float(parts[2])

  promote = (
    base_trim is not None
    and ft_trim is not None
    and base_fail is not None
    and ft_fail is not None
    and ft_fail <= base_fail
    and ft_trim < base_trim
  )

  if promote:
    text = (
      'PROMOTE_FINETUNED=1\n'
      'Decision: finetuned checkpoint is better than baseline on failure-rate and trimmed-mean.\n'
      f'BaselineTrimmed={base_trim:.6f} FineTunedTrimmed={ft_trim:.6f}\n'
      f'BaselineFail={base_fail:.6f} FineTunedFail={ft_fail:.6f}\n'
      'RecommendedHybridThreshold: 0.6\n'
    )
  else:
    text = (
      'PROMOTE_FINETUNED=0\n'
      'Decision: keep baseline as default; finetuned did not beat baseline on acceptance metrics.\n'
      f'BaselineTrimmed={base_trim if base_trim is not None else float("nan"):.6f} '
      f'FineTunedTrimmed={ft_trim if ft_trim is not None else float("nan"):.6f}\n'
      f'BaselineFail={base_fail if base_fail is not None else float("nan"):.6f} '
      f'FineTunedFail={ft_fail if ft_fail is not None else float("nan"):.6f}\n'
      'RecommendedHybridThreshold: 0.6 (strict fallback; near-baseline behavior)\n'
    )

  decision_path.write_text(text, encoding='utf-8')
  print(text, end='')
  print(f'DecisionFile: {decision_path}')
  PY

echo "[5/5] Run geometry mismatch audit"
python scripts/audit_geometry_mismatch.py \
  --baseline ./data/deca_model.tar \
  --finetuned ./logs/facescape_compare_model_a_lip_recover_v7_full20k/model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples "${MAX_SAMPLES}" \
  --device cuda \
  --top_k 40 \
  --out_csv ./logs/geometry_audit_lip_recover_v7_full20k.csv \
  --debug_dir ./logs/geometry_audit_lip_recover_v7_full20k

echo "Pipeline complete."
