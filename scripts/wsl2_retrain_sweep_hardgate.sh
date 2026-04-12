#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"
CFG_PATH="${1:-configs/release_version/deca_facescape_compare_model_a_lip_recover_v12_headonly_anchor.yml}"
FINAL_MAX_SAMPLES="${2:-2000}"
SWEEP_MAX_SAMPLES="${3:-800}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate deca-wsl

SP="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SP}/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${SP}/cuda_runtime/lib:${SP}/cublas/lib:${SP}/cusparse/lib:${SP}/cusolver/lib:${LD_LIBRARY_PATH:-}"

cd "${DECA_DIR}"

OUTPUT_DIR=$(python - <<'PY' "${CFG_PATH}"
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
out = None
for raw in cfg.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if line.startswith('output_dir:'):
        out = line.split(':', 1)[1].strip().strip('"').strip("'")
        break
if not out:
    raise SystemExit(f'output_dir not found in {cfg}')
print(out)
PY
)

RUN_NAME="$(basename "${OUTPUT_DIR}")"
RUN_TAG="${RUN_NAME#facescape_}"
MODELS_GLOB="${OUTPUT_DIR}/models/*.tar"
SWEEP_DIR="./logs/ckpt_sweep_${RUN_TAG}"
BEST_REPORT_PATH="./logs/model_compare_report_${RUN_TAG}_best.txt"
BEST_DEBUG_DIR="./logs/model_compare_debug_${RUN_TAG}_best"
DECISION_PATH="./logs/${RUN_TAG}_best_hardgate_decision.txt"
BEST_PATH_FILE="./logs/${RUN_TAG}_best_checkpoint.txt"


echo "[1/5] Train"
python main_train.py --cfg "${CFG_PATH}"

echo "[2/5] Sweep checkpoints (fast ranking)"
python scripts/sweep_checkpoints.py \
  --checkpoints_glob "${MODELS_GLOB}" \
  --baseline ./data/deca_model.tar \
  --val_list ../val_list_wsl.txt \
  --max_samples "${SWEEP_MAX_SAMPLES}" \
  --device cuda \
  --out_dir "${SWEEP_DIR}"

echo "[3/5] Pick best checkpoint from sweep"
BEST_CKPT=$(python - <<'PY' "${SWEEP_DIR}/summary.csv"
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
rows = []
with summary.open('r', encoding='utf-8') as f:
    rd = csv.DictReader(f)
    for r in rd:
        try:
            rc = int(float(r.get('return_code', '1') or '1'))
            ft_trim = float(r.get('finetuned_trimmed'))
            ft_fail = float(r.get('finetuned_fail_rate'))
            ft_mean = float(r.get('finetuned_mean'))
        except Exception:
            continue
        if rc != 0:
            continue
        rows.append((ft_trim, ft_fail, ft_mean, r['checkpoint']))
if not rows:
    raise SystemExit('No valid checkpoint rows found in sweep summary')
rows.sort(key=lambda x: (x[0], x[1], x[2]))
print(rows[0][3])
PY
)

echo "Best checkpoint: ${BEST_CKPT}"
printf '%s\n' "${BEST_CKPT}" > "${BEST_PATH_FILE}"

echo "[4/5] Evaluate baseline vs best checkpoint (final)"
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
  --finetuned "${BEST_CKPT}" \
  --val_list ../val_list_wsl.txt \
  --max_samples "${FINAL_MAX_SAMPLES}" \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 40 \
  --top_k_examples 40 \
  --hybrid_metric lmk_abs_max \
  --hybrid_cam_thresholds 0.6,0.7,0.8,0.9,1.0,1.1,1.2 \
    --hybrid_selective_abs1 0.90 \
    --hybrid_selective_cam 10.0 \
    --hybrid_selective_abs2 0.40 \
  --out "${BEST_REPORT_PATH}" \
  --debug_dir "${BEST_DEBUG_DIR}"

echo "[5/5] Hard gate on trimmed mean and failure rate"
python - <<'PY' "${BEST_REPORT_PATH}" "${DECISION_PATH}" "${BEST_CKPT}"
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
best_ckpt = sys.argv[3]

base_trim = ft_trim = None
base_fail = ft_fail = None
base_mean = ft_mean = None
for raw in report_path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if line.startswith('LandmarkErrorMean'):
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 3:
            base_mean = float(p[1]); ft_mean = float(p[2])
    elif line.startswith('LandmarkErrorTrimmedMean'):
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 3:
            base_trim = float(p[1]); ft_trim = float(p[2])
    elif line.startswith('LandmarkFailureRate(>2.0)'):
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 3:
            base_fail = float(p[1]); ft_fail = float(p[2])

if None in (base_trim, ft_trim, base_fail, ft_fail, base_mean, ft_mean):
    raise SystemExit('Failed to parse required metrics from report')

hard_pass = (ft_trim < base_trim) and (ft_fail < base_fail)
ideal_mean = (ft_mean < 0.9)
impossible_strict_failrate = (base_fail <= 0.0)

lines = [
    f'BEST_CHECKPOINT={best_ckpt}',
    f'HARD_PASS={1 if hard_pass else 0}',
    f'IDEAL_MEAN_LT_0_9={1 if ideal_mean else 0}',
    f'IMPOSSIBLE_STRICT_FAILRATE_WHEN_BASELINE_ZERO={1 if impossible_strict_failrate else 0}',
    f'BaselineMean={base_mean:.6f}',
    f'FineTunedMean={ft_mean:.6f}',
    f'BaselineTrimmed={base_trim:.6f}',
    f'FineTunedTrimmed={ft_trim:.6f}',
    f'BaselineFailRate={base_fail:.6f}',
    f'FineTunedFailRate={ft_fail:.6f}',
]
out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))

if not hard_pass:
    raise SystemExit(42)
PY

echo "Hard gate passed"
echo "Decision file: ${DECISION_PATH}"
echo "Best checkpoint file: ${BEST_PATH_FILE}"
