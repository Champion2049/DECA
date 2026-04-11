#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DECA_DIR="${REPO_ROOT}/DECA"
CFG_PATH="${1:-configs/release_version/deca_facescape_compare_model_a_lip_recover_v10_noise_reject.yml}"
MAX_SAMPLES="${2:-400}"

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
MODEL_PATH="${OUTPUT_DIR}/model.tar"
REPORT_PATH="./logs/model_compare_report_${RUN_TAG}.txt"
DEBUG_DIR="./logs/model_compare_debug_${RUN_TAG}"
DECISION_PATH="./logs/${RUN_TAG}_hardgate_decision.txt"

echo "[1/4] Train with noisy-label rejection"
python main_train.py --cfg "${CFG_PATH}"

echo "[2/4] Evaluate baseline vs finetuned"
python scripts/compare_models.py \
  --baseline ./data/deca_model.tar \
    --finetuned "${MODEL_PATH}" \
  --val_list ../val_list_wsl.txt \
  --max_samples "${MAX_SAMPLES}" \
  --device cuda \
  --failure_threshold 2.0 \
  --top_k_failures 40 \
  --top_k_examples 40 \
  --hybrid_metric lmk_abs_max \
  --hybrid_cam_thresholds 0.6,0.7,0.8,0.9,1.0,1.1,1.2 \
    --out "${REPORT_PATH}" \
    --debug_dir "${DEBUG_DIR}"

echo "[3/4] Hard gate on trimmed mean and failure rate"
python - <<'PY' "${REPORT_PATH}" "${DECISION_PATH}"
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

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

echo "[4/4] Hard gate passed"
echo "Decision file: ${DECISION_PATH}"
