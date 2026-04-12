# Deployment Summary: Primary Checkpoint 200 with Selective Guard

## Model Selection (Final)
- Primary checkpoint: `./logs/facescape_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11/models/00000200.tar`
- Fallback checkpoint: `./data/deca_model.tar`
- Guard metric: `finetuned_lmk_abs_max`
- Global guard threshold: `0.6`
- Default selective policy:
  - `fallback if finetuned_lmk_abs_max > 0.76`
  - `OR if finetuned_cam_score > 6.5 AND finetuned_lmk_abs_max > 0.40`

## Visual Priority Mode
- Deployment priority is visual reliability over metric minimization.
- Policy is intentionally more conservative to avoid visually wrong fine-tuned outputs.

## Confirmation Run
- Report: `logs/model_compare_report_primary200_guard5000.txt`
- Debug dir: `logs/model_compare_debug_primary200_guard5000`
- Requested max_samples: `5000`
- Evaluated samples: `2149` (dataset-limited)

## Metrics (Baseline vs Primary)
- LandmarkErrorMean: `0.135634` vs `0.530431`
- LandmarkErrorTrimmedMean: `0.105343` vs `0.130465`
- LandmarkErrorMedian: `0.062222` vs `0.072214`
- LandmarkErrorP90: `0.305798` vs `0.433403`
- LandmarkFailureRate(>2.0): `0.000000` vs `0.018148`

## Real .npy Only Slice
- RealNpyLandmarkErrorMean: `0.132961` vs `0.531043`
- RealNpyLandmarkErrorTrimmedMean: `0.102343` vs `0.128081`
- RealNpyLandmarkErrorMedian: `0.061822` vs `0.071573`
- RealNpyLandmarkErrorP90: `0.290888` vs `0.407796`

## Rendered Artifact Check
- Worst failures rendered images: `20`
- Sanity examples rendered images: `20`
- Hybrid auto-failures rendered images: `20`
- Paths:
  - `logs/model_compare_debug_primary200_guard5000/worst_failures`
  - `logs/model_compare_debug_primary200_guard5000/sanity_examples`
  - `logs/model_compare_debug_primary200_guard5000/hybrid_auto_failures`

## Metric Audit (Independent Recompute)
- Source CSV audited: `logs/model_compare_debug_primary200_guard5000/all_samples.csv`
- Independent recomputation matched report values (within normal float/percentile rounding):
  - Baseline mean/trimmed/fail: `0.135634 / 0.105343 / 0.000000`
  - Fine-tuned mean/trimmed/fail: `0.530431 / 0.130465 / 0.018148`
- Conclusion: no calculation bug found in mean, trimmed mean, or failure-rate computation.

## Additional Note
- `scripts/compare_models.py` now reports image write success/failure counts (`WorstFailureRendersWritten`, `HybridAutoFailureRendersWritten`, `SanityExamplesRendersWritten`) to make missing-output diagnosis explicit in future runs.

## Selective Guard Lock (2026-04-12)
- Official integrated report:
  - `logs/model_compare_report_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11_best_visualfirst.txt`
- Selective metrics on checkpoint 200 (2000 samples):
  - `fallback=1955`
  - `lmk_mean=0.136087`
  - `lmk_trimmed=0.106023`
  - `fail_rate=0.000000`
- Same-run global gate (`abs>0.6`) reference:
  - `fallback=1959`
  - `lmk_mean=0.136494`
  - `lmk_trimmed=0.106335`
  - `fail_rate=0.000000`
- Delta (selective - global):
  - `mean=-0.000407`
  - `trimmed=-0.000313`
  - `fail_rate=0.000000`
  - `fallback_count=-4`

## Anchor Recheck Confirmation (175/225)
- CSV: `logs/v13_seed11_micro_recheck2000_with_selective.csv`
- Checkpoint 175 selective:
  - `mean=0.135967`, `trimmed=0.105841`, `fail=0.000000`
- Checkpoint 225 selective:
  - `mean=0.136486`, `trimmed=0.106544`, `fail=0.000000`
- Decision: checkpoint `200` remains best under selective policy and is locked for deployment.
