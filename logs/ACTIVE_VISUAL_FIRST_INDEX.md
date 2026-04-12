# Active Visual-First Artifacts

This workspace is now operating in visual-priority mode.

## Primary Model
- Checkpoint dir: `logs/facescape_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11/models`
- Locked checkpoint: `00000200.tar`

## Main Reports
- Primary visual-first report:
  - `logs/model_compare_report_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11_best_visualfirst.txt`
- Default-path smoke validation:
  - `logs/model_compare_report_recheck_defaults_visualfirst_smoke_00000200_300.txt`
- Latest full rerun (2000 samples):
  - `logs/model_compare_report_recheck_defaults_visualfirst_2000_00000200_2000.txt`
  - `logs/recheck_defaults_visualfirst_2000.csv`

## Visual Outputs (what to inspect)
- Worst failures:
  - `logs/model_compare_debug_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11_best_visualfirst/worst_failures`
- Hybrid auto-failures:
  - `logs/model_compare_debug_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11_best_visualfirst/hybrid_auto_failures`
- Sanity examples:
  - `logs/model_compare_debug_compare_model_a_lip_recover_v13_seed11_micro150_300_beta0015_s11_best_visualfirst/sanity_examples`
- Latest rerun visual outputs:
  - `logs/model_compare_debug_recheck_defaults_visualfirst_2000_00000200_2000/worst_failures`
  - `logs/model_compare_debug_recheck_defaults_visualfirst_2000_00000200_2000/hybrid_auto_failures`
  - `logs/model_compare_debug_recheck_defaults_visualfirst_2000_00000200_2000/sanity_examples`

## Current Deployment Decision
- Decision file:
  - `logs/lip_recover_promotion_decision.txt`
- Summary file:
  - `logs/deployment_primary200_guard5000_summary.md`

## Modern Technique Experiments
- Visual mismatch diagnosis:
  - `logs/visual_mismatch_beta0015_selective/visual_mismatch_diagnosis.md`
- Learned visual router experiment:
  - `logs/visual_router_beta0015/visual_router_experiment.md`

## Cleanup
- Deleted artifacts manifest:
  - `logs/cleanup_deleted_manifest_2026-04-12.txt`
