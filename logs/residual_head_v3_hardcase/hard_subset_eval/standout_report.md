# Standout Report: residual_v3_hardcase

Policy: abs1=0.90, cam=10.0, abs2=0.40

## Hard-Subset Leaderboard

| Bucket | N | Base Trim | Residual Trim | Gated Trim | Residual Win-Rate | Gated Win-Rate | Fallback Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 2000 | 0.106129 | 0.102584 | 0.102591 | 0.760 | 0.759 | 0.001 |
| mouth | 808 | 0.110821 | 0.107518 | 0.107518 | 0.757 | 0.757 | 0.000 |
| eyes | 309 | 0.111398 | 0.107858 | 0.107867 | 0.751 | 0.748 | 0.003 |
| pose | 1476 | 0.074042 | 0.070811 | 0.070821 | 0.768 | 0.767 | 0.001 |

## Win-Rate Summary
- overall: residual wins=76.000%, gated wins=75.900%, fallback=0.100%
- mouth: residual wins=75.743%, gated wins=75.743%, fallback=0.000%
- eyes: residual wins=75.081%, gated wins=74.757%, fallback=0.324%
- pose: residual wins=76.829%, gated wins=76.694%, fallback=0.136%

## Side-by-Side Evidence
- Evidence panels saved under `evidence/mouth`, `evidence/eyes`, and `evidence/pose` with `best` and `worst` groups.
- Per-file index is in `evidence/manifest.csv`.
