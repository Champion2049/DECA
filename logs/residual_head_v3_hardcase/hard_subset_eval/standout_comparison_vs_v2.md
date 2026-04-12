# Standout Comparison: residual_v3_hardcase vs residual_v2

Policy: abs1=0.90, cam=10.0, abs2=0.40

| Bucket | N | V2 Gated Trim | V3 Gated Trim | Delta (V3-V2) | V2 Gated Win-Rate | V3 Gated Win-Rate | Delta Win-Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 2000 | 0.103650 | 0.102591 | -0.001059 | 0.669 | 0.759 | +0.090 |
| mouth | 808 | 0.108665 | 0.107518 | -0.001147 | 0.649 | 0.757 | +0.109 |
| eyes | 309 | 0.109171 | 0.107867 | -0.001304 | 0.621 | 0.748 | +0.126 |
| pose | 1476 | 0.071407 | 0.070821 | -0.000585 | 0.671 | 0.767 | +0.096 |

## Key Takeaways

- overall: trimmed improved by 0.001059 and gated win-rate rose by 9.0 points.
- mouth: trimmed improved by 0.001147 and gated win-rate rose by 10.9 points.
- eyes: trimmed improved by 0.001304 and gated win-rate rose by 12.6 points.
- pose: trimmed improved by 0.000585 and gated win-rate rose by 9.6 points.

## Evidence

- Side-by-side panels: `logs/residual_head_v3_hardcase/hard_subset_eval/evidence/`
- Panel index: `logs/residual_head_v3_hardcase/hard_subset_eval/evidence/manifest.csv`
