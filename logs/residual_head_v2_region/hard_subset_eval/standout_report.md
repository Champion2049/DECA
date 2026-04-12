# Standout Report: residual_v2

Policy: abs1=0.90, cam=10.0, abs2=0.40

## Hard-Subset Leaderboard

| Bucket | N | Base Trim | Residual Trim | Gated Trim | Residual Win-Rate | Gated Win-Rate | Fallback Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 2000 | 0.106129 | 0.103639 | 0.103650 | 0.670 | 0.669 | 0.001 |
| mouth | 808 | 0.110821 | 0.108665 | 0.108665 | 0.649 | 0.649 | 0.000 |
| eyes | 309 | 0.111398 | 0.109157 | 0.109171 | 0.625 | 0.621 | 0.003 |
| pose | 1476 | 0.074041 | 0.071392 | 0.071407 | 0.672 | 0.671 | 0.001 |

## Win-Rate Summary
- overall: residual wins=67.000%, gated wins=66.900%, fallback=0.100%
- mouth: residual wins=64.851%, gated wins=64.851%, fallback=0.000%
- eyes: residual wins=62.460%, gated wins=62.136%, fallback=0.324%
- pose: residual wins=67.209%, gated wins=67.073%, fallback=0.136%
