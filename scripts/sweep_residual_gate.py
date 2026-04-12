import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def robust_trimmed_mean(values: np.ndarray, trim_ratio: float = 0.05) -> float:
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    k = int(values.size * trim_ratio)
    if k == 0 or values.size - 2 * k <= 0:
        return float(values.mean())
    return float(values[k:-k].mean())


def stats(errors: np.ndarray, failure_threshold: float) -> dict:
    return {
        "mean": float(errors.mean()),
        "trimmed": robust_trimmed_mean(errors),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "fail_rate": float((errors > failure_threshold).mean()),
    }


def parse_grid(text: str) -> list:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Sweep residual-head visual-first thresholds")
    parser.add_argument("--csv", required=True, help="all_samples.csv from compare_residual_head.py")
    parser.add_argument("--failure_threshold", type=float, default=2.0)
    parser.add_argument("--abs1_grid", default="0.76,0.80,0.85,0.90,1.00,1.20")
    parser.add_argument("--cam_grid", default="6.5,7.0,7.5,8.0,9.0,10.0")
    parser.add_argument("--abs2_grid", default="0.40,0.45,0.50,0.55,0.60,0.70")
    parser.add_argument("--p90_margin", type=float, default=0.0)
    parser.add_argument("--trimmed_margin", type=float, default=0.0)
    parser.add_argument("--fail_margin", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "base": float(r["baseline_lmk_err"]),
                    "res": float(r["residual_lmk_err"]),
                    "cam": float(r["residual_cam_score"]),
                    "abs": float(r["residual_lmk_abs_max"]),
                }
            )

    base = np.array([r["base"] for r in rows], dtype=np.float64)
    res = np.array([r["res"] for r in rows], dtype=np.float64)
    cam = np.array([r["cam"] for r in rows], dtype=np.float64)
    abs_max = np.array([r["abs"] for r in rows], dtype=np.float64)

    base_stats = stats(base, args.failure_threshold)
    residual_stats = stats(res, args.failure_threshold)

    candidates = []
    abs1_grid = parse_grid(args.abs1_grid)
    cam_grid = parse_grid(args.cam_grid)
    abs2_grid = parse_grid(args.abs2_grid)

    for abs1, cam_thr, abs2 in itertools.product(abs1_grid, cam_grid, abs2_grid):
        fallback = (abs_max > abs1) | ((cam > cam_thr) & (abs_max > abs2))
        fused = np.where(fallback, base, res)
        fused_stats = stats(fused, args.failure_threshold)

        is_safe = (
            fused_stats["p90"] <= base_stats["p90"] + args.p90_margin
            and fused_stats["trimmed"] <= base_stats["trimmed"] + args.trimmed_margin
            and fused_stats["fail_rate"] <= base_stats["fail_rate"] + args.fail_margin
        )

        candidates.append(
            {
                "abs1": abs1,
                "cam": cam_thr,
                "abs2": abs2,
                "fallback_count": int(fallback.sum()),
                "fallback_rate": float(fallback.mean()),
                "safe": bool(is_safe),
                "mean": fused_stats["mean"],
                "trimmed": fused_stats["trimmed"],
                "median": fused_stats["median"],
                "p90": fused_stats["p90"],
                "fail_rate": fused_stats["fail_rate"],
            }
        )

    safe = [c for c in candidates if c["safe"]]
    safe_sorted = sorted(safe, key=lambda c: (c["fallback_rate"], c["trimmed"], c["p90"]))
    all_sorted = sorted(candidates, key=lambda c: (0 if c["safe"] else 1, c["fallback_rate"], c["trimmed"]))

    result = {
        "baseline": base_stats,
        "residual_only": residual_stats,
        "grid": {
            "abs1": abs1_grid,
            "cam": cam_grid,
            "abs2": abs2_grid,
        },
        "safety_margins": {
            "p90_margin": args.p90_margin,
            "trimmed_margin": args.trimmed_margin,
            "fail_margin": args.fail_margin,
        },
        "safe_count": len(safe),
        "top_safe": safe_sorted[: args.topk],
        "top_all": all_sorted[: args.topk],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Residual Gate Sweep")
    print(f"Samples: {len(rows)}")
    print(f"Safe candidates: {len(safe)} / {len(candidates)}")
    if safe_sorted:
        best = safe_sorted[0]
        print(
            "Best safe gate: "
            f"abs1={best['abs1']:.3f}, cam={best['cam']:.3f}, abs2={best['abs2']:.3f}, "
            f"fallback={best['fallback_count']} ({best['fallback_rate']:.3%}), "
            f"trimmed={best['trimmed']:.6f}, p90={best['p90']:.6f}, fail={best['fail_rate']:.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
