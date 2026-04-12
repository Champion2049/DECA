import argparse
import csv
from pathlib import Path

import numpy as np


def trimmed_mean(arr: np.ndarray, keep_ratio: float = 0.95) -> float:
    vals = np.sort(arr)
    trim_ratio = 1.0 - keep_ratio
    k = int(len(vals) * trim_ratio)
    if k == 0:
        return float(np.mean(vals))
    if len(vals) - 2 * k <= 0:
        return float(np.mean(vals))
    return float(np.mean(vals[k:-k]))


def compute_metrics(err: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(err)),
        "trimmed": trimmed_mean(err),
        "median": float(np.median(err)),
        "p90": float(np.quantile(err, 0.9)),
        "fail_rate": float(np.mean(err > 2.0)),
    }


def read_samples_csv(path: str) -> dict:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    def col_float(name: str) -> np.ndarray:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(name, "nan")))
            except ValueError:
                vals.append(np.nan)
        return np.array(vals, dtype=np.float64)

    return {
        "rows": rows,
        "baseline_lmk_err": col_float("baseline_lmk_err"),
        "finetuned_lmk_err": col_float("finetuned_lmk_err"),
        "finetuned_lmk_abs_max": col_float("finetuned_lmk_abs_max"),
        "finetuned_cam_score": col_float("finetuned_cam_score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective subset gating policy")
    parser.add_argument("--train_csv", required=True, help="Guard-set all_samples.csv for policy selection")
    parser.add_argument("--test_csv", required=True, help="Evaluation all_samples.csv for final metrics")
    parser.add_argument("--out_md", required=True, help="Markdown report output path")
    parser.add_argument("--out_csv", required=True, help="Per-sample selective predictions output path")
    parser.add_argument("--fixed_abs1", default=-1.0, type=float, help="Use fixed selective mode-1 threshold when >= 0")
    parser.add_argument("--fixed_cam", default=-1.0, type=float, help="Use fixed selective mode-2 camera threshold when >= 0")
    parser.add_argument("--fixed_abs2", default=-1.0, type=float, help="Use fixed selective mode-2 abs threshold when >= 0")
    args = parser.parse_args()

    train = read_samples_csv(args.train_csv)
    test = read_samples_csv(args.test_csv)

    fixed_enabled = args.fixed_abs1 >= 0.0 and args.fixed_cam >= 0.0 and args.fixed_abs2 >= 0.0
    if fixed_enabled:
        abs1, cam_thr, abs2 = float(args.fixed_abs1), float(args.fixed_cam), float(args.fixed_abs2)
    else:
        abs_grid = np.round(np.arange(0.55, 0.95, 0.01), 2)
        cam_grid = np.round(np.arange(7.0, 10.8, 0.1), 2)
        abs2_grid = np.round(np.arange(0.45, 0.85, 0.01), 2)

        best_score = None
        best_policy = None

        for abs1 in abs_grid:
            mode1 = train["finetuned_lmk_abs_max"] > abs1
            if float(np.mean(mode1)) > 0.9:
                continue
            for cam_thr in cam_grid:
                cam_mask = train["finetuned_cam_score"] > cam_thr
                if float(np.mean(cam_mask)) < 0.01:
                    continue
                for abs2 in abs2_grid:
                    gate = mode1 | (cam_mask & (train["finetuned_lmk_abs_max"] > abs2))
                    fused = np.where(gate, train["baseline_lmk_err"], train["finetuned_lmk_err"])
                    score = (
                        float(np.mean(fused > 2.0)),
                        trimmed_mean(fused),
                        float(np.mean(fused)),
                        float(np.mean(gate)),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_policy = (float(abs1), float(cam_thr), float(abs2))

        if best_policy is None:
            raise RuntimeError("No selective policy found during search")

        abs1, cam_thr, abs2 = best_policy

    test_gate = (test["finetuned_lmk_abs_max"] > abs1) | (
        (test["finetuned_cam_score"] > cam_thr) & (test["finetuned_lmk_abs_max"] > abs2)
    )
    test_selective = np.where(test_gate, test["baseline_lmk_err"], test["finetuned_lmk_err"])

    test_global_gate = test["finetuned_lmk_abs_max"] > 0.6
    test_global = np.where(test_global_gate, test["baseline_lmk_err"], test["finetuned_lmk_err"])

    ft_metrics = compute_metrics(test["finetuned_lmk_err"])
    bs_metrics = compute_metrics(test["baseline_lmk_err"])
    global_metrics = compute_metrics(test_global)
    selective_metrics = compute_metrics(test_selective)

    out_rows = []
    for i, r in enumerate(test["rows"]):
        out_rows.append(
            {
                "idx": r.get("idx", str(i)),
                "image_path": r.get("image_path", ""),
                "baseline_lmk_err": test["baseline_lmk_err"][i],
                "finetuned_lmk_err": test["finetuned_lmk_err"][i],
                "finetuned_lmk_abs_max": test["finetuned_lmk_abs_max"][i],
                "finetuned_cam_score": test["finetuned_cam_score"][i],
                "fallback_selective": int(test_gate[i]),
                "hybrid_err_selective": float(test_selective[i]),
            }
        )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "baseline_lmk_err",
                "finetuned_lmk_err",
                "finetuned_lmk_abs_max",
                "finetuned_cam_score",
                "fallback_selective",
                "hybrid_err_selective",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    lines = []
    lines.append("# Selective Subset Gating Experiment (beta0015 anchor)")
    lines.append("")
    lines.append(f"Policy learned on: `{args.train_csv}`")
    lines.append(f"Evaluated on: `{args.test_csv}`")
    lines.append("")
    lines.append(f"PolicyMode: {'fixed' if fixed_enabled else 'learned_from_train_csv'}")
    lines.append("## Learned Selective Gate")
    lines.append(f"- fallback if `finetuned_lmk_abs_max > {abs1:.2f}`")
    lines.append(f"- OR if (`finetuned_cam_score > {cam_thr:.2f}` AND `finetuned_lmk_abs_max > {abs2:.2f}`)")
    lines.append("")
    lines.append("## Test Metrics (2000 samples)")
    lines.append("| Policy | Mean | Trimmed | Median | P90 | FailRate | FallbackCount | FallbackRate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| FineTuned raw | {ft_metrics['mean']:.6f} | {ft_metrics['trimmed']:.6f} | {ft_metrics['median']:.6f} | {ft_metrics['p90']:.6f} | {ft_metrics['fail_rate']:.6f} | 0 | 0.000000 |"
    )
    lines.append(
        f"| Baseline raw | {bs_metrics['mean']:.6f} | {bs_metrics['trimmed']:.6f} | {bs_metrics['median']:.6f} | {bs_metrics['p90']:.6f} | {bs_metrics['fail_rate']:.6f} | {len(test['rows'])} | 1.000000 |"
    )
    lines.append(
        f"| Global gate abs>0.60 | {global_metrics['mean']:.6f} | {global_metrics['trimmed']:.6f} | {global_metrics['median']:.6f} | {global_metrics['p90']:.6f} | {global_metrics['fail_rate']:.6f} | {int(np.sum(test_global_gate))} | {float(np.mean(test_global_gate)):.6f} |"
    )
    lines.append(
        f"| Selective gate (clustered) | {selective_metrics['mean']:.6f} | {selective_metrics['trimmed']:.6f} | {selective_metrics['median']:.6f} | {selective_metrics['p90']:.6f} | {selective_metrics['fail_rate']:.6f} | {int(np.sum(test_gate))} | {float(np.mean(test_gate)):.6f} |"
    )
    lines.append("")
    lines.append("## Deltas vs Global gate abs>0.60")
    lines.append(f"- mean: {selective_metrics['mean'] - global_metrics['mean']:+.6f}")
    lines.append(f"- trimmed: {selective_metrics['trimmed'] - global_metrics['trimmed']:+.6f}")
    lines.append(f"- fail_rate: {selective_metrics['fail_rate'] - global_metrics['fail_rate']:+.6f}")
    lines.append(f"- fallback_count: {int(np.sum(test_gate)) - int(np.sum(test_global_gate)):+d}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[-8:]))
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
