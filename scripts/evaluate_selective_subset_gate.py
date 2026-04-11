import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def trimmed_mean(s: pd.Series, keep_ratio: float = 0.95) -> float:
    k = int(np.ceil(keep_ratio * len(s)))
    return float(s.sort_values().iloc[:k].mean())


def compute_metrics(err: pd.Series) -> dict:
    return {
        "mean": float(err.mean()),
        "trimmed": trimmed_mean(err),
        "median": float(err.median()),
        "p90": float(err.quantile(0.9)),
        "fail_rate": float((err > 2.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective subset gating policy")
    parser.add_argument("--train_csv", required=True, help="Guard-set all_samples.csv for policy selection")
    parser.add_argument("--test_csv", required=True, help="Evaluation all_samples.csv for final metrics")
    parser.add_argument("--out_md", required=True, help="Markdown report output path")
    parser.add_argument("--out_csv", required=True, help="Per-sample selective predictions output path")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)

    for df in (train, test):
        df["baseline_lmk_err"] = pd.to_numeric(df["baseline_lmk_err"], errors="coerce")
        df["finetuned_lmk_err"] = pd.to_numeric(df["finetuned_lmk_err"], errors="coerce")
        df["finetuned_lmk_abs_max"] = pd.to_numeric(df["finetuned_lmk_abs_max"], errors="coerce")
        df["finetuned_cam_score"] = pd.to_numeric(df["finetuned_cam_score"], errors="coerce")

    abs_grid = np.round(np.arange(0.55, 0.95, 0.01), 2)
    cam_grid = np.round(np.arange(7.0, 10.8, 0.1), 2)
    abs2_grid = np.round(np.arange(0.45, 0.85, 0.01), 2)

    best_score = None
    best_policy = None

    for abs1 in abs_grid:
        mode1 = train["finetuned_lmk_abs_max"] > abs1
        if mode1.mean() > 0.9:
            continue
        for cam_thr in cam_grid:
            cam_mask = train["finetuned_cam_score"] > cam_thr
            if cam_mask.mean() < 0.01:
                continue
            for abs2 in abs2_grid:
                gate = mode1 | (cam_mask & (train["finetuned_lmk_abs_max"] > abs2))
                fused = pd.Series(np.where(gate, train["baseline_lmk_err"], train["finetuned_lmk_err"]))
                score = (
                    float((fused > 2.0).mean()),
                    trimmed_mean(fused),
                    float(fused.mean()),
                    float(gate.mean()),
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
    test_selective = pd.Series(np.where(test_gate, test["baseline_lmk_err"], test["finetuned_lmk_err"]))

    test_global_gate = test["finetuned_lmk_abs_max"] > 0.6
    test_global = pd.Series(np.where(test_global_gate, test["baseline_lmk_err"], test["finetuned_lmk_err"]))

    ft_metrics = compute_metrics(test["finetuned_lmk_err"])
    bs_metrics = compute_metrics(test["baseline_lmk_err"])
    global_metrics = compute_metrics(test_global)
    selective_metrics = compute_metrics(test_selective)

    pred = test[
        [
            "idx",
            "image_path",
            "baseline_lmk_err",
            "finetuned_lmk_err",
            "finetuned_lmk_abs_max",
            "finetuned_cam_score",
        ]
    ].copy()
    pred["fallback_selective"] = test_gate.astype(int)
    pred["hybrid_err_selective"] = test_selective
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_csv, index=False)

    lines = []
    lines.append("# Selective Subset Gating Experiment (beta0015 anchor)")
    lines.append("")
    lines.append(f"Policy learned on: `{args.train_csv}`")
    lines.append(f"Evaluated on: `{args.test_csv}`")
    lines.append("")
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
        f"| Baseline raw | {bs_metrics['mean']:.6f} | {bs_metrics['trimmed']:.6f} | {bs_metrics['median']:.6f} | {bs_metrics['p90']:.6f} | {bs_metrics['fail_rate']:.6f} | {len(test)} | 1.000000 |"
    )
    lines.append(
        f"| Global gate abs>0.60 | {global_metrics['mean']:.6f} | {global_metrics['trimmed']:.6f} | {global_metrics['median']:.6f} | {global_metrics['p90']:.6f} | {global_metrics['fail_rate']:.6f} | {int(test_global_gate.sum())} | {float(test_global_gate.mean()):.6f} |"
    )
    lines.append(
        f"| Selective gate (clustered) | {selective_metrics['mean']:.6f} | {selective_metrics['trimmed']:.6f} | {selective_metrics['median']:.6f} | {selective_metrics['p90']:.6f} | {selective_metrics['fail_rate']:.6f} | {int(test_gate.sum())} | {float(test_gate.mean()):.6f} |"
    )
    lines.append("")
    lines.append("## Deltas vs Global gate abs>0.60")
    lines.append(f"- mean: {selective_metrics['mean'] - global_metrics['mean']:+.6f}")
    lines.append(f"- trimmed: {selective_metrics['trimmed'] - global_metrics['trimmed']:+.6f}")
    lines.append(f"- fail_rate: {selective_metrics['fail_rate'] - global_metrics['fail_rate']:+.6f}")
    lines.append(f"- fallback_count: {int(test_gate.sum()) - int(test_global_gate.sum()):+d}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[-8:]))
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
