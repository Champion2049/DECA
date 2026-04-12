import argparse
import csv
import json
from pathlib import Path

import numpy as np


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def robust_trimmed_mean(values: np.ndarray) -> float:
    vals = np.sort(values)
    k = int(len(vals) * 0.05)
    if k == 0 or len(vals) - 2 * k <= 0:
        return float(np.mean(vals))
    return float(np.mean(vals[k:-k]))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.05, steps: int = 3000, l2: float = 1e-4):
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    n = X.shape[0]
    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z)
        e = p - y
        dw = (X.T @ e) / n + l2 * w
        db = float(np.mean(e))
        w -= lr * dw
        b -= lr * db
    return w, b


def evaluate_policy(base: np.ndarray, ft: np.ndarray, fallback_mask: np.ndarray):
    hybrid = np.where(fallback_mask, base, ft)
    return {
        "fallback": int(np.sum(fallback_mask)),
        "fallback_rate": float(np.mean(fallback_mask)),
        "mean": float(np.mean(hybrid)),
        "trimmed": robust_trimmed_mean(hybrid),
        "p90": float(np.percentile(hybrid, 90)),
        "fail_rate": float(np.mean(hybrid > 2.0)),
        "false_safe": int(np.sum((~fallback_mask) & (ft > base))),
    }


def read_rows(path: str):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight visual-risk router from compare all_samples CSV")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_fallback", type=float, default=0.985, help="Maximum allowed fallback rate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(args.train_csv)
    eval_rows = read_rows(args.eval_csv)

    def featurize(rows):
        base = np.array([to_float(r.get("baseline_lmk_err", "nan")) for r in rows], dtype=np.float64)
        ft = np.array([to_float(r.get("finetuned_lmk_err", "nan")) for r in rows], dtype=np.float64)
        absmax = np.array([to_float(r.get("finetuned_lmk_abs_max", "nan")) for r in rows], dtype=np.float64)
        cam = np.array([to_float(r.get("finetuned_cam_score", "nan")) for r in rows], dtype=np.float64)
        delta = ft - base

        # Labels: 1 means baseline should be used (FT worse than baseline)
        y = (delta > 0).astype(np.float64)

        # Standardized features + interaction captures non-linear instability with tiny model.
        z_abs = (absmax - np.nanmean(absmax)) / (np.nanstd(absmax) + 1e-8)
        z_cam = (cam - np.nanmean(cam)) / (np.nanstd(cam) + 1e-8)
        z_inter = z_abs * z_cam
        X = np.stack([z_abs, z_cam, z_inter], axis=1)
        return base, ft, absmax, cam, y, X

    tr_base, tr_ft, _, _, tr_y, tr_X = featurize(train_rows)
    ev_base, ev_ft, ev_abs, _, _, ev_X = featurize(eval_rows)

    w, b = fit_logreg(tr_X, tr_y)
    tr_prob = sigmoid(tr_X @ w + b)
    ev_prob = sigmoid(ev_X @ w + b)

    # Baseline policy reference (current visual-first default)
    baseline_mask = (ev_abs > 0.90) | (((np.array([to_float(r.get("finetuned_cam_score", "nan")) for r in eval_rows]) > 10.0) & (ev_abs > 0.40)))
    baseline_stats = evaluate_policy(ev_base, ev_ft, baseline_mask)

    # Search probability threshold for visual-first objective
    best = None
    best_thr = None
    best_mask = None
    for thr in np.arange(0.10, 0.96, 0.01):
        mask = ev_prob >= thr
        stats = evaluate_policy(ev_base, ev_ft, mask)
        if stats["fail_rate"] > 0.0:
            continue
        if stats["fallback_rate"] > args.max_fallback:
            continue
        score = (stats["false_safe"], stats["trimmed"], stats["mean"], stats["fallback"])
        if best is None or score < best:
            best = score
            best_thr = float(round(thr, 2))
            best_mask = mask
            best_stats = stats

    if best_thr is None:
        raise SystemExit("No valid threshold found under constraints")

    out_model = {
        "features": ["z_absmax", "z_cam", "z_absmax_x_z_cam"],
        "weights": [float(x) for x in w],
        "bias": float(b),
        "recommended_threshold": best_thr,
        "objective": "visual-first minimize false_safe with zero fail_rate",
    }
    (out_dir / "visual_router_model.json").write_text(json.dumps(out_model, indent=2), encoding="utf-8")

    # Save top risky kept-by-router cases for manual review
    risky_rows = []
    for i, r in enumerate(eval_rows):
        keep_ft = not bool(best_mask[i])
        if keep_ft and ev_ft[i] > ev_base[i]:
            rr = {
                "idx": r.get("idx", str(i)),
                "image_path": r.get("image_path", ""),
                "baseline_lmk_err": float(ev_base[i]),
                "finetuned_lmk_err": float(ev_ft[i]),
                "delta_ft_minus_base": float(ev_ft[i] - ev_base[i]),
                "router_prob_baseline": float(ev_prob[i]),
            }
            risky_rows.append(rr)
    risky_rows.sort(key=lambda r: (r["delta_ft_minus_base"], r["router_prob_baseline"]), reverse=True)

    with (out_dir / "router_false_safe_cases.csv").open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=["idx", "image_path", "baseline_lmk_err", "finetuned_lmk_err", "delta_ft_minus_base", "router_prob_baseline"],
        )
        wcsv.writeheader()
        wcsv.writerows(risky_rows[:120])

    lines = []
    lines.append("# Visual Router Experiment")
    lines.append("")
    lines.append(f"Train CSV: `{args.train_csv}`")
    lines.append(f"Eval CSV: `{args.eval_csv}`")
    lines.append("")
    lines.append("## Baseline Policy (visual-first thresholds)")
    for k, v in baseline_stats.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Learned Router Policy")
    lines.append(f"- threshold: `{best_thr}`")
    for k, v in best_stats.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Delta (learned - baseline)")
    for k in ["fallback", "mean", "trimmed", "p90", "fail_rate", "false_safe"]:
        lines.append(f"- {k}: `{best_stats[k] - baseline_stats[k]}`")
    (out_dir / "visual_router_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(out_dir / "visual_router_model.json")
    print(out_dir / "visual_router_experiment.md")
    print(out_dir / "router_false_safe_cases.csv")


if __name__ == "__main__":
    main()
