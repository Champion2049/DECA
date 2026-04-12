import argparse
import csv
from pathlib import Path

import numpy as np


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze visual mismatch risk patterns from compare_models all_samples.csv")
    parser.add_argument("--all_samples_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--policy_abs1", type=float, default=0.90)
    parser.add_argument("--policy_cam", type=float, default=10.0)
    parser.add_argument("--policy_abs2", type=float, default=0.40)
    parser.add_argument("--top_k", type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.all_samples_csv, "r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            base = to_float(r.get("baseline_lmk_err", "nan"))
            ft = to_float(r.get("finetuned_lmk_err", "nan"))
            delta = to_float(r.get("lmk_err_delta_ft_minus_base", "nan"))
            cam = to_float(r.get("finetuned_cam_score", "nan"))
            absmax = to_float(r.get("finetuned_lmk_abs_max", "nan"))
            fallback = int(float(r.get("hybrid_selective_fallback", "0") or 0))

            if np.isnan(delta):
                delta = ft - base

            # Risk score favors geometry instability proxies even when lmk looks fine.
            risk = 1.5 * max(0.0, absmax - args.policy_abs2) + 0.2 * max(0.0, cam - args.policy_cam)
            # Add penalty when fine-tuned is actually worse but still used.
            if fallback == 0 and delta > 0:
                risk += 2.0 * delta

            r2 = dict(r)
            r2["baseline_lmk_err"] = base
            r2["finetuned_lmk_err"] = ft
            r2["lmk_err_delta_ft_minus_base"] = delta
            r2["finetuned_cam_score"] = cam
            r2["finetuned_lmk_abs_max"] = absmax
            r2["hybrid_selective_fallback"] = fallback
            r2["risk_score"] = risk
            rows.append(r2)

    n = len(rows)
    if n == 0:
        raise SystemExit("No rows found")

    abs_vals = np.array([r["finetuned_lmk_abs_max"] for r in rows], dtype=np.float64)
    cam_vals = np.array([r["finetuned_cam_score"] for r in rows], dtype=np.float64)
    deltas = np.array([r["lmk_err_delta_ft_minus_base"] for r in rows], dtype=np.float64)

    q_abs95 = float(np.nanpercentile(abs_vals, 95))
    q_cam95 = float(np.nanpercentile(cam_vals, 95))

    def selective_gate(r):
        return (
            (r["finetuned_lmk_abs_max"] > args.policy_abs1)
            or (
                r["finetuned_cam_score"] > args.policy_cam
                and r["finetuned_lmk_abs_max"] > args.policy_abs2
            )
        )

    # 1) False-safe: fine-tuned worse than baseline but selective did not fallback.
    false_safe = [
        r for r in rows
        if r["hybrid_selective_fallback"] == 0 and r["lmk_err_delta_ft_minus_base"] > 0
    ]
    false_safe.sort(key=lambda r: (r["risk_score"], r["lmk_err_delta_ft_minus_base"]), reverse=True)

    # 2) Metric-improved but visually risky proxy (high cam/absmax tail) and still used.
    improved_but_risky = [
        r for r in rows
        if r["hybrid_selective_fallback"] == 0
        and r["lmk_err_delta_ft_minus_base"] < 0
        and (r["finetuned_lmk_abs_max"] >= q_abs95 or r["finetuned_cam_score"] >= q_cam95)
    ]
    improved_but_risky.sort(key=lambda r: r["risk_score"], reverse=True)

    # 3) Near-threshold unstable zone still used by selective gate.
    near_boundary = [
        r for r in rows
        if r["hybrid_selective_fallback"] == 0
        and (
            (args.policy_abs1 - 0.05) <= r["finetuned_lmk_abs_max"] <= (args.policy_abs1 + 0.02)
            or (
                (args.policy_cam - 0.4) <= r["finetuned_cam_score"] <= (args.policy_cam + 0.4)
                and r["finetuned_lmk_abs_max"] > (args.policy_abs2 - 0.05)
            )
        )
    ]
    near_boundary.sort(key=lambda r: r["risk_score"], reverse=True)

    # 4) Disagreement where fine-tuned is better but policy fallbacks; can indicate conservative gating.
    conservative_fallbacks = [
        r for r in rows
        if r["hybrid_selective_fallback"] == 1 and r["lmk_err_delta_ft_minus_base"] < 0
    ]
    conservative_fallbacks.sort(key=lambda r: r["lmk_err_delta_ft_minus_base"])  # most negative first

    short_fields = [
        "idx", "image_path", "baseline_lmk_err", "finetuned_lmk_err", "lmk_err_delta_ft_minus_base",
        "finetuned_cam_score", "finetuned_lmk_abs_max", "hybrid_selective_fallback", "risk_score"
    ]

    write_rows(out_dir / "false_safe_ft_worse_but_used.csv", false_safe[: args.top_k], short_fields)
    write_rows(out_dir / "improved_but_risky_proxy.csv", improved_but_risky[: args.top_k], short_fields)
    write_rows(out_dir / "near_boundary_used.csv", near_boundary[: args.top_k], short_fields)
    write_rows(out_dir / "conservative_fallback_ft_better.csv", conservative_fallbacks[: args.top_k], short_fields)

    fallback_rate = float(np.mean([r["hybrid_selective_fallback"] for r in rows]))
    false_safe_rate = float(len(false_safe)) / n
    improved_risky_rate = float(len(improved_but_risky)) / n
    near_boundary_rate = float(len(near_boundary)) / n

    md = []
    md.append("# Visual Mismatch Diagnosis (Selective Policy)")
    md.append("")
    md.append(f"Source: `{args.all_samples_csv}`")
    md.append(f"Rows: `{n}`")
    md.append("")
    md.append("## Policy")
    md.append(f"- abs1: `{args.policy_abs1}`")
    md.append(f"- cam: `{args.policy_cam}`")
    md.append(f"- abs2: `{args.policy_abs2}`")
    md.append("")
    md.append("## Key Rates")
    md.append(f"- selective fallback rate: `{fallback_rate:.4f}`")
    md.append(f"- false-safe (FT worse but used): `{len(false_safe)}/{n}` (`{false_safe_rate:.4f}`)")
    md.append(f"- improved-but-risky-proxy (tail cam/abs): `{len(improved_but_risky)}/{n}` (`{improved_risky_rate:.4f}`)")
    md.append(f"- near-boundary still-used: `{len(near_boundary)}/{n}` (`{near_boundary_rate:.4f}`)")
    md.append("")
    md.append("## Tail Thresholds Used For Proxy")
    md.append(f"- absmax p95: `{q_abs95:.6f}`")
    md.append(f"- cam p95: `{q_cam95:.6f}`")
    md.append("")
    md.append("## Ranked Case Lists")
    md.append("- `false_safe_ft_worse_but_used.csv`")
    md.append("- `improved_but_risky_proxy.csv`")
    md.append("- `near_boundary_used.csv`")
    md.append("- `conservative_fallback_ft_better.csv`")
    md.append("")
    md.append("## Suggested Visual-First Next Tweaks")
    md.append("- Keep visual-first policy as baseline: abs1=0.90, cam=10.0, abs2=0.40.")
    md.append("- If false-safe cases remain visually unacceptable, lower cam threshold in small steps (6.4 then 6.3) before changing abs1.")
    md.append("- Prioritize manual inspection of top 40 from false-safe and near-boundary lists before any new training.")

    (out_dir / "visual_mismatch_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Diagnosis written:")
    print(out_dir / "visual_mismatch_diagnosis.md")
    print(out_dir / "false_safe_ft_worse_but_used.csv")
    print(out_dir / "improved_but_risky_proxy.csv")
    print(out_dir / "near_boundary_used.csv")
    print(out_dir / "conservative_fallback_ft_better.csv")


if __name__ == "__main__":
    main()
