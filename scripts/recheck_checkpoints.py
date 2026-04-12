import argparse
import csv
import re
import subprocess
from pathlib import Path


def parse_report(report_path: Path):
    metrics = {
        "baseline_mean": "",
        "finetuned_mean": "",
        "baseline_trimmed": "",
        "finetuned_trimmed": "",
        "baseline_fail_rate": "",
        "finetuned_fail_rate": "",
        "hybrid_best_threshold": "",
        "hybrid_best_trimmed": "",
        "hybrid_best_fail_rate": "",
        "selective_fallback": "",
        "selective_trimmed": "",
        "selective_fail_rate": "",
        "selective_mean": "",
    }

    if not report_path.exists():
        return metrics

    hybrid_re = re.compile(r"HybridBest,\s*thr=([0-9.]+).*lmk_trimmed=([0-9.]+).*fail_rate=([0-9.]+)")
    selective_re = re.compile(r"SelectiveStats,\s*fallback=([0-9]+).*lmk_mean=([0-9.]+).*lmk_trimmed=([0-9.]+).*fail_rate=([0-9.]+)")
    for raw in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("LandmarkErrorMean"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                metrics["baseline_mean"] = p[1]
                metrics["finetuned_mean"] = p[2]
        elif line.startswith("LandmarkErrorTrimmedMean"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                metrics["baseline_trimmed"] = p[1]
                metrics["finetuned_trimmed"] = p[2]
        elif line.startswith("LandmarkFailureRate(>2.0)"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                metrics["baseline_fail_rate"] = p[1]
                metrics["finetuned_fail_rate"] = p[2]
        else:
            m = hybrid_re.search(line)
            if m:
                metrics["hybrid_best_threshold"] = m.group(1)
                metrics["hybrid_best_trimmed"] = m.group(2)
                metrics["hybrid_best_fail_rate"] = m.group(3)
            m2 = selective_re.search(line)
            if m2:
                metrics["selective_fallback"] = m2.group(1)
                metrics["selective_mean"] = m2.group(2)
                metrics["selective_trimmed"] = m2.group(3)
                metrics["selective_fail_rate"] = m2.group(4)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="No-retune checkpoint re-check with compare_models")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoints", default="175,200,225")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hybrid_threshold", type=float, default=0.6)
    parser.add_argument("--hybrid_selective_abs1", type=float, default=0.90)
    parser.add_argument("--hybrid_selective_cam", type=float, default=10.0)
    parser.add_argument("--hybrid_selective_abs2", type=float, default=0.40)
    parser.add_argument("--out_csv", default="./logs/recheck_checkpoints.csv")
    parser.add_argument("--tag", default="seed11_micro_recheck")
    args = parser.parse_args()

    checkpoint_ids = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    out_rows = []

    for ckpt in checkpoint_ids:
        ckpt_name = f"{ckpt:08d}.tar"
        finetuned = Path(args.checkpoint_dir) / ckpt_name
        report_path = Path(f"./logs/model_compare_report_{args.tag}_{ckpt:08d}_{args.max_samples}.txt")
        debug_dir = Path(f"./logs/model_compare_debug_{args.tag}_{ckpt:08d}_{args.max_samples}")

        cmd = [
            "python",
            "scripts/compare_models.py",
            "--baseline",
            args.baseline,
            "--finetuned",
            str(finetuned),
            "--val_list",
            args.val_list,
            "--max_samples",
            str(args.max_samples),
            "--device",
            args.device,
            "--hybrid_metric",
            "lmk_abs_max",
            "--hybrid_auto_threshold",
            str(args.hybrid_threshold),
            "--hybrid_selective_abs1",
            str(args.hybrid_selective_abs1),
            "--hybrid_selective_cam",
            str(args.hybrid_selective_cam),
            "--hybrid_selective_abs2",
            str(args.hybrid_selective_abs2),
            "--out",
            str(report_path),
            "--debug_dir",
            str(debug_dir),
        ]

        proc = subprocess.run(cmd)
        row = {
            "checkpoint": ckpt_name,
            "exit_code": proc.returncode,
            "report_path": str(report_path),
            "debug_dir": str(debug_dir),
        }
        row.update(parse_report(report_path))
        out_rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "exit_code",
        "baseline_mean",
        "finetuned_mean",
        "baseline_trimmed",
        "finetuned_trimmed",
        "baseline_fail_rate",
        "finetuned_fail_rate",
        "hybrid_best_threshold",
        "hybrid_best_trimmed",
        "hybrid_best_fail_rate",
        "selective_fallback",
        "selective_mean",
        "selective_trimmed",
        "selective_fail_rate",
        "report_path",
        "debug_dir",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    valid = [r for r in out_rows if r["finetuned_trimmed"] not in ("", None)]
    valid.sort(key=lambda r: (float(r["finetuned_trimmed"]), float(r["finetuned_fail_rate"]), float(r["finetuned_mean"])))

    print(f"[DONE] wrote {out_path}")
    if valid:
        best = valid[0]
        print(
            "[BEST]",
            best["checkpoint"],
            f"mean={best['finetuned_mean']}",
            f"trimmed={best['finetuned_trimmed']}",
            f"fail={best['finetuned_fail_rate']}",
            f"hybrid_trimmed={best['hybrid_best_trimmed']}",
            f"hybrid_fail={best['hybrid_best_fail_rate']}",
            f"selective_trimmed={best['selective_trimmed']}",
            f"selective_fail={best['selective_fail_rate']}",
        )


if __name__ == "__main__":
    main()
