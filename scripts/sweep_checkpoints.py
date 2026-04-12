import argparse
import csv
import glob
import os
import re
import subprocess
from pathlib import Path


def parse_metric(report_text, metric_name):
    pattern = rf"^{re.escape(metric_name)},\s*([0-9.eE+-]+),\s*([0-9.eE+-]+),"
    m = re.search(pattern, report_text, flags=re.MULTILINE)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def run_compare(baseline, finetuned, val_list, max_samples, device, out_report, debug_dir):
    cmd = [
        "python",
        "scripts/compare_models.py",
        "--baseline",
        baseline,
        "--finetuned",
        finetuned,
        "--val_list",
        val_list,
        "--max_samples",
        str(max_samples),
        "--device",
        device,
        "--failure_threshold",
        "2.0",
        "--top_k_failures",
        "20",
        "--top_k_examples",
        "20",
        "--hybrid_metric",
        "lmk_abs_max",
        "--hybrid_cam_thresholds",
        "0.6,0.7,0.8,0.9,1.0,1.1,1.2",
        "--hybrid_selective_abs1",
        "0.90",
        "--hybrid_selective_cam",
        "10.0",
        "--hybrid_selective_abs2",
        "0.40",
        "--out",
        out_report,
        "--debug_dir",
        debug_dir,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if os.path.exists(out_report):
        return Path(out_report).read_text(encoding="utf-8", errors="replace"), proc.returncode
    return proc.stdout, proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Sweep DECA checkpoint snapshots and summarize metrics.")
    parser.add_argument("--checkpoints_glob", required=True)
    parser.add_argument("--baseline", default="./data/deca_model.tar")
    parser.add_argument("--val_list", default="../val_list_wsl.txt")
    parser.add_argument("--max_samples", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    ckpts = sorted(glob.glob(args.checkpoints_glob))
    if not ckpts:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints_glob}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    for ckpt in ckpts:
        ckpt_name = Path(ckpt).stem
        report_path = os.path.join(args.out_dir, f"report_{ckpt_name}.txt")
        debug_dir = os.path.join(args.out_dir, f"debug_{ckpt_name}")
        print(f"[SWEEP] {ckpt_name}")

        report_text, rc = run_compare(
            baseline=args.baseline,
            finetuned=ckpt,
            val_list=args.val_list,
            max_samples=args.max_samples,
            device=args.device,
            out_report=report_path,
            debug_dir=debug_dir,
        )

        base_mean, ft_mean = parse_metric(report_text, "LandmarkErrorMean")
        base_trim, ft_trim = parse_metric(report_text, "LandmarkErrorTrimmedMean")
        base_fail, ft_fail = parse_metric(report_text, "LandmarkFailureRate(>2.0)")

        summary_rows.append(
            {
                "checkpoint": ckpt,
                "ckpt_name": ckpt_name,
                "return_code": rc,
                "baseline_mean": base_mean,
                "finetuned_mean": ft_mean,
                "baseline_trimmed": base_trim,
                "finetuned_trimmed": ft_trim,
                "baseline_fail_rate": base_fail,
                "finetuned_fail_rate": ft_fail,
            }
        )

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "ckpt_name",
                "return_code",
                "baseline_mean",
                "finetuned_mean",
                "baseline_trimmed",
                "finetuned_trimmed",
                "baseline_fail_rate",
                "finetuned_fail_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    scored = [
        r
        for r in summary_rows
        if r["finetuned_trimmed"] is not None and r["finetuned_fail_rate"] is not None and r["finetuned_mean"] is not None
    ]
    scored.sort(key=lambda r: (r["finetuned_trimmed"], r["finetuned_fail_rate"], r["finetuned_mean"]))

    print(f"[DONE] Summary: {summary_csv}")
    print("[TOP5] ranked by finetuned trimmed, fail rate, mean")
    for row in scored[:5]:
        print(
            row["ckpt_name"],
            f"trim={row['finetuned_trimmed']:.6f}",
            f"fail={row['finetuned_fail_rate']:.6f}",
            f"mean={row['finetuned_mean']:.6f}",
        )


if __name__ == "__main__":
    main()
