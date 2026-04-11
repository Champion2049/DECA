import argparse
import csv
import subprocess
from pathlib import Path


def parse_metrics(report_path):
    base_mean = ft_mean = None
    base_trim = ft_trim = None
    base_fail = ft_fail = None
    for raw in Path(report_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("LandmarkErrorMean"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                base_mean, ft_mean = float(p[1]), float(p[2])
        elif line.startswith("LandmarkErrorTrimmedMean"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                base_trim, ft_trim = float(p[1]), float(p[2])
        elif line.startswith("LandmarkFailureRate(>2.0)"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                base_fail, ft_fail = float(p[1]), float(p[2])
    return {
        "baseline_mean": base_mean,
        "finetuned_mean": ft_mean,
        "baseline_trimmed": base_trim,
        "finetuned_trimmed": ft_trim,
        "baseline_fail_rate": base_fail,
        "finetuned_fail_rate": ft_fail,
    }


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed DECA sweep-hardgate experiments")
    parser.add_argument("--base_cfg", required=True)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--final_max_samples", type=int, default=2000)
    parser.add_argument("--sweep_max_samples", type=int, default=800)
    parser.add_argument("--results_csv", default="./logs/multiseed_v13_results.csv")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_cfg)
    base_text = base_cfg_path.read_text(encoding="utf-8")
    base_output_dir = ""
    for raw in base_text.splitlines():
        line = raw.strip()
        if line.startswith("output_dir:"):
            base_output_dir = line.split(":", 1)[1].strip().strip('"').strip("'")
            break
    if not base_output_dir:
        raise SystemExit("base cfg missing output_dir")

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    generated_dir = Path("./configs/generated_multiseed")
    generated_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in seed_list:
        cfg_path = generated_dir / f"{base_cfg_path.stem}_s{seed}.yml"
        run_output_dir = f"{base_output_dir}_s{seed}"
        lines = base_text.splitlines()
        out_lines = []
        in_train_block = False
        train_indent = None
        seed_inserted = False
        output_replaced = False

        for i, raw in enumerate(lines):
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip(" "))

            if stripped.startswith("output_dir:"):
                out_lines.append(f'output_dir: "{run_output_dir}"')
                output_replaced = True
                continue

            if stripped == "train:":
                in_train_block = True
                train_indent = indent
                seed_inserted = False
                out_lines.append(raw)
                continue

            if in_train_block:
                if stripped != "" and indent <= (train_indent or 0):
                    if not seed_inserted:
                        out_lines.append("  seed: " + str(seed))
                        seed_inserted = True
                    in_train_block = False
                elif stripped.startswith("seed:"):
                    out_lines.append("  seed: " + str(seed))
                    seed_inserted = True
                    continue

            out_lines.append(raw)

        if in_train_block and not seed_inserted:
            out_lines.append("  seed: " + str(seed))
        if not output_replaced:
            out_lines.insert(0, f'output_dir: "{run_output_dir}"')

        cfg_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        print(f"[SEED {seed}] running sweep-hardgate with {cfg_path}")
        cmd = [
            "bash",
            "scripts/wsl2_retrain_sweep_hardgate.sh",
            str(cfg_path),
            str(args.final_max_samples),
            str(args.sweep_max_samples),
        ]
        proc = subprocess.run(cmd)

        run_name = Path(run_output_dir).name
        run_tag = run_name[10:] if run_name.startswith("facescape_") else run_name
        decision_path = Path(f"./logs/{run_tag}_best_hardgate_decision.txt")
        report_path = Path(f"./logs/model_compare_report_{run_tag}_best.txt")

        if not report_path.exists():
            print(f"[SEED {seed}] missing report: {report_path}")
            rows.append(
                {
                    "seed": seed,
                    "exit_code": proc.returncode,
                    "report_path": str(report_path),
                    "decision_path": str(decision_path),
                    "best_checkpoint": "",
                    "baseline_mean": "",
                    "finetuned_mean": "",
                    "baseline_trimmed": "",
                    "finetuned_trimmed": "",
                    "baseline_fail_rate": "",
                    "finetuned_fail_rate": "",
                }
            )
            continue

        best_checkpoint = ""
        if decision_path.exists():
            for raw in decision_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if raw.startswith("BEST_CHECKPOINT="):
                    best_checkpoint = raw.split("=", 1)[1].strip()
                    break

        metrics = parse_metrics(report_path)
        row = {
            "seed": seed,
            "exit_code": proc.returncode,
            "report_path": str(report_path),
            "decision_path": str(decision_path),
            "best_checkpoint": best_checkpoint,
            **metrics,
        }
        rows.append(row)

    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "exit_code",
        "best_checkpoint",
        "baseline_mean",
        "finetuned_mean",
        "baseline_trimmed",
        "finetuned_trimmed",
        "baseline_fail_rate",
        "finetuned_fail_rate",
        "report_path",
        "decision_path",
    ]
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r.get("finetuned_trimmed") not in (None, "")]
    valid.sort(key=lambda r: (float(r["finetuned_trimmed"]), float(r["finetuned_fail_rate"]), float(r["finetuned_mean"])))

    print(f"[DONE] Results: {results_path}")
    if valid:
        best = valid[0]
        print("[BEST_SEED]", best["seed"], best["best_checkpoint"])
        print(
            "[BEST_METRICS]",
            f"mean={best['finetuned_mean']}",
            f"trimmed={best['finetuned_trimmed']}",
            f"fail={best['finetuned_fail_rate']}",
        )


if __name__ == "__main__":
    main()
