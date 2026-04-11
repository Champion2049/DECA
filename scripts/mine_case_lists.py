import argparse
import csv
from pathlib import Path


def to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def main():
    parser = argparse.ArgumentParser(description="Mine hard/good case lists from compare_models all_samples.csv")
    parser.add_argument("--all_samples", required=True)
    parser.add_argument("--hard_out", required=True)
    parser.add_argument("--good_out", required=True)
    parser.add_argument("--hard_delta", type=float, default=0.08)
    parser.add_argument("--hard_base_max", type=float, default=0.35)
    parser.add_argument("--good_delta", type=float, default=-0.05)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.all_samples, "r", encoding="utf-8")))
    real = [r for r in rows if r.get("landmark_target_source", "") == "real_npy"]

    n = len(real)
    fail = [r for r in real if int(to_float(r, "is_failure", 0)) == 1]
    improved = [r for r in real if to_float(r, "lmk_err_delta_ft_minus_base") < 0]
    strong_improved = [r for r in real if to_float(r, "lmk_err_delta_ft_minus_base") < args.good_delta]
    worse = [r for r in real if to_float(r, "lmk_err_delta_ft_minus_base") > 0.05]
    baseline_hard = [r for r in real if to_float(r, "baseline_lmk_err") > 0.30]

    print("N", n)
    print("FT_fail_rate", len(fail) / max(1, n))
    print("Improved_rate", len(improved) / max(1, n))
    print("Strong_improve_rate", len(strong_improved) / max(1, n))
    print("Worse_rate", len(worse) / max(1, n))
    print("Baseline_gt_0_30_count", len(baseline_hard))
    print(
        "Baseline_gt_0_30_ft_better_rate",
        sum(1 for r in baseline_hard if to_float(r, "lmk_err_delta_ft_minus_base") < 0)
        / max(1, len(baseline_hard)),
    )

    hard = [
        r
        for r in real
        if to_float(r, "lmk_err_delta_ft_minus_base") > args.hard_delta
        and to_float(r, "baseline_lmk_err") < args.hard_base_max
    ]
    good = [r for r in real if to_float(r, "lmk_err_delta_ft_minus_base") < args.good_delta]

    hard.sort(key=lambda r: to_float(r, "lmk_err_delta_ft_minus_base"), reverse=True)
    good.sort(key=lambda r: to_float(r, "lmk_err_delta_ft_minus_base"))

    cols = [
        "image_path",
        "baseline_lmk_err",
        "finetuned_lmk_err",
        "lmk_err_delta_ft_minus_base",
        "finetuned_lmk_abs_max",
    ]

    with open(args.hard_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in hard:
            w.writerow({k: r.get(k, "") for k in cols})

    with open(args.good_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in good:
            w.writerow({k: r.get(k, "") for k in cols})

    print("Hard_case_rows", len(hard))
    print("Good_case_rows", len(good))
    print("Wrote", args.hard_out)
    print("Wrote", args.good_out)


if __name__ == "__main__":
    main()
