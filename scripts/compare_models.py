import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from decalib.deca import DECA
from decalib.datasets.facescape_list import FaceScapeListDataset
from decalib.utils.config import get_cfg_defaults


def load_deca(checkpoint_path: str, device: str) -> DECA:
    cfg = get_cfg_defaults()
    cfg.pretrained_modelpath = checkpoint_path
    cfg.model.use_tex = False
    cfg.rasterizer_type = "standard"
    cfg.dataset.image_size = 224
    model = DECA(config=cfg, device=device)
    model.eval()
    return model


def _robust_mean(values: np.ndarray, trim_ratio: float = 0.05) -> float:
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    k = int(values.size * trim_ratio)
    if k == 0:
        return float(values.mean())
    if values.size - 2 * k <= 0:
        return float(values.mean())
    return float(values[k:-k].mean())


def _safe_mean(values):
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _stats(errors, photos, runtimes, failure_threshold):
    err = np.array(errors, dtype=np.float64)
    pho = np.array(photos, dtype=np.float64)
    run = np.array(runtimes, dtype=np.float64)
    return {
        "lmk_mean": float(err.mean()) if err.size else float("nan"),
        "lmk_trimmed_mean": _robust_mean(err),
        "lmk_median": float(np.median(err)) if err.size else float("nan"),
        "lmk_p90": float(np.percentile(err, 90)) if err.size else float("nan"),
        "lmk_failure_rate": float((err > failure_threshold).mean()) if err.size else float("nan"),
        "photo_l1_mean": float(pho.mean()) if pho.size else float("nan"),
        "runtime_ms_mean": float(run.mean()) if run.size else float("nan"),
    }


def _run_one(model: DECA, image, lmk_gt, record_runtime=True):
    start = time.perf_counter()
    with torch.no_grad():
        codedict = model.encode(image)
        opdict, visdict = model.decode(codedict)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    lmk_pred = opdict["landmarks2d"][:, :, :2]
    lmk_err = torch.sqrt(((lmk_pred - lmk_gt[:, :, :2]) ** 2).sum(dim=2)).mean().item()
    photo_l1 = float("nan")
    if "rendered_images" in opdict:
        photo_l1 = torch.abs(opdict["rendered_images"] - image).mean().item()
    cam = codedict["cam"]
    cam_score = (cam[:, 0] - 1.0).abs().mean().item() + cam[:, 1:].abs().mean().item()
    lmk_abs_max = opdict["landmarks2d"][:, :, :2].abs().max().item()
    return {
        "lmk_err": float(lmk_err),
        "photo_l1": float(photo_l1),
        "runtime_ms": float(runtime_ms) if record_runtime else float("nan"),
        "cam_score": float(cam_score),
        "lmk_abs_max": float(lmk_abs_max),
        "visdict": visdict,
    }


def _annotate(image, title, subtitle):
    canvas = image.copy()
    canvas = cv2.copyMakeBorder(canvas, 40, 6, 6, 6, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def _to_safe_name(path, idx):
    base = os.path.splitext(os.path.basename(str(path)))[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base)
    return f"{idx:04d}_{safe}"


def _landmark_source(image_path: str) -> str:
    npy_path = os.path.splitext(str(image_path))[0] + ".npy"
    return "real_npy" if os.path.isfile(npy_path) else "template_fallback"


def main():
    parser = argparse.ArgumentParser(description="Compare DECA baseline vs fine-tuned checkpoints")
    parser.add_argument("--baseline", required=True, type=str)
    parser.add_argument("--finetuned", required=True, type=str)
    parser.add_argument("--val_list", required=True, type=str)
    parser.add_argument("--max_samples", default=100, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--out", default="./logs/model_compare_report.txt", type=str)
    parser.add_argument("--failure_threshold", default=2.0, type=float)
    parser.add_argument("--top_k_failures", default=20, type=int)
    parser.add_argument("--top_k_examples", default=20, type=int,
                        help="Number of best non-failure examples to render for sanity checks")
    parser.add_argument("--hybrid_auto_threshold", default=-1.0, type=float,
                        help="Force hybrid fallback threshold. If < 0, auto-select best from sweep or use metric default.")
    parser.add_argument("--debug_dir", default="./logs/model_compare_debug", type=str)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--hybrid_cam_thresholds", default="", type=str,
                        help="Comma-separated camera instability thresholds for baseline fallback, e.g. '0.6,0.8,1.0'")
    parser.add_argument("--hybrid_metric", default="lmk_abs_max", type=str,
                        help="Metric used for fallback thresholding: cam_score or lmk_abs_max")
    parser.add_argument("--hybrid_selective_abs1", default=0.90, type=float,
                        help="Selective gate mode-1 threshold: fallback when finetuned_lmk_abs_max > abs1.")
    parser.add_argument("--hybrid_selective_cam", default=10.0, type=float,
                        help="Selective gate mode-2 camera threshold: finetuned_cam_score > cam.")
    parser.add_argument("--hybrid_selective_abs2", default=0.40, type=float,
                        help="Selective gate mode-2 abs threshold: with cam gate, fallback when finetuned_lmk_abs_max > abs2.")
    args = parser.parse_args()

    if not os.path.isfile(args.baseline):
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline}")
    if not os.path.isfile(args.finetuned):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned}")

    dataset = FaceScapeListDataset(args.val_list, K=1, isSingle=True)

    baseline_model = load_deca(args.baseline, args.device)
    finetuned_model = load_deca(args.finetuned, args.device)

    total = min(len(dataset), args.max_samples)
    rows = []
    baseline_lmk, baseline_photo, baseline_runtime = [], [], []
    finetuned_lmk, finetuned_photo, finetuned_runtime = [], [], []

    os.makedirs(args.debug_dir, exist_ok=True)
    side_by_side_dir = os.path.join(args.debug_dir, "worst_failures")
    sanity_examples_dir = os.path.join(args.debug_dir, "sanity_examples")
    hybrid_failures_dir = os.path.join(args.debug_dir, "hybrid_auto_failures")
    os.makedirs(side_by_side_dir, exist_ok=True)
    os.makedirs(sanity_examples_dir, exist_ok=True)
    os.makedirs(hybrid_failures_dir, exist_ok=True)

    for i in tqdm(range(total), desc="Comparing"):
        sample = dataset[i]
        image_path = sample.get("image_path", f"index_{i}")
        lmk_source = _landmark_source(image_path)
        image = sample["image"].to(args.device)[None, ...]
        lmk_gt = sample["landmark"].to(args.device)[None, ...]

        base = _run_one(baseline_model, image, lmk_gt, record_runtime=(i >= args.warmup))
        ft = _run_one(finetuned_model, image, lmk_gt, record_runtime=(i >= args.warmup))

        baseline_lmk.append(base["lmk_err"])
        finetuned_lmk.append(ft["lmk_err"])
        if np.isfinite(base["photo_l1"]):
            baseline_photo.append(base["photo_l1"])
        if np.isfinite(ft["photo_l1"]):
            finetuned_photo.append(ft["photo_l1"])
        if np.isfinite(base["runtime_ms"]):
            baseline_runtime.append(base["runtime_ms"])
        if np.isfinite(ft["runtime_ms"]):
            finetuned_runtime.append(ft["runtime_ms"])

        rows.append({
            "idx": i,
            "image_path": str(image_path),
            "baseline_lmk_err": base["lmk_err"],
            "finetuned_lmk_err": ft["lmk_err"],
            "lmk_err_delta_ft_minus_base": ft["lmk_err"] - base["lmk_err"],
            "finetuned_cam_score": ft["cam_score"],
            "finetuned_lmk_abs_max": ft["lmk_abs_max"],
            "baseline_photo_l1": base["photo_l1"],
            "finetuned_photo_l1": ft["photo_l1"],
            "baseline_runtime_ms": base["runtime_ms"],
            "finetuned_runtime_ms": ft["runtime_ms"],
            "landmark_target_source": lmk_source,
            "is_failure": int(ft["lmk_err"] > args.failure_threshold),
            "baseline_vis": baseline_model.visualize(base["visdict"]),
            "finetuned_vis": finetuned_model.visualize(ft["visdict"]),
        })

    baseline = _stats(baseline_lmk, baseline_photo, baseline_runtime, args.failure_threshold)
    finetuned = _stats(finetuned_lmk, finetuned_photo, finetuned_runtime, args.failure_threshold)

    metric_key = "finetuned_cam_score" if args.hybrid_metric == "cam_score" else "finetuned_lmk_abs_max"
    thresholds = []
    if args.hybrid_cam_thresholds.strip() != "":
        for token in args.hybrid_cam_thresholds.split(','):
            token = token.strip()
            if token != "":
                thresholds.append(float(token))

    hybrid_sweep = []
    best_hybrid = None
    if len(thresholds) > 0:
        for thr in thresholds:
            h_lmk = []
            h_photo = []
            h_runtime = []
            fallback_count = 0
            for r in rows:
                use_base = r[metric_key] > thr
                if use_base:
                    fallback_count += 1
                    h_lmk.append(r["baseline_lmk_err"])
                    if np.isfinite(r["baseline_photo_l1"]):
                        h_photo.append(r["baseline_photo_l1"])
                    if np.isfinite(r["baseline_runtime_ms"]):
                        h_runtime.append(r["baseline_runtime_ms"])
                else:
                    h_lmk.append(r["finetuned_lmk_err"])
                    if np.isfinite(r["finetuned_photo_l1"]):
                        h_photo.append(r["finetuned_photo_l1"])
                    if np.isfinite(r["finetuned_runtime_ms"]):
                        h_runtime.append(r["finetuned_runtime_ms"])

            h_stats = _stats(h_lmk, h_photo, h_runtime, args.failure_threshold)
            hybrid_sweep.append((thr, fallback_count, h_stats))
            score = (h_stats['lmk_failure_rate'], h_stats['lmk_trimmed_mean'])
            if best_hybrid is None or score < best_hybrid[0]:
                best_hybrid = (score, thr, fallback_count, h_stats)

    if args.hybrid_auto_threshold >= 0.0:
        hybrid_threshold = float(args.hybrid_auto_threshold)
    elif best_hybrid is not None:
        hybrid_threshold = float(best_hybrid[1])
    else:
        hybrid_threshold = 8.0 if metric_key == "finetuned_cam_score" else 1.2

    def use_hybrid_base(row):
        return row[metric_key] > hybrid_threshold

    selective_enabled = (
        args.hybrid_selective_abs1 >= 0.0
        and args.hybrid_selective_cam >= 0.0
        and args.hybrid_selective_abs2 >= 0.0
    )

    selective_fallback_count = 0
    selective_stats = None
    if selective_enabled:
        s_lmk = []
        s_photo = []
        s_runtime = []
        for r in rows:
            use_base_sel = (
                (r["finetuned_lmk_abs_max"] > float(args.hybrid_selective_abs1))
                or (
                    r["finetuned_cam_score"] > float(args.hybrid_selective_cam)
                    and r["finetuned_lmk_abs_max"] > float(args.hybrid_selective_abs2)
                )
            )
            r["hybrid_selective_fallback"] = int(use_base_sel)
            if use_base_sel:
                selective_fallback_count += 1
                s_lmk.append(r["baseline_lmk_err"])
                if np.isfinite(r["baseline_photo_l1"]):
                    s_photo.append(r["baseline_photo_l1"])
                if np.isfinite(r["baseline_runtime_ms"]):
                    s_runtime.append(r["baseline_runtime_ms"])
            else:
                s_lmk.append(r["finetuned_lmk_err"])
                if np.isfinite(r["finetuned_photo_l1"]):
                    s_photo.append(r["finetuned_photo_l1"])
                if np.isfinite(r["finetuned_runtime_ms"]):
                    s_runtime.append(r["finetuned_runtime_ms"])
        selective_stats = _stats(s_lmk, s_photo, s_runtime, args.failure_threshold)

    lines = []
    lines.append("DECA Model Comparison Report")
    lines.append(f"Samples: {total}")
    lines.append("")
    lines.append("Metric, Baseline, FineTuned, Better")

    def better_low(a, b):
        return "FineTuned" if b < a else "Baseline"

    lines.append(
        f"LandmarkErrorMean, {baseline['lmk_mean']:.6f}, {finetuned['lmk_mean']:.6f}, {better_low(baseline['lmk_mean'], finetuned['lmk_mean'])}"
    )
    lines.append(
        f"LandmarkErrorTrimmedMean, {baseline['lmk_trimmed_mean']:.6f}, {finetuned['lmk_trimmed_mean']:.6f}, {better_low(baseline['lmk_trimmed_mean'], finetuned['lmk_trimmed_mean'])}"
    )
    lines.append(
        f"LandmarkErrorMedian, {baseline['lmk_median']:.6f}, {finetuned['lmk_median']:.6f}, {better_low(baseline['lmk_median'], finetuned['lmk_median'])}"
    )
    lines.append(
        f"LandmarkErrorP90, {baseline['lmk_p90']:.6f}, {finetuned['lmk_p90']:.6f}, {better_low(baseline['lmk_p90'], finetuned['lmk_p90'])}"
    )
    lines.append(
        f"LandmarkFailureRate(>2.0), {baseline['lmk_failure_rate']:.6f}, {finetuned['lmk_failure_rate']:.6f}, {better_low(baseline['lmk_failure_rate'], finetuned['lmk_failure_rate'])}"
    )
    lines.append(
        f"PhotometricL1Mean, {baseline['photo_l1_mean']:.6f}, {finetuned['photo_l1_mean']:.6f}, {better_low(baseline['photo_l1_mean'], finetuned['photo_l1_mean'])}"
    )
    lines.append(
        f"RuntimeMsMean, {baseline['runtime_ms_mean']:.3f}, {finetuned['runtime_ms_mean']:.3f}, {better_low(baseline['runtime_ms_mean'], finetuned['runtime_ms_mean'])}"
    )

    all_csv = os.path.join(args.debug_dir, "all_samples.csv")
    with open(all_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "baseline_lmk_err",
                "finetuned_lmk_err",
                "lmk_err_delta_ft_minus_base",
                "finetuned_cam_score",
                "finetuned_lmk_abs_max",
                "baseline_photo_l1",
                "finetuned_photo_l1",
                "baseline_runtime_ms",
                "finetuned_runtime_ms",
                "landmark_target_source",
                "is_failure",
                "hybrid_selective_fallback",
            ],
        )
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row.setdefault("hybrid_selective_fallback", 0)
            writer.writerow({k: out_row[k] for k in writer.fieldnames})

    real_rows = [r for r in rows if r["landmark_target_source"] == "real_npy"]
    template_rows = [r for r in rows if r["landmark_target_source"] == "template_fallback"]

    failing = [r for r in rows if r["is_failure"] == 1]
    failing_sorted = sorted(failing, key=lambda r: r["finetuned_lmk_err"], reverse=True)
    top_failures = failing_sorted[: args.top_k_failures]
    non_failing = [r for r in rows if r["is_failure"] == 0]
    top_examples = sorted(non_failing, key=lambda r: r["finetuned_lmk_err"])[: args.top_k_examples]

    fail_csv = os.path.join(args.debug_dir, "worst_failures.csv")
    with open(fail_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "baseline_lmk_err",
                "finetuned_lmk_err",
                "lmk_err_delta_ft_minus_base",
            ],
        )
        writer.writeheader()
        for row in top_failures:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    all_fail_csv = os.path.join(args.debug_dir, "all_failures.csv")
    with open(all_fail_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "baseline_lmk_err",
                "finetuned_lmk_err",
                "lmk_err_delta_ft_minus_base",
                "finetuned_cam_score",
                "finetuned_lmk_abs_max",
            ],
        )
        writer.writeheader()
        for row in failing_sorted:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    all_success_csv = os.path.join(args.debug_dir, "all_successes.csv")
    with open(all_success_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "baseline_lmk_err",
                "finetuned_lmk_err",
                "lmk_err_delta_ft_minus_base",
                "finetuned_cam_score",
                "finetuned_lmk_abs_max",
            ],
        )
        writer.writeheader()
        for row in sorted(non_failing, key=lambda r: r["finetuned_lmk_err"]):
            writer.writerow({k: row[k] for k in writer.fieldnames})

    examples_csv = os.path.join(args.debug_dir, "sanity_examples.csv")
    with open(examples_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "baseline_lmk_err",
                "finetuned_lmk_err",
                "lmk_err_delta_ft_minus_base",
            ],
        )
        writer.writeheader()
        for row in top_examples:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    worst_write_ok = 0
    worst_write_fail = 0
    hybrid_write_ok = 0
    hybrid_write_fail = 0
    sanity_write_ok = 0
    sanity_write_fail = 0

    for row in top_failures:
        left = _annotate(
            row["baseline_vis"],
            "Baseline",
            f"lmk={row['baseline_lmk_err']:.4f}",
        )
        right = _annotate(
            row["finetuned_vis"],
            "Fine-tuned",
            f"lmk={row['finetuned_lmk_err']:.4f}",
        )
        combo = np.concatenate([left, right], axis=1)
        caption = f"path: {row['image_path']}"
        combo = cv2.copyMakeBorder(combo, 28, 8, 8, 8, cv2.BORDER_CONSTANT, value=(8, 8, 8))
        cv2.putText(combo, caption, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        out_name = _to_safe_name(row["image_path"], row["idx"]) + ".jpg"
        if cv2.imwrite(os.path.join(side_by_side_dir, out_name), combo):
            worst_write_ok += 1
        else:
            worst_write_fail += 1

    for row in top_failures:
        use_base = use_hybrid_base(row)
        hybrid_vis = row["baseline_vis"] if use_base else row["finetuned_vis"]
        hybrid_src = "Baseline fallback" if use_base else "Fine-tuned"
        left = _annotate(
            row["baseline_vis"],
            "Baseline",
            f"lmk={row['baseline_lmk_err']:.4f}",
        )
        mid = _annotate(
            row["finetuned_vis"],
            "Fine-tuned",
            f"lmk={row['finetuned_lmk_err']:.4f}",
        )
        right = _annotate(
            hybrid_vis,
            "Hybrid (auto)",
            f"{hybrid_src}; {metric_key}={row[metric_key]:.4f}",
        )
        combo = np.concatenate([left, mid, right], axis=1)
        caption = f"path: {row['image_path']}"
        combo = cv2.copyMakeBorder(combo, 28, 8, 8, 8, cv2.BORDER_CONSTANT, value=(8, 8, 8))
        cv2.putText(combo, caption, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        out_name = _to_safe_name(row["image_path"], row["idx"]) + ".jpg"
        if cv2.imwrite(os.path.join(hybrid_failures_dir, out_name), combo):
            hybrid_write_ok += 1
        else:
            hybrid_write_fail += 1

    for row in top_examples:
        left = _annotate(
            row["baseline_vis"],
            "Baseline",
            f"lmk={row['baseline_lmk_err']:.4f}",
        )
        right = _annotate(
            row["finetuned_vis"],
            "Fine-tuned",
            f"lmk={row['finetuned_lmk_err']:.4f}",
        )
        combo = np.concatenate([left, right], axis=1)
        caption = f"path: {row['image_path']}"
        combo = cv2.copyMakeBorder(combo, 28, 8, 8, 8, cv2.BORDER_CONSTANT, value=(8, 8, 8))
        cv2.putText(combo, caption, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        out_name = _to_safe_name(row["image_path"], row["idx"]) + ".jpg"
        if cv2.imwrite(os.path.join(sanity_examples_dir, out_name), combo):
            sanity_write_ok += 1
        else:
            sanity_write_fail += 1

    lines.append("")
    lines.append(f"FailureThreshold: {args.failure_threshold}")
    lines.append(f"FailureCount: {len(failing)}")
    lines.append(f"RealLandmarkSamples: {len(real_rows)}")
    lines.append(f"TemplateLandmarkSamples: {len(template_rows)}")
    if len(real_rows) == 0:
        lines.append("WARNING: No real .npy landmarks found for evaluated samples.")
        lines.append("WARNING: LandmarkError* numbers are template-fallback proxy metrics, not geometric-accuracy ground truth.")
    lines.append(f"AllSamplesCSV: {all_csv}")
    lines.append(f"AllFailuresCSV: {all_fail_csv}")
    lines.append(f"AllSuccessesCSV: {all_success_csv}")
    lines.append(f"WorstFailuresCSV: {fail_csv}")
    lines.append(f"WorstFailureRendersDir: {side_by_side_dir}")
    lines.append(f"WorstFailureRendersWritten: ok={worst_write_ok}, failed={worst_write_fail}")
    lines.append(f"HybridAutoMetric: {metric_key}")
    lines.append(f"HybridAutoThreshold: {hybrid_threshold:.6f}")
    lines.append(f"HybridAutoFailureRendersDir: {hybrid_failures_dir}")
    lines.append(f"HybridAutoFailureRendersWritten: ok={hybrid_write_ok}, failed={hybrid_write_fail}")
    lines.append(f"SanityExamplesCSV: {examples_csv}")
    lines.append(f"SanityExamplesRendersDir: {sanity_examples_dir}")
    lines.append(f"SanityExamplesRendersWritten: ok={sanity_write_ok}, failed={sanity_write_fail}")

    if len(hybrid_sweep) > 0:
        lines.append("")
        lines.append(f"HybridFallbackResults (baseline fallback when {metric_key} > threshold)")
        for thr, fallback_count, h_stats in hybrid_sweep:
            lines.append(
                f"thr={thr:.3f}, fallback={fallback_count}, lmk_trimmed={h_stats['lmk_trimmed_mean']:.6f}, lmk_median={h_stats['lmk_median']:.6f}, fail_rate={h_stats['lmk_failure_rate']:.6f}, runtime_ms={h_stats['runtime_ms_mean']:.3f}"
            )
        if best_hybrid is not None:
            _, thr, fallback_count, h_stats = best_hybrid
            lines.append(
                f"HybridBest, thr={thr:.3f}, fallback={fallback_count}, lmk_mean={h_stats['lmk_mean']:.6f}, lmk_trimmed={h_stats['lmk_trimmed_mean']:.6f}, lmk_median={h_stats['lmk_median']:.6f}, lmk_p90={h_stats['lmk_p90']:.6f}, fail_rate={h_stats['lmk_failure_rate']:.6f}, runtime_ms={h_stats['runtime_ms_mean']:.3f}"
            )

    if selective_enabled and selective_stats is not None:
        lines.append("")
        lines.append("HybridSelectiveClusterGate")
        lines.append(
            f"SelectivePolicy, abs1={args.hybrid_selective_abs1:.3f}, cam={args.hybrid_selective_cam:.3f}, abs2={args.hybrid_selective_abs2:.3f}"
        )
        lines.append(
            f"SelectiveStats, fallback={selective_fallback_count}, lmk_mean={selective_stats['lmk_mean']:.6f}, lmk_trimmed={selective_stats['lmk_trimmed_mean']:.6f}, lmk_median={selective_stats['lmk_median']:.6f}, lmk_p90={selective_stats['lmk_p90']:.6f}, fail_rate={selective_stats['lmk_failure_rate']:.6f}, runtime_ms={selective_stats['runtime_ms_mean']:.3f}"
        )

    if len(real_rows) > 0:
        lines.append("")
        lines.append("Landmark Stats On Real .npy Targets Only")
        real_base = _stats(
            [r["baseline_lmk_err"] for r in real_rows],
            [r["baseline_photo_l1"] for r in real_rows if np.isfinite(r["baseline_photo_l1"])],
            [r["baseline_runtime_ms"] for r in real_rows if np.isfinite(r["baseline_runtime_ms"])],
            args.failure_threshold,
        )
        real_ft = _stats(
            [r["finetuned_lmk_err"] for r in real_rows],
            [r["finetuned_photo_l1"] for r in real_rows if np.isfinite(r["finetuned_photo_l1"])],
            [r["finetuned_runtime_ms"] for r in real_rows if np.isfinite(r["finetuned_runtime_ms"])],
            args.failure_threshold,
        )
        lines.append(
            f"RealNpyLandmarkErrorMean, {real_base['lmk_mean']:.6f}, {real_ft['lmk_mean']:.6f}, {better_low(real_base['lmk_mean'], real_ft['lmk_mean'])}"
        )
        lines.append(
            f"RealNpyLandmarkErrorTrimmedMean, {real_base['lmk_trimmed_mean']:.6f}, {real_ft['lmk_trimmed_mean']:.6f}, {better_low(real_base['lmk_trimmed_mean'], real_ft['lmk_trimmed_mean'])}"
        )
        lines.append(
            f"RealNpyLandmarkErrorMedian, {real_base['lmk_median']:.6f}, {real_ft['lmk_median']:.6f}, {better_low(real_base['lmk_median'], real_ft['lmk_median'])}"
        )
        lines.append(
            f"RealNpyLandmarkErrorP90, {real_base['lmk_p90']:.6f}, {real_ft['lmk_p90']:.6f}, {better_low(real_base['lmk_p90'], real_ft['lmk_p90'])}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
