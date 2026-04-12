import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from decalib.deca import DECA
from decalib.datasets.facescape_list import FaceScapeListDataset
from decalib.models.residual_code_head import ResidualCodeHead, apply_residual_correction, build_head_input
from decalib.utils.config import get_cfg_defaults


def _robust_mean(values: np.ndarray, trim_ratio: float = 0.05) -> float:
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    k = int(values.size * trim_ratio)
    if k == 0 or values.size - 2 * k <= 0:
        return float(values.mean())
    return float(values[k:-k].mean())


def _stats(errors, failure_threshold):
    err = np.array(errors, dtype=np.float64)
    return {
        "lmk_mean": float(err.mean()) if err.size else float("nan"),
        "lmk_trimmed_mean": _robust_mean(err),
        "lmk_median": float(np.median(err)) if err.size else float("nan"),
        "lmk_p90": float(np.percentile(err, 90)) if err.size else float("nan"),
        "lmk_failure_rate": float((err > failure_threshold).mean()) if err.size else float("nan"),
    }


def load_deca(checkpoint_path: str, device: str) -> DECA:
    cfg = get_cfg_defaults()
    cfg.pretrained_modelpath = checkpoint_path
    cfg.model.use_tex = False
    cfg.rasterizer_type = "standard"
    cfg.dataset.image_size = 224
    model = DECA(config=cfg, device=device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare baseline DECA vs baseline+residual head")
    parser.add_argument("--baseline", default="./data/deca_model.tar")
    parser.add_argument("--residual_head", required=True)
    parser.add_argument("--val_list", default="../val_list_wsl.txt")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--failure_threshold", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug_dir", required=True)
    parser.add_argument("--hybrid_auto_threshold", type=float, default=0.6)
    parser.add_argument("--hybrid_selective_abs1", type=float, default=0.90)
    parser.add_argument("--hybrid_selective_cam", type=float, default=10.0)
    parser.add_argument("--hybrid_selective_abs2", type=float, default=0.40)
    args = parser.parse_args()

    dataset = FaceScapeListDataset(args.val_list, K=1, isSingle=True)
    total = min(len(dataset), args.max_samples)

    baseline_model = load_deca(args.baseline, args.device)
    head_ckpt = torch.load(args.residual_head, map_location=torch.device(args.device))
    head = ResidualCodeHead(hidden=int(head_ckpt.get("hidden", 128))).to(args.device)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()

    rows = []
    base_errs = []
    corr_errs = []

    os.makedirs(args.debug_dir, exist_ok=True)

    for i in tqdm(range(total), desc="Comparing"):
        sample = dataset[i]
        image = sample["image"].to(args.device)[None, ...]
        lmk_gt = sample["landmark"].to(args.device)[None, ...]

        with torch.no_grad():
            base_code = baseline_model.encode(image, use_detail=True)
            base_out, base_vis = baseline_model.decode(base_code, rendering=True, return_vis=True, use_detail=True)
            res = head(build_head_input(base_code))
            corr_code = apply_residual_correction(base_code, res)
            corr_out, corr_vis = baseline_model.decode(corr_code, rendering=True, return_vis=True, use_detail=True)

        b_err = torch.sqrt(((base_out["landmarks2d"][:, :, :2] - lmk_gt[:, :, :2]) ** 2).sum(dim=2)).mean().item()
        c_err = torch.sqrt(((corr_out["landmarks2d"][:, :, :2] - lmk_gt[:, :, :2]) ** 2).sum(dim=2)).mean().item()

        cam = corr_code["cam"]
        cam_score = (cam[:, 0] - 1.0).abs().mean().item() + cam[:, 1:].abs().mean().item()
        lmk_abs_max = corr_out["landmarks2d"][:, :, :2].abs().max().item()

        rows.append(
            {
                "idx": i,
                "image_path": sample.get("image_path", f"idx_{i}"),
                "baseline_lmk_err": b_err,
                "residual_lmk_err": c_err,
                "delta_residual_minus_base": c_err - b_err,
                "residual_cam_score": cam_score,
                "residual_lmk_abs_max": lmk_abs_max,
            }
        )
        base_errs.append(b_err)
        corr_errs.append(c_err)

    base_stats = _stats(base_errs, args.failure_threshold)
    corr_stats = _stats(corr_errs, args.failure_threshold)

    # Visual-first selective fallback for residual model.
    fallback_mask = []
    fused = []
    for r in rows:
        use_base = (
            (r["residual_lmk_abs_max"] > args.hybrid_selective_abs1)
            or (
                r["residual_cam_score"] > args.hybrid_selective_cam
                and r["residual_lmk_abs_max"] > args.hybrid_selective_abs2
            )
        )
        fallback_mask.append(int(use_base))
        fused.append(r["baseline_lmk_err"] if use_base else r["residual_lmk_err"])
    hybrid_stats = _stats(fused, args.failure_threshold)

    all_csv = Path(args.debug_dir) / "all_samples.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "idx",
            "image_path",
            "baseline_lmk_err",
            "residual_lmk_err",
            "delta_residual_minus_base",
            "residual_cam_score",
            "residual_lmk_abs_max",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append("Residual Head Comparison Report")
    lines.append(f"Samples: {total}")
    lines.append("")
    lines.append("Metric, Baseline, ResidualHead, Better")

    def better(a, b):
        return "ResidualHead" if b < a else "Baseline"

    lines.append(f"LandmarkErrorMean, {base_stats['lmk_mean']:.6f}, {corr_stats['lmk_mean']:.6f}, {better(base_stats['lmk_mean'], corr_stats['lmk_mean'])}")
    lines.append(f"LandmarkErrorTrimmedMean, {base_stats['lmk_trimmed_mean']:.6f}, {corr_stats['lmk_trimmed_mean']:.6f}, {better(base_stats['lmk_trimmed_mean'], corr_stats['lmk_trimmed_mean'])}")
    lines.append(f"LandmarkErrorMedian, {base_stats['lmk_median']:.6f}, {corr_stats['lmk_median']:.6f}, {better(base_stats['lmk_median'], corr_stats['lmk_median'])}")
    lines.append(f"LandmarkErrorP90, {base_stats['lmk_p90']:.6f}, {corr_stats['lmk_p90']:.6f}, {better(base_stats['lmk_p90'], corr_stats['lmk_p90'])}")
    lines.append(f"LandmarkFailureRate(>2.0), {base_stats['lmk_failure_rate']:.6f}, {corr_stats['lmk_failure_rate']:.6f}, {better(base_stats['lmk_failure_rate'], corr_stats['lmk_failure_rate'])}")
    lines.append("")
    lines.append("HybridSelectiveClusterGate")
    lines.append(
        f"SelectivePolicy, abs1={args.hybrid_selective_abs1:.3f}, cam={args.hybrid_selective_cam:.3f}, abs2={args.hybrid_selective_abs2:.3f}"
    )
    lines.append(
        f"SelectiveStats, fallback={int(np.sum(fallback_mask))}, lmk_mean={hybrid_stats['lmk_mean']:.6f}, lmk_trimmed={hybrid_stats['lmk_trimmed_mean']:.6f}, lmk_median={hybrid_stats['lmk_median']:.6f}, lmk_p90={hybrid_stats['lmk_p90']:.6f}, fail_rate={hybrid_stats['lmk_failure_rate']:.6f}"
    )
    lines.append(f"AllSamplesCSV: {all_csv}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
