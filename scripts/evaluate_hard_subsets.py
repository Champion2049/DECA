import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from decalib.deca import DECA
from decalib.datasets.facescape_list import FaceScapeListDataset
from decalib.models.residual_code_head import ResidualCodeHead, apply_residual_correction, build_head_input
from decalib.utils.config import get_cfg_defaults


MOUTH_TAGS = ("mouth", "lip", "smile", "grin")
EYE_TAGS = ("eye", "brow", "squint", "blink")
POSE_TAGS = ("jaw_left", "jaw_right", "left", "right", "head", "pose")


def robust_trimmed_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    k = int(values.size * 0.05)
    if k == 0 or values.size - 2 * k <= 0:
        return float(values.mean())
    return float(values[k:-k].mean())


def load_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            image_path = r["image_path"]
            stem = os.path.basename(image_path).lower()
            b = float(r["baseline_lmk_err"])
            c = float(r["residual_lmk_err"])
            cam = float(r["residual_cam_score"])
            absmax = float(r["residual_lmk_abs_max"])
            rows.append(
                {
                    "idx": int(r["idx"]),
                    "image_path": image_path,
                    "basename": stem,
                    "base": b,
                    "res": c,
                    "cam": cam,
                    "abs": absmax,
                }
            )
    return rows


def in_bucket(row, bucket_name, pose_cam_threshold):
    name = row["basename"]
    if bucket_name == "overall":
        return True
    if bucket_name == "mouth":
        return any(t in name for t in MOUTH_TAGS)
    if bucket_name == "eyes":
        return any(t in name for t in EYE_TAGS)
    if bucket_name == "pose":
        return any(t in name for t in POSE_TAGS) or row["cam"] >= pose_cam_threshold
    return False


def compute_bucket(rows, bucket_name, abs1, cam_thr, abs2, pose_cam_threshold):
    sel = [r for r in rows if in_bucket(r, bucket_name, pose_cam_threshold)]
    if not sel:
        return None

    base = np.array([r["base"] for r in sel], dtype=np.float64)
    res = np.array([r["res"] for r in sel], dtype=np.float64)
    fallback = np.array([
        (r["abs"] > abs1) or (r["cam"] > cam_thr and r["abs"] > abs2)
        for r in sel
    ], dtype=bool)
    gated = np.where(fallback, base, res)

    return {
        "bucket": bucket_name,
        "n": int(len(sel)),
        "baseline_mean": float(base.mean()),
        "residual_mean": float(res.mean()),
        "gated_mean": float(gated.mean()),
        "baseline_trimmed": robust_trimmed_mean(base),
        "residual_trimmed": robust_trimmed_mean(res),
        "gated_trimmed": robust_trimmed_mean(gated),
        "residual_win_rate": float((res < base).mean()),
        "gated_win_rate": float((gated < base).mean()),
        "fallback_rate": float(fallback.mean()),
        "residual_p90": float(np.percentile(res, 90)),
        "gated_p90": float(np.percentile(gated, 90)),
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


def t2bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_panel(input_img, b_img, r_img, h_img, title: str, out_path: Path):
    h, w = input_img.shape[:2]
    canvas = np.full((h + 60, w * 4 + 30, 3), 255, dtype=np.uint8)
    imgs = [input_img, b_img, r_img, h_img]
    labels = ["INPUT", "BASELINE", "RESIDUAL", "HYBRID(GATED)"]
    x = 6
    for i, im in enumerate(imgs):
        canvas[30 : 30 + h, x : x + w] = im
        cv2.rectangle(canvas, (x, 30), (x + w, 30 + h), (180, 180, 180), 1)
        cv2.putText(canvas, labels[i], (x + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
        x += w + 6
    cv2.putText(canvas, title, (8, h + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def generate_evidence(args, rows, out_dir: Path, abs1, cam_thr, abs2, pose_cam_threshold):
    bucket_rows = {k: [] for k in ["mouth", "eyes", "pose"]}
    for r in rows:
        delta = r["res"] - r["base"]
        use_base = (r["abs"] > abs1) or (r["cam"] > cam_thr and r["abs"] > abs2)
        entry = {**r, "delta": delta, "use_base": use_base}
        for b in bucket_rows.keys():
            if in_bucket(r, b, pose_cam_threshold):
                bucket_rows[b].append(entry)

    selected = {}
    for b, arr in bucket_rows.items():
        arr_sorted_best = sorted(arr, key=lambda x: x["delta"])[: args.evidence_topk]
        arr_sorted_worst = sorted(arr, key=lambda x: x["delta"], reverse=True)[: args.evidence_topk]
        selected[b] = {"best": arr_sorted_best, "worst": arr_sorted_worst}

    all_idxs = set()
    for b in selected.values():
        for c in b["best"] + b["worst"]:
            all_idxs.add(c["idx"])

    dataset = FaceScapeListDataset(args.val_list, K=1, isSingle=True)
    baseline = load_deca(args.baseline, args.device)
    head_ckpt = torch.load(args.residual_head, map_location=torch.device(args.device))
    head = ResidualCodeHead(hidden=int(head_ckpt.get("hidden", 128))).to(args.device)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()

    cache = {}
    with torch.no_grad():
        for idx in sorted(all_idxs):
            sample = dataset[idx]
            image = sample["image"].to(args.device)[None, ...]
            base_code = baseline.encode(image, use_detail=True)
            base_out = baseline.decode(base_code, rendering=True, return_vis=False, use_detail=True)
            res = head(build_head_input(base_code))
            corr_code = apply_residual_correction(base_code, res)
            corr_out = baseline.decode(corr_code, rendering=True, return_vis=False, use_detail=True)

            use_base = (corr_out["landmarks2d"][:, :, :2].abs().max().item() > abs1) or (
                ((corr_code["cam"][:, 0] - 1.0).abs().mean().item() + corr_code["cam"][:, 1:].abs().mean().item()) > cam_thr
                and corr_out["landmarks2d"][:, :, :2].abs().max().item() > abs2
            )
            hyb_img = base_out["rendered_images"][0] if use_base else corr_out["rendered_images"][0]

            cache[idx] = {
                "input": t2bgr(image[0]),
                "base": t2bgr(base_out["rendered_images"][0]),
                "res": t2bgr(corr_out["rendered_images"][0]),
                "hyb": t2bgr(hyb_img),
            }

    manifest_lines = []
    for bucket_name, groups in selected.items():
        for tag, arr in groups.items():
            for rank, r in enumerate(arr, start=1):
                imgs = cache.get(r["idx"])
                if imgs is None:
                    continue
                title = (
                    f"{bucket_name}/{tag} rank={rank} idx={r['idx']} "
                    f"base={r['base']:.4f} res={r['res']:.4f} delta={r['delta']:+.4f}"
                )
                out_file = out_dir / bucket_name / tag / f"{rank:02d}_idx{r['idx']}.jpg"
                make_panel(imgs["input"], imgs["base"], imgs["res"], imgs["hyb"], title, out_file)
                manifest_lines.append(f"{bucket_name},{tag},{rank},{r['idx']},{out_file}")

    (out_dir / "manifest.csv").write_text("bucket,tag,rank,idx,path\n" + "\n".join(manifest_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hard subsets and build leaderboard/report")
    parser.add_argument("--csv", required=True, help="all_samples.csv from compare_residual_head")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--abs1", type=float, default=0.90)
    parser.add_argument("--cam", type=float, default=10.0)
    parser.add_argument("--abs2", type=float, default=0.40)
    parser.add_argument("--pose_cam_threshold", type=float, default=8.0)
    parser.add_argument("--model_label", default="residual_v3")
    parser.add_argument("--generate_evidence", action="store_true")
    parser.add_argument("--evidence_topk", type=int, default=6)
    parser.add_argument("--baseline", default="./data/deca_model.tar")
    parser.add_argument("--residual_head", default="")
    parser.add_argument("--val_list", default="../val_list_wsl.txt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = ["overall", "mouth", "eyes", "pose"]
    stats_rows = []
    for b in buckets:
        s = compute_bucket(rows, b, args.abs1, args.cam, args.abs2, args.pose_cam_threshold)
        if s is not None:
            stats_rows.append(s)

    leaderboard_csv = out_dir / "leaderboard.csv"
    with leaderboard_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "bucket",
            "n",
            "baseline_mean",
            "residual_mean",
            "gated_mean",
            "baseline_trimmed",
            "residual_trimmed",
            "gated_trimmed",
            "residual_win_rate",
            "gated_win_rate",
            "fallback_rate",
            "residual_p90",
            "gated_p90",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats_rows)

    md = []
    md.append(f"# Standout Report: {args.model_label}")
    md.append("")
    md.append(f"Policy: abs1={args.abs1:.2f}, cam={args.cam:.1f}, abs2={args.abs2:.2f}")
    md.append("")
    md.append("## Hard-Subset Leaderboard")
    md.append("")
    md.append("| Bucket | N | Base Trim | Residual Trim | Gated Trim | Residual Win-Rate | Gated Win-Rate | Fallback Rate |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in stats_rows:
        md.append(
            f"| {r['bucket']} | {r['n']} | {r['baseline_trimmed']:.6f} | {r['residual_trimmed']:.6f} | {r['gated_trimmed']:.6f} | {r['residual_win_rate']:.3f} | {r['gated_win_rate']:.3f} | {r['fallback_rate']:.3f} |"
        )

    md.append("")
    md.append("## Win-Rate Summary")
    for r in stats_rows:
        md.append(
            f"- {r['bucket']}: residual wins={r['residual_win_rate']:.3%}, gated wins={r['gated_win_rate']:.3%}, fallback={r['fallback_rate']:.3%}"
        )

    if args.generate_evidence:
        if not args.residual_head:
            raise ValueError("--residual_head is required when --generate_evidence is set")
        ev_dir = out_dir / "evidence"
        generate_evidence(args, rows, ev_dir, args.abs1, args.cam, args.abs2, args.pose_cam_threshold)
        md.append("")
        md.append("## Side-by-Side Evidence")
        md.append("- Evidence panels saved under `evidence/mouth`, `evidence/eyes`, and `evidence/pose` with `best` and `worst` groups.")
        md.append("- Per-file index is in `evidence/manifest.csv`.")

    report_path = out_dir / "standout_report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Saved leaderboard: {leaderboard_csv}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
