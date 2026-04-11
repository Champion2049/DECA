import argparse
import csv
import os
import sys

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


def _to_safe_name(path, idx):
    base = os.path.splitext(os.path.basename(str(path)))[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base)
    return f"{idx:04d}_{safe}"


def _annotate(image, title, subtitle):
    canvas = image.copy()
    canvas = cv2.copyMakeBorder(canvas, 40, 6, 6, 6, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def _run_one(model: DECA, image):
    with torch.no_grad():
        codedict = model.encode(image)
        opdict, visdict = model.decode(codedict)
    landmarks = opdict["landmarks2d"][:, :, :2]
    rendered = opdict.get("rendered_images", None)
    return {
        "landmarks": landmarks,
        "rendered": rendered,
        "vis": model.visualize(visdict),
    }


def _mean_lmk_dist(lmk_a, lmk_b):
    return torch.sqrt(((lmk_a - lmk_b) ** 2).sum(dim=2)).mean().item()


def _mouth_lmk_dist(lmk_a, lmk_b):
    a = lmk_a[:, 48:68, :]
    b = lmk_b[:, 48:68, :]
    return torch.sqrt(((a - b) ** 2).sum(dim=2)).mean().item()


def _out_of_frame_score(lmk):
    excess = torch.relu(torch.abs(lmk) - 1.0)
    return excess.mean().item()


def _render_diff(img_a, img_b):
    if img_a is None or img_b is None:
        return float("nan")
    return torch.abs(img_a - img_b).mean().item()


def main():
    parser = argparse.ArgumentParser(description="Rank worst geometry mismatches between baseline and fine-tuned DECA")
    parser.add_argument("--baseline", required=True, type=str)
    parser.add_argument("--finetuned", required=True, type=str)
    parser.add_argument("--val_list", required=True, type=str)
    parser.add_argument("--max_samples", default=200, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--top_k", default=40, type=int)
    parser.add_argument("--out_csv", default="./logs/geometry_audit.csv", type=str)
    parser.add_argument("--debug_dir", default="./logs/geometry_audit_debug", type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.baseline):
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline}")
    if not os.path.isfile(args.finetuned):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    dataset = FaceScapeListDataset(args.val_list, K=1, isSingle=True)
    total = min(len(dataset), args.max_samples)

    baseline = load_deca(args.baseline, args.device)
    finetuned = load_deca(args.finetuned, args.device)

    rows = []
    for i in tqdm(range(total), desc="Auditing"):
        sample = dataset[i]
        image_path = sample.get("image_path", f"index_{i}")
        image = sample["image"].to(args.device)[None, ...]

        b = _run_one(baseline, image)
        f = _run_one(finetuned, image)

        lmk_dist = _mean_lmk_dist(b["landmarks"], f["landmarks"])
        mouth_dist = _mouth_lmk_dist(b["landmarks"], f["landmarks"])
        frame_penalty = _out_of_frame_score(f["landmarks"])
        render_l1 = _render_diff(b["rendered"], f["rendered"])
        if not np.isfinite(render_l1):
            render_l1 = 0.0

        mismatch_score = (
            1.0 * lmk_dist
            + 1.5 * mouth_dist
            + 10.0 * frame_penalty
            + 1.0 * render_l1
        )

        rows.append({
            "idx": i,
            "image_path": str(image_path),
            "mismatch_score": float(mismatch_score),
            "landmark_dist": float(lmk_dist),
            "mouth_landmark_dist": float(mouth_dist),
            "finetuned_out_of_frame": float(frame_penalty),
            "render_l1_baseline_vs_finetuned": float(render_l1),
            "baseline_vis": b["vis"],
            "finetuned_vis": f["vis"],
        })

    ranked = sorted(rows, key=lambda r: r["mismatch_score"], reverse=True)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "mismatch_score",
                "landmark_dist",
                "mouth_landmark_dist",
                "finetuned_out_of_frame",
                "render_l1_baseline_vs_finetuned",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    for row in ranked[: args.top_k]:
        left = _annotate(row["baseline_vis"], "Baseline", f"score_ref={row['mismatch_score']:.4f}")
        right = _annotate(row["finetuned_vis"], "Fine-tuned", f"mouth_dist={row['mouth_landmark_dist']:.4f}")
        combo = np.concatenate([left, right], axis=1)
        caption = f"path: {row['image_path']}"
        combo = cv2.copyMakeBorder(combo, 28, 8, 8, 8, cv2.BORDER_CONSTANT, value=(8, 8, 8))
        cv2.putText(combo, caption, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        out_name = _to_safe_name(row["image_path"], row["idx"]) + ".jpg"
        cv2.imwrite(os.path.join(args.debug_dir, out_name), combo)

    print(f"Audited samples: {total}")
    print(f"CSV: {args.out_csv}")
    print(f"Worst-case renders: {args.debug_dir}")


if __name__ == "__main__":
    main()
