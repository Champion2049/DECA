import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from decalib.deca import DECA
from decalib.datasets.facescape_list import FaceScapeListDataset
from decalib.models.residual_code_head import ResidualCodeHead, apply_residual_correction, build_head_input
from decalib.utils.config import get_cfg_defaults


EYE_IDXS = list(range(36, 48))
MOUTH_IDXS = list(range(48, 68))


def lmk_err(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred[:, :, :2] - gt[:, :, :2]) ** 2).sum(dim=2)).mean(dim=1)


def weighted_lmk_loss(pred: torch.Tensor, gt: torch.Tensor, eye_weight: float, mouth_weight: float, beta: float) -> torch.Tensor:
    # Start with uniform per-point weights and boost eye/mouth regions for visual fidelity.
    num_pts = pred.shape[1]
    w = torch.ones((1, num_pts, 1), device=pred.device, dtype=pred.dtype)
    for idx in EYE_IDXS:
        if idx < num_pts:
            w[:, idx, :] = eye_weight
    for idx in MOUTH_IDXS:
        if idx < num_pts:
            w[:, idx, :] = mouth_weight
    loss = F.smooth_l1_loss(pred[:, :, :2], gt[:, :, :2], beta=beta, reduction="none")
    return (loss * w).mean()


def robust_trimmed_mean(values):
    arr = np.array(values, dtype=np.float64)
    arr = np.sort(arr)
    k = int(len(arr) * 0.05)
    if k == 0 or len(arr) - 2 * k <= 0:
        return float(arr.mean())
    return float(arr[k:-k].mean())


def evaluate(deca, head, loader, device, max_batches=50):
    head.eval()
    base_errs = []
    corr_errs = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            images = batch["image"].to(device)
            lmk_gt = batch["landmark"].to(device)
            base_code = deca.encode(images, use_detail=True)
            base_out = deca.decode(base_code, rendering=True, return_vis=False, use_detail=True)

            residual = head(build_head_input(base_code))
            corr_code = apply_residual_correction(base_code, residual)
            corr_out = deca.decode(corr_code, rendering=True, return_vis=False, use_detail=True)

            base_errs.extend(lmk_err(base_out["landmarks2d"], lmk_gt).cpu().numpy().tolist())
            corr_errs.extend(lmk_err(corr_out["landmarks2d"], lmk_gt).cpu().numpy().tolist())

    return {
        "baseline_mean": float(np.mean(base_errs)),
        "corrected_mean": float(np.mean(corr_errs)),
        "baseline_trimmed": robust_trimmed_mean(base_errs),
        "corrected_trimmed": robust_trimmed_mean(corr_errs),
    }


def main():
    parser = argparse.ArgumentParser(description="Train residual correction head over frozen DECA")
    parser.add_argument("--baseline_ckpt", default="./data/deca_model.tar")
    parser.add_argument("--train_list", default="../train_list_wsl.txt")
    parser.add_argument("--val_list", default="../val_list_wsl.txt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--eye_weight", type=float, default=1.0)
    parser.add_argument("--mouth_weight", type=float, default=1.0)
    parser.add_argument("--lmk_beta", type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg_defaults()
    cfg.pretrained_modelpath = args.baseline_ckpt
    cfg.model.use_tex = False
    cfg.rasterizer_type = "standard"
    cfg.dataset.image_size = 224

    deca = DECA(config=cfg, device=args.device).to(args.device)
    deca.eval()
    for p in deca.parameters():
        p.requires_grad = False

    head = ResidualCodeHead(hidden=128).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    train_ds = FaceScapeListDataset(args.train_list, K=1, isSingle=True, require_landmarks=True)
    val_ds = FaceScapeListDataset(args.val_list, K=1, isSingle=True, require_landmarks=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    log_rows = []
    best_val_trim = None
    best_path = out_dir / "residual_head_best.pt"

    step = 0
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            head.train()
            images = batch["image"].to(args.device)
            lmk_gt = batch["landmark"].to(args.device)

            with torch.no_grad():
                base_code = deca.encode(images, use_detail=True)
                base_out = deca.decode(base_code, rendering=True, return_vis=False, use_detail=True)

            residual = head(build_head_input(base_code))
            corr_code = apply_residual_correction(base_code, residual)
            corr_out = deca.decode(corr_code, rendering=True, return_vis=False, use_detail=True)

            loss_lmk = weighted_lmk_loss(
                corr_out["landmarks2d"],
                lmk_gt,
                eye_weight=args.eye_weight,
                mouth_weight=args.mouth_weight,
                beta=args.lmk_beta,
            )
            loss_teacher_lmk = F.smooth_l1_loss(corr_out["landmarks2d"], base_out["landmarks2d"].detach(), beta=0.02)
            loss_teacher_cam = F.smooth_l1_loss(corr_code["cam"], base_code["cam"].detach(), beta=0.02)
            loss_percept = F.l1_loss(corr_out["rendered_images"], images)
            loss_res = torch.mean(torch.abs(residual))

            loss = (
                1.0 * loss_lmk
                + 1.2 * loss_teacher_lmk
                + 0.6 * loss_teacher_cam
                + 0.5 * loss_percept
                + 0.1 * loss_res
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            row = {
                "step": step,
                "loss": float(loss.item()),
                "loss_lmk": float(loss_lmk.item()),
                "loss_teacher_lmk": float(loss_teacher_lmk.item()),
                "loss_teacher_cam": float(loss_teacher_cam.item()),
                "loss_percept": float(loss_percept.item()),
                "loss_res": float(loss_res.item()),
            }

            if step % args.val_every == 0 or step == args.steps - 1:
                metrics = evaluate(deca, head, val_loader, args.device, max_batches=50)
                row.update(metrics)
                val_trim = metrics["corrected_trimmed"]
                if best_val_trim is None or val_trim < best_val_trim:
                    best_val_trim = val_trim
                    torch.save(
                        {
                            "state_dict": head.state_dict(),
                            "hidden": 128,
                            "step": step,
                            "best_val_trimmed": best_val_trim,
                            "config": vars(args),
                        },
                        best_path,
                    )
            log_rows.append(row)
            step += 1

            if step % 50 == 0:
                print(f"step={step} loss={row['loss']:.6f}")

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in log_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(log_rows)

    print(f"Saved best head: {best_path}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
