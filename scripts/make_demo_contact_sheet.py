import argparse
from pathlib import Path

import cv2
import numpy as np


def collect_files(folder: Path, suffix: str):
    return sorted([p for p in folder.glob(f"*{suffix}") if p.is_file()])


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.55, color=(20, 20, 20)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side baseline vs residual contact sheet")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--residual_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suffix", default="_vis.jpg")
    parser.add_argument("--cell_w", type=int, default=320)
    parser.add_argument("--cell_h", type=int, default=320)
    parser.add_argument("--rows", type=int, default=0, help="0 means auto")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    residual_dir = Path(args.residual_dir)

    base_files = collect_files(baseline_dir, args.suffix)
    res_files = collect_files(residual_dir, args.suffix)

    base_map = {p.name: p for p in base_files}
    res_map = {p.name: p for p in res_files}
    names = sorted(set(base_map.keys()) & set(res_map.keys()))
    if not names:
        raise RuntimeError("No matching images found between baseline and residual folders.")

    n = len(names)
    rows = args.rows if args.rows > 0 else n

    pair_gap = 28
    col_gap = 20
    header_h = 34
    footer_h = 42

    sheet_w = col_gap + args.cell_w + pair_gap + args.cell_w + col_gap
    sheet_h = rows * (header_h + args.cell_h + footer_h) + 20
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    y = 12
    for idx, name in enumerate(names[:rows]):
        b_img = cv2.imread(str(base_map[name]))
        r_img = cv2.imread(str(res_map[name]))
        if b_img is None or r_img is None:
            continue

        b_fit = fit_image(b_img, args.cell_w, args.cell_h)
        r_fit = fit_image(r_img, args.cell_w, args.cell_h)

        put_text(sheet, f"{idx + 1:02d}. {name}", col_gap, y + 22, scale=0.52)

        y0 = y + header_h
        x_base = col_gap
        x_res = col_gap + args.cell_w + pair_gap
        sheet[y0 : y0 + args.cell_h, x_base : x_base + args.cell_w] = b_fit
        sheet[y0 : y0 + args.cell_h, x_res : x_res + args.cell_w] = r_fit

        cv2.rectangle(sheet, (x_base, y0), (x_base + args.cell_w, y0 + args.cell_h), (180, 180, 180), 1)
        cv2.rectangle(sheet, (x_res, y0), (x_res + args.cell_w, y0 + args.cell_h), (180, 180, 180), 1)

        put_text(sheet, "BASELINE", x_base + 8, y0 + args.cell_h + 24, scale=0.56)
        put_text(sheet, "RESIDUAL", x_res + 8, y0 + args.cell_h + 24, scale=0.56)

        y += header_h + args.cell_h + footer_h

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)

    print(f"Saved contact sheet: {out_path}")
    print(f"Pairs rendered: {min(rows, n)}")


if __name__ == "__main__":
    main()
