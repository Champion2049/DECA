import argparse
import csv
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def _read_list(list_path: str) -> List[str]:
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")
    base_dir = os.path.dirname(os.path.abspath(list_path))
    items: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().strip('"')
            if not line:
                continue
            path = line.replace("\\", os.sep)
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            path = os.path.normpath(path)
            if os.path.isfile(path):
                items.append(path)
    return items


def _select_best_face(landmarks_list: List[np.ndarray]) -> Optional[np.ndarray]:
    if landmarks_list is None or len(landmarks_list) == 0:
        return None
    best = None
    best_area = -1.0
    for lmk in landmarks_list:
        if lmk is None or lmk.shape[0] < 68:
            continue
        x_min = float(np.min(lmk[:, 0]))
        x_max = float(np.max(lmk[:, 0]))
        y_min = float(np.min(lmk[:, 1]))
        y_max = float(np.max(lmk[:, 1]))
        area = max(0.0, (x_max - x_min) * (y_max - y_min))
        if area > best_area:
            best_area = area
            best = lmk
    return best


def _get_detector(device: str):
    try:
        import face_alignment  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise RuntimeError(
            "face-alignment is required. Install with `pip install face-alignment`."
        ) from exc

    return face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=device,
    )


def _draw_landmarks(image_bgr: np.ndarray, lmk: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    for p in lmk[:68]:
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 1, (0, 255, 0), -1)
    return out


def _save_landmarks(image_path: str, lmk: np.ndarray) -> str:
    out_path = os.path.splitext(image_path)[0] + ".npy"
    np.save(out_path, lmk[:68, :2].astype(np.float32))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate 68-point .npy landmarks for images in a list")
    parser.add_argument("--list", required=True, type=str, help="Path to list file (one image path per line)")
    parser.add_argument("--device", default="cuda", type=str, help="face-alignment device: cuda or cpu")
    parser.add_argument("--max_samples", default=0, type=int, help="0 means all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images that already have .npy")
    parser.add_argument("--save_debug", action="store_true", help="Save debug overlays for generated landmarks")
    parser.add_argument("--debug_dir", default="./logs/landmark_debug", type=str)
    parser.add_argument("--report", default="./logs/landmark_generation_report.csv", type=str)
    parser.add_argument("--failed_list", default="./logs/landmark_generation_failed.txt", type=str,
                        help="Write newline-separated image paths where detection failed")
    args = parser.parse_args()

    image_paths = _read_list(args.list)
    if len(image_paths) == 0:
        raise ValueError("No images found in list")
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    if args.save_debug:
        os.makedirs(args.debug_dir, exist_ok=True)

    detector = _get_detector(args.device)

    rows = []
    generated = 0
    skipped_existing = 0
    failed = 0

    for idx, image_path in enumerate(tqdm(image_paths, desc="Generating landmarks")):
        npy_path = os.path.splitext(image_path)[0] + ".npy"
        if args.skip_existing and os.path.isfile(npy_path):
            skipped_existing += 1
            rows.append({
                "idx": idx,
                "image_path": image_path,
                "status": "skipped_existing",
                "npy_path": npy_path,
            })
            continue

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            failed += 1
            rows.append({
                "idx": idx,
                "image_path": image_path,
                "status": "read_failed",
                "npy_path": "",
            })
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            lmk_list = detector.get_landmarks(image_rgb)
        except Exception:
            lmk_list = None

        best = _select_best_face(lmk_list)
        if best is None:
            failed += 1
            rows.append({
                "idx": idx,
                "image_path": image_path,
                "status": "detect_failed",
                "npy_path": "",
            })
            continue

        out_npy = _save_landmarks(image_path, best)
        generated += 1
        rows.append({
            "idx": idx,
            "image_path": image_path,
            "status": "generated",
            "npy_path": out_npy,
        })

        if args.save_debug:
            vis = _draw_landmarks(image_bgr, best)
            safe_name = f"{idx:05d}_" + os.path.basename(image_path)
            cv2.imwrite(os.path.join(args.debug_dir, safe_name), vis)

    with open(args.report, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "image_path", "status", "npy_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    os.makedirs(os.path.dirname(args.failed_list) or ".", exist_ok=True)
    with open(args.failed_list, "w", encoding="utf-8") as f:
        for row in rows:
            if row["status"] in ("read_failed", "detect_failed"):
                f.write(str(row["image_path"]) + "\n")

    print(f"Total: {len(image_paths)}")
    print(f"Generated: {generated}")
    print(f"SkippedExisting: {skipped_existing}")
    print(f"Failed: {failed}")
    print(f"Report: {args.report}")
    print(f"FailedList: {args.failed_list}")


if __name__ == "__main__":
    main()
