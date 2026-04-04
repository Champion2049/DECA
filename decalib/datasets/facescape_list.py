import os
from collections import defaultdict

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset


class FaceScapeListDataset(Dataset):
    """Dataset for flattened FaceScape image exports using list files.

    Expected list file format: one absolute or relative image path per line.
    Landmark files are optional; if absent, a stable 68-point template is used.
    """

    def __init__(self, list_path, K=1, isSingle=True):
        self.K = 1 if isSingle else K
        self.isSingle = isSingle
        self.image_paths = self._read_list(list_path)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in list file: {list_path}")

        # Group images so K-view sampling can be done from same subject/expression bucket.
        grouped = defaultdict(list)
        for path in self.image_paths:
            grouped[self._group_key(path)].append(path)
        self.groups = [v for v in grouped.values() if len(v) > 0]
        self.template_68 = self._build_template_68()

    def _read_list(self, list_path):
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"List file does not exist: {list_path}")

        base_dir = os.path.dirname(os.path.abspath(list_path))
        image_paths = []
        with open(list_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip().strip('"')
                if not line:
                    continue
                path = line.replace("\\", os.sep)
                if not os.path.isabs(path):
                    path = os.path.join(base_dir, path)
                if os.path.isfile(path):
                    image_paths.append(os.path.normpath(path))
        return image_paths

    def _group_key(self, image_path):
        # Example filename: fsmview_trainset_10_10_dimpler_0.jpg
        # Group key: fsmview_trainset_10_10_dimpler
        stem = os.path.splitext(os.path.basename(image_path))[0]
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return stem

    def _build_template_68(self):
        # Mean-face style normalized landmarks in [0, 1], centered for 224x224 crops.
        # Used as fallback when explicit landmarks are unavailable.
        x = np.array([
            0.12, 0.16, 0.21, 0.27, 0.34, 0.41, 0.47, 0.53, 0.59, 0.66, 0.73, 0.79, 0.84, 0.88, 0.90, 0.92, 0.94,
            0.25, 0.29, 0.35, 0.41, 0.47, 0.53, 0.59, 0.65, 0.71,
            0.50, 0.50, 0.50, 0.50,
            0.39, 0.44, 0.50, 0.56, 0.61,
            0.30, 0.34, 0.39, 0.44, 0.39, 0.34,
            0.56, 0.61, 0.66, 0.70, 0.66, 0.61,
            0.36, 0.42, 0.48, 0.50, 0.52, 0.58, 0.64, 0.58, 0.52, 0.50, 0.48, 0.42,
            0.42, 0.47, 0.50, 0.53, 0.58, 0.53, 0.50, 0.47,
        ], dtype=np.float32)
        y = np.array([
            0.72, 0.79, 0.84, 0.88, 0.91, 0.93, 0.94, 0.94, 0.93, 0.91, 0.88, 0.84, 0.79, 0.72, 0.65, 0.58, 0.51,
            0.37, 0.34, 0.33, 0.34, 0.36, 0.36, 0.34, 0.33, 0.34,
            0.40, 0.47, 0.54, 0.61,
            0.64, 0.67, 0.68, 0.67, 0.64,
            0.42, 0.39, 0.38, 0.41, 0.43, 0.43,
            0.41, 0.38, 0.39, 0.42, 0.43, 0.43,
            0.76, 0.73, 0.72, 0.72, 0.72, 0.73, 0.76, 0.80, 0.82, 0.82, 0.82, 0.80,
            0.77, 0.76, 0.76, 0.76, 0.77, 0.79, 0.80, 0.79,
        ], dtype=np.float32)
        template = np.stack([x, y], axis=1)
        return template

    def _landmark_path(self, image_path):
        return os.path.splitext(image_path)[0] + ".npy"

    def _load_landmarks(self, image_path):
        lmk_path = self._landmark_path(image_path)
        if os.path.isfile(lmk_path):
            lmk = np.load(lmk_path)
            if lmk.shape[0] >= 68:
                lmk = lmk[:68, :2].astype(np.float32)
            else:
                lmk = None
        else:
            lmk = None

        if lmk is None:
            # Fallback template in normalized [-1, 1] space.
            return self.template_68 * 2.0 - 1.0
        return lmk

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group_paths = self.groups[idx]
        if len(group_paths) >= self.K:
            selected = np.random.choice(group_paths, size=self.K, replace=False)
        else:
            selected = np.random.choice(group_paths, size=self.K, replace=True)

        images_list = []
        kpt_list = []
        mask_list = []

        for image_path in selected:
            image = imread(image_path).astype(np.float32)
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=2)
            if image.shape[2] > 3:
                image = image[:, :, :3]
            image = image / 255.0

            h, w = image.shape[:2]
            lmk = self._load_landmarks(image_path)
            if lmk.max() > 1.1 or lmk.min() < -1.1:
                # Convert pixel landmarks to normalized [-1, 1] range.
                lmk = lmk.copy()
                lmk[:, 0] = (lmk[:, 0] / max(w, 1)) * 2.0 - 1.0
                lmk[:, 1] = (lmk[:, 1] / max(h, 1)) * 2.0 - 1.0

            mask = np.ones((h, w), dtype=np.float32)

            images_list.append(image.transpose(2, 0, 1))
            kpt_list.append(lmk.astype(np.float32))
            mask_list.append(mask)

        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)

        if self.isSingle:
            images_array = images_array.squeeze(0)
            kpt_array = kpt_array.squeeze(0)
            mask_array = mask_array.squeeze(0)

        return {
            "image": images_array,
            "landmark": kpt_array,
            "mask": mask_array,
        }
