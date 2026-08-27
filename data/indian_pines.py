"""
data/indian_pines.py — PyTorch Dataset for the Indian Pines HSI benchmark.

Indian Pines:
  - Sensor: AVIRIS (Airborne Visible/Infrared Imaging Spectrometer)
  - Location: Northwestern Indiana, USA (agricultural)
  - Size: 145 × 145 pixels, 200 spectral bands (after removing water absorption)
  - GSD: 20 meters
  - Classes: 16 land-cover types (mostly crops + small vegetation classes)
  - Challenge: Severe class imbalance (class sizes range from 20 to 2468 pixels)

This dataset is small enough to be treated as a single "patch" — no tiling needed.
It's primarily used for fast development iteration and debugging of the full pipeline.
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple

from data.transforms import HSITransform
from data.utils import normalize_per_band


# Class names for Indian Pines (0 = background/unlabeled)
INDIAN_PINES_CLASSES = {
    0: "Background",
    1: "Alfalfa",
    2: "Corn-notill",
    3: "Corn-mintill",
    4: "Corn",
    5: "Grass-pasture",
    6: "Grass-trees",
    7: "Grass-pasture-mowed",
    8: "Hay-windrowed",
    9: "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-clean",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers",
}


class IndianPinesDataset(Dataset):
    """
    PyTorch Dataset for Indian Pines hyperspectral image.

    Since Indian Pines is a single 145×145 image, this dataset operates at
    the pixel level for classification, or returns the entire image (or
    padded patches) for segmentation.

    For segmentation (our use case): the entire image is one sample, optionally
    padded to a square patch_size and transformed.
    """

    def __init__(
        self,
        root_dir: str = "./datasets/indian_pines",
        transform: Optional[HSITransform] = None,
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed: int = 42,
        exclude_background: bool = True,
    ):
        """
        Args:
            root_dir: Directory containing the .mat files.
            transform: HSITransform instance for preprocessing/augmentation.
            split: One of "train", "val", "test", or None (full image).
            split_ratios: (train, val, test) fractions for pixel-level splitting.
            seed: Random seed for reproducible splits.
            exclude_background: If True, background pixels (class 0) are masked
                out of the loss computation. They're still in the image but their
                labels are set to -1 (ignored by CrossEntropyLoss).
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.exclude_background = exclude_background

        # --- Load the .mat files ---
        data_path = os.path.join(root_dir, "Indian_pines_corrected.mat")
        label_path = os.path.join(root_dir, "Indian_pines_gt.mat")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Indian Pines data not found at {data_path}. "
                f"Run: python data/download.py --dataset indian_pines"
            )

        # scipy.io.loadmat returns a dict with variable names as keys
        # The actual data array key varies by file — we find it by excluding
        # metadata keys that start with '__'
        data_mat = sio.loadmat(data_path)
        label_mat = sio.loadmat(label_path)

        # Extract the actual arrays (skip MATLAB metadata keys)
        self.data = self._extract_array(data_mat)     # (145, 145, 200)
        self.labels = self._extract_array(label_mat)   # (145, 145)

        # Validate shapes
        assert self.data.shape[:2] == self.labels.shape[:2], (
            f"Spatial dimensions mismatch: data {self.data.shape} vs labels {self.labels.shape}"
        )

        # Convert to float32 for processing
        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.int64)

        # --- Compute normalization statistics from the full image ---
        # (always computed from train split in practice, but Indian Pines is
        # so small that the difference is negligible)
        _, self.band_mean, self.band_std = normalize_per_band(self.data)

        # --- Create pixel-level train/val/test masks ---
        # For segmentation, we mask labels rather than splitting the image
        # spatially (which would lose spatial context at the edges).
        if split is not None:
            self.split_mask = self._create_split_mask(split, split_ratios, seed)
        else:
            self.split_mask = None

        # --- Dataset info ---
        self.num_bands = self.data.shape[2]
        self.num_classes = len(INDIAN_PINES_CLASSES) - 1  # Exclude background
        self.class_names = INDIAN_PINES_CLASSES
        self.spatial_shape = self.data.shape[:2]

        print(f"[IndianPines] Loaded: {self.data.shape}, "
              f"{self.num_classes} classes, split={split}")

    def _extract_array(self, mat_dict: dict) -> np.ndarray:
        """Extract the data array from a .mat file, skipping metadata keys."""
        for key, val in mat_dict.items():
            if not key.startswith("__") and isinstance(val, np.ndarray):
                return val
        raise ValueError(f"No data array found in .mat file. Keys: {list(mat_dict.keys())}")

    def _create_split_mask(
        self, split: str, ratios: Tuple[float, float, float], seed: int
    ) -> np.ndarray:
        """
        Create a pixel-level mask for train/val/test splitting.

        Strategy: stratified by class — for each class, randomly assign pixels
        to train/val/test according to ratios. This ensures each split has
        representation from every class, which is critical given Indian Pines'
        severe class imbalance (some classes have only 20 pixels).
        """
        rng = np.random.RandomState(seed)
        mask = np.full(self.labels.shape, -1, dtype=np.int64)  # -1 = not in this split

        for class_id in range(1, len(INDIAN_PINES_CLASSES)):  # Skip background (0)
            class_pixels = np.argwhere(self.labels == class_id)
            n = len(class_pixels)
            if n == 0:
                continue

            # Shuffle and split
            indices = rng.permutation(n)
            n_train = max(1, int(n * ratios[0]))  # At least 1 pixel per class
            n_val = max(1, int(n * ratios[1]))

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            # Assign split membership
            for idx in train_idx:
                r, c = class_pixels[idx]
                mask[r, c] = 0  # 0 = train
            for idx in val_idx:
                r, c = class_pixels[idx]
                mask[r, c] = 1  # 1 = val
            for idx in test_idx:
                r, c = class_pixels[idx]
                mask[r, c] = 2  # 2 = test

        split_map = {"train": 0, "val": 1, "test": 2}
        return (mask == split_map[split])

    def __len__(self) -> int:
        # For segmentation, we treat the entire image as one sample.
        # During training, augmentations provide variation.
        return 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the full Indian Pines image as a single sample.

        Returns:
            Dict with keys:
              - "data": (B, H, W) tensor
              - "labels": (H, W) tensor (class indices, -1 for ignored pixels)
              - "mask": (H, W) boolean tensor (which pixels are in this split)
        """
        data = self.data.copy()
        labels = self.labels.copy()

        # Mask out pixels not in this split (set their labels to -1,
        # which is ignored by PyTorch's CrossEntropyLoss)
        if self.split_mask is not None:
            labels[~self.split_mask] = -1

        # Also mask background pixels if requested
        if self.exclude_background:
            labels[self.labels == 0] = -1

        # Apply transforms (normalization, augmentation, resize)
        if self.transform is not None:
            data_tensor, labels_tensor = self.transform(data, labels)
        else:
            data_tensor = torch.from_numpy(data).float().permute(2, 0, 1)
            labels_tensor = torch.from_numpy(labels).long()

        return {
            "data": data_tensor,
            "labels": labels_tensor,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the focal loss.

        Classes with fewer pixels get higher weight to combat imbalance.
        Background (class 0) gets weight 0.
        """
        counts = np.bincount(self.labels.flatten(), minlength=len(INDIAN_PINES_CLASSES))
        # Avoid div by zero for missing classes
        counts = np.maximum(counts, 1)
        # Inverse frequency, normalized
        weights = 1.0 / counts.astype(np.float32)
        weights[0] = 0.0  # Background gets zero weight
        weights = weights / weights.sum() * len(weights)
        return torch.from_numpy(weights).float()
