"""
data/pavia.py — PyTorch Dataset for the Pavia University HSI benchmark.

Pavia University:
  - Sensor: ROSIS (Reflective Optics System Imaging Spectrometer)
  - Location: University of Pavia, Italy (urban + vegetation)
  - Size: 610 × 340 pixels, 103 spectral bands
  - GSD: 1.3 meters (much higher resolution than Indian Pines' 20m)
  - Classes: 9 land-cover types including vegetation categories
  - Challenge: Mix of urban and vegetation classes, irregular class shapes

This is our main evaluation dataset. It's large enough to require patching
(610×340 > 128×128 patch size), and has both vegetation and non-vegetation
classes — making NDVI masking in the AL loop meaningful.
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple

from data.transforms import HSITransform
from data.utils import normalize_per_band, extract_patches


# Class names for Pavia University (0 = background/unlabeled)
PAVIA_CLASSES = {
    0: "Background",
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare Soil",
    7: "Bitumen",
    8: "Self-Blocking Bricks",
    9: "Shadows",
}

# Vegetation-related classes (relevant for invasive species context)
# These are the classes where the spectral adapter should provide the most
# benefit, since vegetation spectral signatures (red-edge, chlorophyll
# absorption) are richer than what 3-band PCA can capture.
VEGETATION_CLASSES = {2, 4, 6}  # Meadows, Trees, Bare Soil (partially vegetated)


class PaviaDataset(Dataset):
    """
    PyTorch Dataset for Pavia University hyperspectral image.

    Tiles the 610×340 image into overlapping patches and presents each patch
    as a separate sample. This gives us multiple training samples from a
    single image, and each patch fits in GPU memory.
    """

    def __init__(
        self,
        root_dir: str = "./datasets/pavia",
        transform: Optional[HSITransform] = None,
        patch_size: int = 128,
        patch_overlap: float = 0.5,
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed: int = 42,
        exclude_background: bool = True,
    ):
        """
        Args:
            root_dir: Directory containing the .mat files.
            transform: HSITransform instance for preprocessing/augmentation.
            patch_size: Size of square patches to tile the image into.
            patch_overlap: Fraction of overlap between adjacent patches.
            split: One of "train", "val", "test", or None (full image).
            split_ratios: (train, val, test) fractions.
            seed: Random seed for reproducible splits.
            exclude_background: If True, background pixels' labels set to -1.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size
        self.exclude_background = exclude_background

        # --- Load the .mat files ---
        data_path = os.path.join(root_dir, "PaviaU.mat")
        label_path = os.path.join(root_dir, "PaviaU_gt.mat")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Pavia University data not found at {data_path}. "
                f"Run: python data/download.py --dataset pavia"
            )

        data_mat = sio.loadmat(data_path)
        label_mat = sio.loadmat(label_path)

        # Extract arrays — Pavia uses key 'paviaU' for data, 'paviaU_gt' for labels
        self.full_data = self._extract_array(data_mat).astype(np.float32)   # (610, 340, 103)
        self.full_labels = self._extract_array(label_mat).astype(np.int64)  # (610, 340)

        assert self.full_data.shape[:2] == self.full_labels.shape[:2], (
            f"Shape mismatch: data {self.full_data.shape} vs labels {self.full_labels.shape}"
        )

        # --- Compute normalization stats from full image ---
        _, self.band_mean, self.band_std = normalize_per_band(self.full_data)

        # --- Create pixel-level split mask ---
        if split is not None:
            self.split_mask = self._create_split_mask(split, split_ratios, seed)
        else:
            self.split_mask = None

        # --- Tile into patches ---
        self.patches = extract_patches(
            self.full_data, self.full_labels,
            patch_size=patch_size, overlap=patch_overlap
        )

        # Filter out patches that are entirely background (no useful training signal)
        self.patches = [
            p for p in self.patches
            if np.any(p["labels"] > 0)
        ]

        # --- Dataset info ---
        self.num_bands = self.full_data.shape[2]
        self.num_classes = len(PAVIA_CLASSES) - 1  # Exclude background
        self.class_names = PAVIA_CLASSES
        self.spatial_shape = self.full_data.shape[:2]

        print(f"[Pavia] Loaded: {self.full_data.shape}, "
              f"{self.num_classes} classes, {len(self.patches)} patches, split={split}")

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
        Stratified pixel-level train/val/test split.

        Same strategy as Indian Pines — split within each class independently
        to maintain class representation in every split.
        """
        rng = np.random.RandomState(seed)
        mask = np.full(self.full_labels.shape, -1, dtype=np.int64)

        for class_id in range(1, len(PAVIA_CLASSES)):
            class_pixels = np.argwhere(self.full_labels == class_id)
            n = len(class_pixels)
            if n == 0:
                continue

            indices = rng.permutation(n)
            n_train = max(1, int(n * ratios[0]))
            n_val = max(1, int(n * ratios[1]))

            for i, idx in enumerate(indices):
                r, c = class_pixels[idx]
                if i < n_train:
                    mask[r, c] = 0
                elif i < n_train + n_val:
                    mask[r, c] = 1
                else:
                    mask[r, c] = 2

        split_map = {"train": 0, "val": 1, "test": 2}
        return (mask == split_map[split])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single patch as a training sample.

        Returns:
            Dict with keys:
              - "data": (B, patch_size, patch_size) tensor
              - "labels": (patch_size, patch_size) tensor
              - "position": (row_start, col_start) for spatial bookkeeping
        """
        patch = self.patches[idx]
        data = patch["data"].copy()
        labels = patch["labels"].copy()

        # Apply split mask: set labels outside this split to -1
        if self.split_mask is not None:
            r0, c0 = patch["row_start"], patch["col_start"]
            ps = self.patch_size
            patch_mask = self.split_mask[r0 : r0 + ps, c0 : c0 + ps]
            labels[~patch_mask] = -1

        # Mask background
        if self.exclude_background:
            bg_mask = (patch["labels"] == 0)
            labels[bg_mask] = -1

        # Apply transforms
        if self.transform is not None:
            data_tensor, labels_tensor = self.transform(data, labels)
        else:
            data_tensor = torch.from_numpy(data).float().permute(2, 0, 1)
            labels_tensor = torch.from_numpy(labels).long()

        return {
            "data": data_tensor,
            "labels": labels_tensor,
            "position": torch.tensor([patch["row_start"], patch["col_start"]]),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for the loss function."""
        counts = np.bincount(
            self.full_labels.flatten(), minlength=len(PAVIA_CLASSES)
        )
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts.astype(np.float32)
        weights[0] = 0.0
        weights = weights / weights[1:].sum() * (len(weights) - 1)
        return torch.from_numpy(weights).float()

    def get_full_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the full (non-patched) image and labels.
        Useful for evaluation and visualization.
        """
        return self.full_data.copy(), self.full_labels.copy()
