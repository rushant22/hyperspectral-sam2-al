"""
data/tallgrass.py — PyTorch Dataset for the Tallgrass Prairie HSI dataset.

Tallgrass Prairie Preserve (Flight 017):
  - Sensor: AisaFENIX 1K
  - Location: Tallgrass Prairie Preserve, Oklahoma (The Nature Conservancy)
  - Acquisition: 23 July 2022
  - Spatial resolution: 1m
  - Bands: 235 usable (from 322 original, after atmospheric-window filtering)
  - Usable wavelength range: 433.67-2352.67 nm
  - Classes: 2 — Background (0), Sericea lespedeza (1)
  - Patch size: 128 x 128
  - Patches: 50 spatially distributed

This dataset differs from Pavia/Indian Pines in several ways:
  1. Pre-extracted patches (not a single large image to tile).
  2. Binary classification (not 9 or 16 classes).
  3. Large "ignore" fraction (~79%) — intermediate-cover pixels excluded from loss.
  4. Reflectance scaling from ENVI header: raw / 10000.
  5. Separate spatial train/val/test split by geographic row coordinate.

IMPORTANT: The Sericea_0_100.tif reference map used for mask generation is a
DERIVED product (modeled percent cover), NOT independent ground truth.
Do NOT report agreement with this map as independent field accuracy.
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple, List

from data.transforms import HSITransform
from data.utils import normalize_per_band


# Class names for Tallgrass (0 = background, 1 = Sericea lespedeza)
TALLGRASS_CLASSES = {
    0: "Background",
    1: "Sericea lespedeza",
}

# Reflectance scale factor from ENVI header (z plot titles: reflectance [%*100])
REFLECTANCE_SCALE = 10000.0

# Explicit NoData value from ENVI header (data ignore value = 15000)
ENVI_NODATA = 15000

# Mask values in the extracted *_mask.npy files
MASK_BACKGROUND = 0    # cover < 2.5% — confident absence
MASK_SERICEA = 1       # cover >= 20% — confident presence
MASK_IGNORE = 255      # 2.5-20% cover OR NaN reference — uncertain, exclude from loss


class TallgrassDataset(Dataset):
    """
    PyTorch Dataset for Tallgrass Prairie HSI patches.

    Loads pre-extracted 128x128 patches from Flight 017 and presents each
    as a separate sample. Handles reflectance scaling, mask mapping, and
    spatial train/val/test splitting.
    """

    def __init__(
        self,
        root_dir: str = "./datasets/tallgrass",
        transform: Optional[HSITransform] = None,
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
        repeat_factor: int = 1,
    ):
        """
        Args:
            root_dir: Directory containing the selected_patches/ subfolder
                with *_hsi.npy and *_mask.npy files.
            transform: HSITransform instance for preprocessing/augmentation.
            split: One of "train", "val", "test", or None (all patches).
            split_ratios: (train, val, test) fractions for spatial splitting.
            seed: Random seed for reproducible splits.
            repeat_factor: Repeat each patch this many times per epoch
                (helps with small dataset size, similar to Indian Pines fix).
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.repeat_factor = repeat_factor

        # --- Locate patch files ---
        patches_dir = os.path.join(root_dir, "selected_patches")
        if not os.path.exists(patches_dir):
            # Try direct root_dir (flat layout)
            patches_dir = root_dir

        hsi_files = sorted(glob.glob(os.path.join(patches_dir, "*_hsi.npy")))
        if len(hsi_files) == 0:
            raise FileNotFoundError(
                f"No *_hsi.npy files found in {patches_dir}. "
                f"Ensure the Tallgrass patches are extracted there."
            )

        mask_files = sorted(glob.glob(os.path.join(patches_dir, "*_mask.npy")))
        assert len(hsi_files) == len(mask_files), (
            f"Mismatch: {len(hsi_files)} HSI files vs {len(mask_files)} mask files"
        )

        # --- Parse patch coordinates from filenames for spatial splitting ---
        # Filename pattern: patch_rXXXXX_cXXXXX_hsi.npy
        self.all_patches = []
        for hsi_path, mask_path in zip(hsi_files, mask_files):
            basename = os.path.basename(hsi_path)
            match = re.match(r'patch_r(\d+)_c(\d+)_hsi\.npy', basename)
            if match:
                row_start = int(match.group(1))
                col_start = int(match.group(2))
            else:
                row_start, col_start = 0, 0

            self.all_patches.append({
                "hsi_path": hsi_path,
                "mask_path": mask_path,
                "row_start": row_start,
                "col_start": col_start,
            })

        # --- Spatial train/val/test split ---
        # Sort patches by row coordinate (north-to-south geographic order)
        # and split into contiguous blocks to minimize spatial leakage
        self.all_patches.sort(key=lambda p: (p["row_start"], p["col_start"]))
        n = len(self.all_patches)

        if split is not None:
            rng = np.random.RandomState(seed)
            # Assign patches to groups based on row coordinate
            # Use deterministic assignment: sort by row, then split
            n_train = max(1, int(n * split_ratios[0]))
            n_val = max(1, int(n * split_ratios[1]))

            # Group by unique row coordinates
            row_coords = [p["row_start"] for p in self.all_patches]
            unique_rows = sorted(set(row_coords))
            n_rows = len(unique_rows)
            n_train_rows = max(1, int(n_rows * split_ratios[0]))
            n_val_rows = max(1, int(n_rows * split_ratios[1]))

            # Shuffle row order deterministically for randomized spatial split
            shuffled_rows = list(unique_rows)
            rng.shuffle(shuffled_rows)

            row_to_split = {}
            for i, row in enumerate(shuffled_rows):
                if i < n_train_rows:
                    row_to_split[row] = "train"
                elif i < n_train_rows + n_val_rows:
                    row_to_split[row] = "val"
                else:
                    row_to_split[row] = "test"

            self.patches = [
                p for p in self.all_patches
                if row_to_split.get(p["row_start"]) == split
            ]
        else:
            self.patches = list(self.all_patches)

        # --- Load one patch to get dimensions ---
        sample_hsi = np.load(self.patches[0]["hsi_path"])
        self.num_bands = sample_hsi.shape[0]  # (B, H, W)
        self.patch_h = sample_hsi.shape[1]
        self.patch_w = sample_hsi.shape[2]

        # --- Compute normalization statistics from ALL patches ---
        # (computed from all patches, not just the split, for consistency)
        self.band_mean, self.band_std = self._compute_band_stats()

        # --- Dataset info ---
        self.num_classes = len(TALLGRASS_CLASSES) - 1  # Exclude background (=1: Sericea)
        self.class_names = TALLGRASS_CLASSES
        self.spatial_shape = (self.patch_h, self.patch_w)  # Per-patch shape

        print(f"[Tallgrass] Loaded: {len(self.patches)} patches "
              f"(of {n} total), {self.num_bands} bands, "
              f"{self.num_classes} class(es), split={split}, "
              f"repeat_factor={repeat_factor}")

    def _compute_band_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-band mean and std across ALL patches (not just current split).

        Uses reflectance-scaled values (raw / 10000). Masks out NoData (15000)
        from the stats computation.
        """
        # Accumulate online statistics to avoid loading all patches at once
        band_sum = np.zeros(self.num_bands, dtype=np.float64)
        band_sq_sum = np.zeros(self.num_bands, dtype=np.float64)
        pixel_count = np.zeros(self.num_bands, dtype=np.float64)

        for patch_info in self.all_patches:
            hsi = np.load(patch_info["hsi_path"]).astype(np.float32)  # (B, H, W)

            # Apply reflectance scaling
            hsi = hsi / REFLECTANCE_SCALE

            # Mask out NoData values (15000/10000 = 1.5) — values > 1.0 are suspect
            # Zero values are valid per the ENVI header (NoData = 15000, not 0)
            valid_mask = hsi < (ENVI_NODATA / REFLECTANCE_SCALE)  # (B, H, W)

            for b in range(self.num_bands):
                valid = hsi[b][valid_mask[b]]
                band_sum[b] += valid.sum()
                band_sq_sum[b] += (valid ** 2).sum()
                pixel_count[b] += valid.size

        mean = (band_sum / np.maximum(pixel_count, 1)).astype(np.float32)
        variance = (band_sq_sum / np.maximum(pixel_count, 1)) - mean ** 2
        std = np.sqrt(np.maximum(variance, 0)).astype(np.float32)

        # Prevent division by zero for constant bands
        std[std < 1e-8] = 1.0

        return mean, std

    def __len__(self) -> int:
        return len(self.patches) * self.repeat_factor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single patch as a training sample.

        Returns:
            Dict with keys:
              - "data": (B, H, W) tensor — reflectance-scaled, normalized HSI
              - "labels": (H, W) tensor — class indices, -1 for ignored pixels
              - "position": (row_start, col_start) for spatial bookkeeping
        """
        # Handle repeat factor
        real_idx = idx % len(self.patches)
        patch_info = self.patches[real_idx]

        # --- Load HSI and mask ---
        hsi = np.load(patch_info["hsi_path"]).astype(np.float32)  # (B, H, W)
        mask = np.load(patch_info["mask_path"]).astype(np.int64)   # (H, W)

        # --- Apply reflectance scaling ---
        hsi = hsi / REFLECTANCE_SCALE

        # --- Mask NoData pixels (15000 in raw = 1.5 in reflectance) ---
        nodata_mask = hsi >= (ENVI_NODATA / REFLECTANCE_SCALE)
        hsi[nodata_mask] = 0.0  # Zero out NoData pixels

        # --- Map mask values ---
        # 0 -> -1 (background, excluded from loss like Pavia/IP)
        # 1 -> 1 (Sericea)
        # 255 -> -1 (ignore in loss)
        labels = mask.copy()
        labels[mask == MASK_IGNORE] = -1
        labels[mask == MASK_BACKGROUND] = -1

        # --- Convert from (B, H, W) to (H, W, B) for transform ---
        hsi_hwb = hsi.transpose(1, 2, 0)  # (H, W, B)

        # --- Apply transforms (normalization + augmentation) ---
        if self.transform is not None:
            data_tensor, labels_tensor = self.transform(hsi_hwb, labels)
        else:
            # Manual conversion without transform
            data_tensor = torch.from_numpy(hsi_hwb.copy()).float().permute(2, 0, 1)
            labels_tensor = torch.from_numpy(labels.copy()).long()

        return {
            "data": data_tensor,
            "labels": labels_tensor,
            "position": torch.tensor([
                patch_info["row_start"], patch_info["col_start"]
            ]),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the loss function.

        Only considers confident labeled pixels (background=0, Sericea=1).
        Ignore pixels (255) are excluded.
        """
        total_bg = 0
        total_sericea = 0

        for patch_info in self.all_patches:
            mask = np.load(patch_info["mask_path"])
            total_bg += (mask == MASK_BACKGROUND).sum()
            total_sericea += (mask == MASK_SERICEA).sum()

        counts = np.array([max(1, total_bg), max(1, total_sericea)], dtype=np.float32)
        weights = 1.0 / counts
        weights[0] = 0.0  # Background gets zero weight (excluded via ignore_index)
        weights = weights / max(weights.sum(), 1e-8) * len(weights)

        return torch.from_numpy(weights).float()

    def get_al_composite(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a composite strip image for the active learning loop.

        Concatenates all patches in the current split horizontally into a
        single synthetic image of shape (H, total_W, B). This preserves
        the AL loop's interface which expects a single (B, H, W) image.

        Returns:
            full_data: (H, total_W, B) — reflectance-scaled HSI
            full_labels: (H, total_W) — ground truth labels (0=bg, 1=sericea)
                         -1 for ignore pixels (255 in source mask)
        """
        n = len(self.patches)
        total_w = self.patch_w * n
        full_data = np.zeros((self.patch_h, total_w, self.num_bands), dtype=np.float32)
        full_labels = np.full((self.patch_h, total_w), -1, dtype=np.int64)

        for i, patch_info in enumerate(self.patches):
            hsi = np.load(patch_info["hsi_path"]).astype(np.float32) / REFLECTANCE_SCALE
            mask = np.load(patch_info["mask_path"]).astype(np.int64)

            # NoData handling
            nodata = hsi >= (ENVI_NODATA / REFLECTANCE_SCALE)
            hsi[nodata] = 0.0

            # Map mask: 0 -> 0 (background, kept as class for oracle),
            #           1 -> 1 (Sericea), 255 -> -1 (ignore)
            labels = mask.copy()
            labels[mask == MASK_IGNORE] = -1

            col_start = i * self.patch_w
            full_data[:, col_start:col_start + self.patch_w, :] = hsi.transpose(1, 2, 0)
            full_labels[:, col_start:col_start + self.patch_w] = labels

        return full_data, full_labels

    def get_full_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the composite image (for compatibility with evaluation code).
        Delegates to get_al_composite().
        """
        return self.get_al_composite()

    @staticmethod
    def get_wavelengths(header_path: Optional[str] = None) -> np.ndarray:
        """
        Return the 235 usable wavelengths for the Tallgrass dataset.

        If header_path is provided, parses from the ENVI header.
        Otherwise, returns a hardcoded list based on Flight 017 header.
        """
        usable_ranges = [
            (431.10, 1299.36),
            (1487.71, 1775.03),
            (1998.23, 2353.76),
        ]

        if header_path is not None and os.path.exists(header_path):
            with open(header_path, 'r') as f:
                content = f.read()
            match = re.search(r'wavelength\s*=\s*\{(.*?)\}', content, re.DOTALL)
            if match:
                all_wls = [float(x.strip()) for x in match.group(1).split(',')
                           if x.strip()]
                filtered = [wl for wl in all_wls
                           if any(lo <= wl <= hi for lo, hi in usable_ranges)]
                return np.array(filtered)

        # Hardcoded fallback — first and last wavelengths for quick reference
        # Full list would be 235 values from 433.67 to 2352.67 nm
        return np.linspace(433.67, 2352.67, 235)
