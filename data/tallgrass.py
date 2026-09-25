"""
data/tallgrass.py — PyTorch Dataset for the Tallgrass Prairie HSI dataset.

Tallgrass Prairie Preserve (Flight 017):
  - Sensor: AisaFENIX 1K
  - Location: Tallgrass Prairie Preserve, Oklahoma
  - Acquisition: 23 July 2022
  - Spatial resolution: 1m
  - Bands: 235 usable bands
  - Usable wavelength range: approximately 433.67–2352.67 nm
  - Classes: 2 — Background (0), Sericea lespedeza (1)
  - Patch size: 128 x 128
  - Patches: 50 spatially distributed patches

Mask convention:
  0   = Background
  1   = Sericea lespedeza
  255 = Ignore / uncertain

Important:
  - Background is a valid segmentation class and is NOT ignored.
  - Only mask value 255 is converted to -1 for loss/evaluation.
  - Spatial splitting is performed by geographic row coordinate.
  - Normalization statistics are computed ONLY from training patches.
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple, List

from data.transforms import HSITransform


# =============================================================================
# Dataset constants
# =============================================================================

TALLGRASS_CLASSES = {
    0: "Background",
    1: "Sericea lespedeza",
}

# Raw ENVI values are scaled by 10000 to obtain reflectance.
REFLECTANCE_SCALE = 10000.0

# Explicit NoData value from the ENVI header.
ENVI_NODATA = 15000

# Mask values in *_mask.npy files.
MASK_BACKGROUND = 0
MASK_SERICEA = 1
MASK_IGNORE = 255


class TallgrassDataset(Dataset):
    """
    PyTorch Dataset for Tallgrass Prairie Preserve Flight 017.

    The dataset contains pre-extracted 128x128 HSI patches.

    HSI format:
        Raw file:       (bands, height, width)
        Dataset output: (bands, height, width)

    Label format:
        0   -> background
        1   -> Sericea
        255 -> converted to -1 (ignored)
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
            root_dir:
                Directory containing selected_patches/.

            transform:
                HSI preprocessing/augmentation transform.

            split:
                One of:
                    "train"
                    "val"
                    "test"
                    None

                None loads all patches.

            split_ratios:
                Geographic row split ratios:
                    train, validation, test

                Default:
                    70%, 10%, 20%

            seed:
                Kept for reproducibility/API compatibility.

                The spatial split itself is deterministic and does NOT
                randomly shuffle geographic rows.

            repeat_factor:
                Number of times each training patch is repeated per epoch.
        """

        super().__init__()

        if split not in (None, "train", "val", "test"):
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected None, 'train', 'val', or 'test'."
            )

        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed
        self.repeat_factor = max(1, int(repeat_factor))

        # ---------------------------------------------------------------------
        # Locate patch directory
        # ---------------------------------------------------------------------

        patches_dir = os.path.join(root_dir, "selected_patches")

        if not os.path.exists(patches_dir):
            # Also support a flat directory layout.
            patches_dir = root_dir

        if not os.path.exists(patches_dir):
            raise FileNotFoundError(
                f"Tallgrass patch directory not found:\n{patches_dir}"
            )

        # ---------------------------------------------------------------------
        # Find HSI and mask files
        # ---------------------------------------------------------------------

        hsi_files = sorted(
            glob.glob(os.path.join(patches_dir, "*_hsi.npy"))
        )

        mask_files = sorted(
            glob.glob(os.path.join(patches_dir, "*_mask.npy"))
        )

        if len(hsi_files) == 0:
            raise FileNotFoundError(
                f"No *_hsi.npy files found in:\n{patches_dir}"
            )

        if len(hsi_files) != len(mask_files):
            raise RuntimeError(
                f"Mismatch between HSI and mask files:\n"
                f"  HSI files  : {len(hsi_files)}\n"
                f"  Mask files : {len(mask_files)}"
            )

        # ---------------------------------------------------------------------
        # Build patch metadata
        # ---------------------------------------------------------------------

        self.all_patches: List[Dict] = []

        for hsi_path, mask_path in zip(hsi_files, mask_files):

            basename = os.path.basename(hsi_path)

            match = re.match(
                r"patch_r(\d+)_c(\d+)_hsi\.npy",
                basename,
            )

            if match:
                row_start = int(match.group(1))
                col_start = int(match.group(2))
            else:
                raise ValueError(
                    f"Could not parse spatial coordinates from filename:\n"
                    f"{basename}"
                )

            expected_mask = hsi_path.replace(
                "_hsi.npy",
                "_mask.npy"
            )

            if os.path.abspath(expected_mask) != os.path.abspath(mask_path):
                raise RuntimeError(
                    f"HSI/mask pairing mismatch:\n"
                    f"HSI : {hsi_path}\n"
                    f"Mask: {mask_path}"
                )

            self.all_patches.append(
                {
                    "hsi_path": hsi_path,
                    "mask_path": mask_path,
                    "row_start": row_start,
                    "col_start": col_start,
                }
            )

        # Sort geographically.
        self.all_patches.sort(
            key=lambda p: (p["row_start"], p["col_start"])
        )

        n_total = len(self.all_patches)

        # ---------------------------------------------------------------------
        # Validate split ratios
        # ---------------------------------------------------------------------

        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(
                f"split_ratios must sum to 1.0, got {split_ratios}"
            )

        # ---------------------------------------------------------------------
        # Create geographic split
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        # We split by UNIQUE ROW COORDINATES.
        #
        # This means all patches sharing the same row coordinate stay
        # together in the same split.
        #
        # We DO NOT randomly shuffle rows.
        #
        # With the current 50 Flight 017 patches and 28 unique rows:
        #
        #   19 rows -> train -> 37 patches
        #    2 rows -> val   ->  3 patches
        #    7 rows -> test  -> 10 patches
        #
        # This produces geographically contiguous row blocks.
        # ---------------------------------------------------------------------

        self.row_to_split = self._create_spatial_split(
            self.all_patches,
            split_ratios,
        )

        # Store the split-specific patches.
        if split is None:
            self.patches = list(self.all_patches)
        else:
            self.patches = [
                p
                for p in self.all_patches
                if self.row_to_split[p["row_start"]] == split
            ]

        if len(self.patches) == 0:
            raise RuntimeError(
                f"No patches found for split='{split}'."
            )

        # ---------------------------------------------------------------------
        # Dataset dimensions
        # ---------------------------------------------------------------------

        sample_hsi = np.load(
            self.patches[0]["hsi_path"],
            mmap_mode="r",
        )

        if sample_hsi.ndim != 3:
            raise ValueError(
                f"Expected HSI shape (bands, H, W), "
                f"got {sample_hsi.shape}"
            )

        self.num_bands = sample_hsi.shape[0]
        self.patch_h = sample_hsi.shape[1]
        self.patch_w = sample_hsi.shape[2]

        # ---------------------------------------------------------------------
        # Validate mask dimensions
        # ---------------------------------------------------------------------

        sample_mask = np.load(
            self.patches[0]["mask_path"],
            mmap_mode="r",
        )

        if sample_mask.shape != (self.patch_h, self.patch_w):
            raise ValueError(
                f"HSI spatial shape {(self.patch_h, self.patch_w)} "
                f"does not match mask shape {sample_mask.shape}"
            )

        # ---------------------------------------------------------------------
        # Compute normalization statistics
        # ---------------------------------------------------------------------
        #
        # CRITICAL:
        # Statistics are ALWAYS computed from TRAINING patches.
        #
        # Therefore:
        #
        #   train -> train statistics
        #   val   -> train statistics
        #   test  -> train statistics
        #
        # No validation/test information is used to calculate normalization.
        # ---------------------------------------------------------------------

        self.band_mean, self.band_std = self._compute_train_band_stats()

        # ---------------------------------------------------------------------
        # Dataset metadata
        # ---------------------------------------------------------------------

        # Keep the existing repository convention:
        # train_adapter.py uses dataset.num_classes + 1.
        self.num_classes = len(TALLGRASS_CLASSES) - 1

        self.class_names = TALLGRASS_CLASSES

        self.spatial_shape = (
            self.patch_h,
            self.patch_w,
        )

        # ---------------------------------------------------------------------
        # Print useful dataset information
        # ---------------------------------------------------------------------

        print(
            f"[Tallgrass] Loaded {len(self.patches)} patches "
            f"(of {n_total}), "
            f"{self.num_bands} bands, "
            f"split={split}, "
            f"repeat_factor={self.repeat_factor}"
        )

        if split is not None:
            rows_in_split = sorted(
                {
                    p["row_start"]
                    for p in self.patches
                }
            )

            print(
                f"[Tallgrass] {split} row groups: "
                f"{len(rows_in_split)}"
            )

            print(
                f"[Tallgrass] Row range: "
                f"{min(rows_in_split)} -> {max(rows_in_split)}"
            )

    # =========================================================================
    # Spatial split
    # =========================================================================

    @staticmethod
    def _create_spatial_split(
        patches: List[Dict],
        split_ratios: Tuple[float, float, float],
    ) -> Dict[int, str]:
        """
        Create a deterministic geographic split using row coordinates.

        All patches with the same row_start belong to the same split.

        Rows are sorted spatially and divided into contiguous blocks.

        Returns:
            Dictionary:
                row_start -> "train" / "val" / "test"
        """

        unique_rows = sorted(
            {
                p["row_start"]
                for p in patches
            }
        )

        n_rows = len(unique_rows)

        train_ratio, val_ratio, test_ratio = split_ratios

        # Initial row counts.
        n_train = max(
            1,
            int(n_rows * train_ratio)
        )

        n_val = max(
            1,
            int(n_rows * val_ratio)
        )

        # Make sure all rows are assigned.
        if n_train + n_val >= n_rows:
            n_val = max(
                1,
                n_rows - n_train - 1
            )

        n_test = n_rows - n_train - n_val

        if n_test < 1:
            raise RuntimeError(
                f"Could not create non-empty spatial split "
                f"from {n_rows} row groups."
            )

        train_rows = unique_rows[:n_train]

        val_rows = unique_rows[
            n_train:n_train + n_val
        ]

        test_rows = unique_rows[
            n_train + n_val:
        ]

        row_to_split = {}

        for row in train_rows:
            row_to_split[row] = "train"

        for row in val_rows:
            row_to_split[row] = "val"

        for row in test_rows:
            row_to_split[row] = "test"

        print("\n[Tallgrass] Geographic split")
        print("----------------------------------------")
        print(
            f"Unique row groups : {n_rows}"
        )
        print(
            f"Train rows        : {len(train_rows)}"
        )
        print(
            f"Validation rows   : {len(val_rows)}"
        )
        print(
            f"Test rows         : {len(test_rows)}"
        )

        print(
            f"\nTrain row range: "
            f"{train_rows[0]} -> {train_rows[-1]}"
        )

        print(
            f"Val row range  : "
            f"{val_rows[0]} -> {val_rows[-1]}"
        )

        print(
            f"Test row range : "
            f"{test_rows[0]} -> {test_rows[-1]}"
        )

        return row_to_split

    # =========================================================================
    # Normalization statistics
    # =========================================================================

    def _compute_train_band_stats(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-band mean/std using TRAINING PATCHES ONLY.

        This prevents validation/test information from entering the
        preprocessing pipeline.

        The calculation is performed incrementally so that all 50 patches
        do not need to be loaded into RAM simultaneously.
        """

        train_patches = [
            p
            for p in self.all_patches
            if self.row_to_split[p["row_start"]] == "train"
        ]

        if len(train_patches) == 0:
            raise RuntimeError(
                "No training patches available for normalization statistics."
            )

        print(
            f"[Tallgrass] Computing normalization statistics "
            f"from {len(train_patches)} training patches..."
        )

        band_sum = np.zeros(
            self.num_bands,
            dtype=np.float64,
        )

        band_sq_sum = np.zeros(
            self.num_bands,
            dtype=np.float64,
        )

        pixel_count = np.zeros(
            self.num_bands,
            dtype=np.float64,
        )

        for patch_idx, patch_info in enumerate(train_patches):

            hsi = np.load(
                patch_info["hsi_path"]
            ).astype(np.float32)

            # Raw DN -> reflectance.
            hsi /= REFLECTANCE_SCALE

            # Valid pixels are below the explicit NoData threshold.
            valid_mask = (
                hsi < (ENVI_NODATA / REFLECTANCE_SCALE)
            )

            for b in range(self.num_bands):

                valid = hsi[b][valid_mask[b]]

                if valid.size == 0:
                    continue

                band_sum[b] += np.sum(
                    valid,
                    dtype=np.float64,
                )

                band_sq_sum[b] += np.sum(
                    valid.astype(np.float64) ** 2
                )

                pixel_count[b] += valid.size

        mean = (
            band_sum
            / np.maximum(pixel_count, 1.0)
        ).astype(np.float32)

        variance = (
            band_sq_sum
            / np.maximum(pixel_count, 1.0)
            - mean.astype(np.float64) ** 2
        )

        variance = np.maximum(
            variance,
            0.0,
        )

        std = np.sqrt(
            variance
        ).astype(np.float32)

        # Avoid division by zero.
        std[std < 1e-8] = 1.0

        print(
            "[Tallgrass] Normalization statistics ready."
        )

        return mean, std

    # =========================================================================
    # Dataset interface
    # =========================================================================

    def __len__(self) -> int:
        return len(self.patches) * self.repeat_factor

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:

        # Repeat patches without duplicating them in RAM.
        real_idx = idx % len(self.patches)

        patch_info = self.patches[real_idx]

        # ---------------------------------------------------------------------
        # Load HSI
        # ---------------------------------------------------------------------

        hsi = np.load(
            patch_info["hsi_path"]
        ).astype(np.float32)

        # Expected:
        # (235, 128, 128)

        if hsi.shape != (
            self.num_bands,
            self.patch_h,
            self.patch_w,
        ):
            raise ValueError(
                f"Unexpected HSI shape: {hsi.shape}"
            )

        # Convert raw DN to reflectance.
        hsi /= REFLECTANCE_SCALE

        # ---------------------------------------------------------------------
        # Handle NoData
        # ---------------------------------------------------------------------

        nodata_mask = (
            hsi >= (ENVI_NODATA / REFLECTANCE_SCALE)
        )

        hsi[nodata_mask] = 0.0

        # ---------------------------------------------------------------------
        # Load mask
        # ---------------------------------------------------------------------

        mask = np.load(
            patch_info["mask_path"]
        ).astype(np.int64)

        # Expected:
        # 0, 1, 255

        unique_values = np.unique(mask)

        invalid_values = [
            v
            for v in unique_values
            if v not in (
                MASK_BACKGROUND,
                MASK_SERICEA,
                MASK_IGNORE,
            )
        ]

        if invalid_values:
            raise ValueError(
                f"Unexpected mask values {invalid_values} "
                f"in {patch_info['mask_path']}"
            )

        # ---------------------------------------------------------------------
        # Label mapping
        # ---------------------------------------------------------------------
        #
        # 0   -> 0  Background
        # 1   -> 1  Sericea
        # 255 -> -1 Ignore
        #
        # IMPORTANT:
        # Background remains a valid class.
        # ---------------------------------------------------------------------

        labels = mask.copy()

        labels[
            mask == MASK_IGNORE
        ] = -1

        # ---------------------------------------------------------------------
        # Convert HSI:
        #
        # (B, H, W) -> (H, W, B)
        #
        # because HSITransform expects HWC.
        # ---------------------------------------------------------------------

        hsi_hwb = hsi.transpose(
            1, 2, 0
        )

        # ---------------------------------------------------------------------
        # Apply transforms
        # ---------------------------------------------------------------------

        if self.transform is not None:

            data_tensor, labels_tensor = self.transform(
                hsi_hwb,
                labels,
            )

        else:

            data_tensor = (
                torch.from_numpy(
                    hsi_hwb.copy()
                )
                .float()
                .permute(2, 0, 1)
            )

            labels_tensor = torch.from_numpy(
                labels.copy()
            ).long()

        return {
            "data": data_tensor,
            "labels": labels_tensor,
            "position": torch.tensor(
                [
                    patch_info["row_start"],
                    patch_info["col_start"],
                ],
                dtype=torch.long,
            ),
        }

    # =========================================================================
    # Class weights
    # =========================================================================

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights using TRAINING PATCHES ONLY.

        Ignore pixels (255) are excluded.

        Returns:
            Tensor of shape (2,):
                [background_weight, sericea_weight]

        The weights are normalized so their mean is approximately 1.
        """

        total_bg = 0
        total_sericea = 0

        train_patches = [
            p
            for p in self.all_patches
            if self.row_to_split[p["row_start"]] == "train"
        ]

        for patch_info in train_patches:

            mask = np.load(
                patch_info["mask_path"]
            )

            total_bg += int(
                np.sum(mask == MASK_BACKGROUND)
            )

            total_sericea += int(
                np.sum(mask == MASK_SERICEA)
            )

        counts = np.array(
            [
                max(total_bg, 1),
                max(total_sericea, 1),
            ],
            dtype=np.float32,
        )

        # Inverse frequency.
        weights = 1.0 / counts

        # Normalize so mean weight = 1.
        weights = (
            weights
            / weights.mean()
        )

        print(
            "\n[Tallgrass] Training class statistics"
        )
        print("----------------------------------------")
        print(
            f"Background pixels : {total_bg:,}"
        )
        print(
            f"Sericea pixels    : {total_sericea:,}"
        )
        print(
            f"Background weight : {weights[0]:.4f}"
        )
        print(
            f"Sericea weight    : {weights[1]:.4f}"
        )

        return torch.from_numpy(
            weights
        ).float()

    # =========================================================================
    # Active-learning composite
    # =========================================================================

    def get_al_composite(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a horizontal composite of the current split.

        This preserves the existing active-learning interface.

        Returns:

            full_data:
                (H, total_width, B)

            full_labels:
                (H, total_width)

            Label convention:
                0  = background
                1  = Sericea
                -1 = ignore
        """

        n = len(self.patches)

        total_w = (
            self.patch_w * n
        )

        full_data = np.zeros(
            (
                self.patch_h,
                total_w,
                self.num_bands,
            ),
            dtype=np.float32,
        )

        full_labels = np.full(
            (
                self.patch_h,
                total_w,
            ),
            -1,
            dtype=np.int64,
        )

        for i, patch_info in enumerate(
            self.patches
        ):

            hsi = np.load(
                patch_info["hsi_path"]
            ).astype(np.float32)

            hsi /= REFLECTANCE_SCALE

            # NoData -> zero.
            nodata = (
                hsi >= (
                    ENVI_NODATA
                    / REFLECTANCE_SCALE
                )
            )

            hsi[nodata] = 0.0

            mask = np.load(
                patch_info["mask_path"]
            ).astype(np.int64)

            # 0 -> 0
            # 1 -> 1
            # 255 -> -1
            labels = mask.copy()

            labels[
                mask == MASK_IGNORE
            ] = -1

            col_start = (
                i * self.patch_w
            )

            full_data[
                :,
                col_start:col_start + self.patch_w,
                :,
            ] = hsi.transpose(
                1, 2, 0
            )

            full_labels[
                :,
                col_start:col_start + self.patch_w,
            ] = labels

        return full_data, full_labels


    def get_al_split_masks(self) -> Dict[str, np.ndarray]:
        """
        Create geographic train/validation/test masks aligned with
        the horizontal active-learning composite.

        The composite ordering is identical to get_al_composite():
        patches are placed left-to-right in self.patches order.

        Returns:
            Dictionary containing:

                "train_pool_mask":
                    Boolean mask for pixels belonging to geographic
                    training patches.

                "val_mask":
                    Boolean mask for pixels belonging to geographic
                    validation patches.

                "test_mask":
                    Boolean mask for pixels belonging to geographic
                    test patches.

        IMPORTANT:
            These masks are based on patch-level geographic assignment,
            not random pixel sampling.
        """

        if self.split is not None:
            raise ValueError(
                "get_al_split_masks() must be called on "
                "TallgrassDataset(split=None)."
            )

        n = len(self.patches)

        height = self.patch_h
        total_width = self.patch_w * n

        train_pool_mask = np.zeros(
            (height, total_width),
            dtype=bool,
        )

        val_mask = np.zeros(
            (height, total_width),
            dtype=bool,
        )

        test_mask = np.zeros(
            (height, total_width),
            dtype=bool,
        )

        for i, patch_info in enumerate(self.patches):

            col_start = i * self.patch_w
            col_end = col_start + self.patch_w

            split_name = self.row_to_split[
                patch_info["row_start"]
            ]

            if split_name == "train":
                train_pool_mask[
                    :,
                    col_start:col_end,
                ] = True

            elif split_name == "val":
                val_mask[
                    :,
                    col_start:col_end,
                ] = True

            elif split_name == "test":
                test_mask[
                    :,
                    col_start:col_end,
                ] = True

            else:
                raise RuntimeError(
                    f"Unknown split '{split_name}' "
                    f"for row {patch_info['row_start']}"
                )

        return {
            "train_pool_mask": train_pool_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    def get_full_image(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compatibility wrapper.

        Returns the current split as a composite.
        """

        return self.get_al_composite()

    # =========================================================================
    # Dataset diagnostics
    # =========================================================================

    def get_split_summary(self) -> Dict[str, Dict]:
        """
        Return information about the geographic split.

        Useful for verifying that train/val/test are spatially separated.
        """

        summary = {}

        for split_name in (
            "train",
            "val",
            "test",
        ):

            patches = [
                p
                for p in self.all_patches
                if self.row_to_split[
                    p["row_start"]
                ] == split_name
            ]

            rows = sorted(
                {
                    p["row_start"]
                    for p in patches
                }
            )

            summary[split_name] = {
                "num_patches": len(patches),
                "num_rows": len(rows),
                "rows": rows,
                "min_row": min(rows) if rows else None,
                "max_row": max(rows) if rows else None,
            }

        return summary

    def print_split_summary(self) -> None:
        """
        Print train/validation/test geographic split information.
        """

        summary = self.get_split_summary()

        print("\n" + "=" * 70)
        print("TALLGRASS GEOGRAPHIC SPLIT")
        print("=" * 70)

        for split_name in (
            "train",
            "val",
            "test",
        ):

            info = summary[split_name]

            print(
                f"\n{split_name.upper()}"
            )

            print(
                f"  Patches    : {info['num_patches']}"
            )

            print(
                f"  Row groups : {info['num_rows']}"
            )

            print(
                f"  Row range  : "
                f"{info['min_row']} -> "
                f"{info['max_row']}"
            )

            print(
                f"  Rows       : "
                f"{info['rows']}"
            )

        print("=" * 70)

    # =========================================================================
    # Wavelengths
    # =========================================================================

    @staticmethod
    def get_wavelengths(
        header_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return the usable wavelengths for Flight 017.

        If an ENVI header is supplied, wavelengths are parsed from it.
        Otherwise a fallback wavelength array is returned.
        """

        usable_ranges = [
            (431.10, 1299.36),
            (1487.71, 1775.03),
            (1998.23, 2353.76),
        ]

        if (
            header_path is not None
            and os.path.exists(header_path)
        ):

            with open(
                header_path,
                "r",
            ) as f:

                content = f.read()

            match = re.search(
                r"wavelength\s*=\s*\{(.*?)\}",
                content,
                re.DOTALL,
            )

            if match:

                all_wls = [
                    float(x.strip())
                    for x in match.group(1).split(",")
                    if x.strip()
                ]

                filtered = [
                    wl
                    for wl in all_wls
                    if any(
                        lo <= wl <= hi
                        for lo, hi in usable_ranges
                    )
                ]

                return np.array(
                    filtered,
                    dtype=np.float32,
                )

        # Fallback.
        return np.linspace(
            433.67,
            2352.67,
            235,
            dtype=np.float32,
        )