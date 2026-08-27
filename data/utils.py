"""
data/utils.py — Utility functions for hyperspectral data processing.

Provides:
  - Per-band normalization (zero-mean, unit-variance)
  - PCA dimensionality reduction (for baseline model)
  - NDVI computation (for vegetation masking in active learning)
  - Band selection helpers
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple


def normalize_per_band(
    data: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each spectral band to zero mean and unit variance.

    This is the standard preprocessing for HSI data — each band has very
    different reflectance scales (e.g., visible vs. SWIR), so per-band
    normalization prevents high-magnitude bands from dominating learned features.

    Args:
        data: HSI cube, shape (H, W, B) where B = number of bands.
        mean: Pre-computed per-band means, shape (B,). If None, computed from data.
        std: Pre-computed per-band stds, shape (B,). If None, computed from data.

    Returns:
        normalized_data: Same shape as input, normalized.
        mean: Per-band means (save these for test-time normalization).
        std: Per-band standard deviations.
    """
    H, W, B = data.shape

    # Reshape to (num_pixels, num_bands) for easy stats computation
    flat = data.reshape(-1, B).astype(np.float32)

    if mean is None:
        mean = flat.mean(axis=0)
    if std is None:
        std = flat.std(axis=0)
        # Prevent division by zero for constant bands (e.g., water absorption
        # bands that were zeroed out but not removed)
        std[std < 1e-8] = 1.0

    normalized = (flat - mean) / std
    return normalized.reshape(H, W, B), mean, std


def apply_pca(
    data: np.ndarray,
    n_components: int = 3,
    pca_model: Optional[PCA] = None,
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce spectral dimensionality via PCA.

    Used by the baseline model to project B bands → 3 channels for SAM2's
    standard RGB input pathway. Also useful for visualization (false-color
    composites from top-3 principal components).

    Why PCA over random band selection:
      PCA captures maximum variance across the spectral range, giving a more
      informative 3-channel representation than manually picking e.g., bands
      at 660nm/850nm/550nm. It also decorrelates the inputs, which helps
      the frozen SAM2 stem (designed for decorrelated RGB inputs).

    Args:
        data: HSI cube, shape (H, W, B).
        n_components: Number of PCA components to keep.
        pca_model: Pre-fitted PCA model. If None, fits a new one on this data.

    Returns:
        reduced: Shape (H, W, n_components), PCA-transformed data.
        pca_model: Fitted PCA model (save for test-time transformation).
    """
    H, W, B = data.shape
    flat = data.reshape(-1, B).astype(np.float32)

    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        pca_model.fit(flat)

    reduced = pca_model.transform(flat)
    return reduced.reshape(H, W, n_components), pca_model


def compute_ndvi(
    data: np.ndarray,
    red_band_idx: int,
    nir_band_idx: int,
) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Used in the active learning loop to mask out non-vegetation pixels before
    uncertainty ranking, so the AL budget isn't wasted querying roads, buildings,
    water, etc. that are irrelevant to the vegetation segmentation task.

    NDVI ranges:
      - < 0.0: water, shadows
      - 0.0–0.2: bare soil, rock, impervious surfaces
      - 0.2–0.5: sparse vegetation, grassland
      - 0.5–1.0: dense vegetation, forest canopy

    Args:
        data: HSI cube, shape (H, W, B). Must be surface reflectance (not raw DN).
        red_band_idx: Index of the red band (~660nm) in the spectral dimension.
        nir_band_idx: Index of the NIR band (~850nm) in the spectral dimension.

    Returns:
        ndvi: Shape (H, W), values in [-1, 1].
    """
    red = data[:, :, red_band_idx].astype(np.float32)
    nir = data[:, :, nir_band_idx].astype(np.float32)

    # Epsilon prevents division by zero in shadowed pixels where both bands → 0
    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)

    return ndvi


def get_ndvi_band_indices(dataset_name: str) -> Tuple[int, int]:
    """
    Return (red_band_idx, nir_band_idx) for known datasets.

    These indices are dataset-specific because different sensors sample
    different wavelengths at different band positions.

    Args:
        dataset_name: One of "indian_pines", "pavia", "toulouse".

    Returns:
        (red_idx, nir_idx): Zero-indexed band positions.
    """
    # Band centers are approximate — close enough for NDVI thresholding.
    # Exact wavelength-to-index mapping is in each dataset's metadata.
    indices = {
        # AVIRIS: 10nm sampling, 400-2500nm, 200 bands after water removal
        # Red ≈ 660nm → band ~29, NIR ≈ 850nm → band ~49
        "indian_pines": (29, 49),
        # ROSIS: 430-860nm, 103 bands
        # Red ≈ 660nm → band ~53, NIR ≈ 850nm → band ~97
        "pavia": (53, 97),
        # AisaFENIX: 400-2500nm, 310 bands
        # Red ≈ 660nm → band ~40, NIR ≈ 850nm → band ~70
        "toulouse": (40, 70),
    }

    if dataset_name not in indices:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(indices.keys())}. "
            f"For custom datasets, specify --ndvi-red-band and --ndvi-nir-band manually."
        )

    return indices[dataset_name]


def extract_patches(
    data: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 128,
    overlap: float = 0.5,
) -> list:
    """
    Tile a large HSI image into overlapping patches.

    Pavia University (610×340) and Toulouse (large) images are too big to
    process as single inputs to SAM2. We tile them into manageable patches.
    Indian Pines (145×145) is small enough to use as a single patch.

    The overlap ensures that border pixels appear in multiple patches,
    giving the model context for edge regions. During inference, predictions
    from overlapping regions are averaged.

    Args:
        data: HSI cube, shape (H, W, B).
        labels: Ground truth, shape (H, W).
        patch_size: Spatial size of each square patch.
        overlap: Fraction of overlap between adjacent patches (0.0–0.99).

    Returns:
        List of dicts, each containing:
          - "data": patch HSI cube, shape (patch_size, patch_size, B)
          - "labels": patch labels, shape (patch_size, patch_size)
          - "row_start", "col_start": top-left corner in the original image
    """
    H, W, B = data.shape
    stride = int(patch_size * (1.0 - overlap))
    # Ensure stride is at least 1 pixel to avoid infinite loops
    stride = max(stride, 1)

    patches = []

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            patch_data = data[row : row + patch_size, col : col + patch_size, :]
            patch_labels = labels[row : row + patch_size, col : col + patch_size]

            patches.append(
                {
                    "data": patch_data,
                    "labels": patch_labels,
                    "row_start": row,
                    "col_start": col,
                }
            )

    # Handle edge patches — if the image doesn't divide evenly, add patches
    # anchored to the bottom and right edges
    # Bottom edge
    if (H - patch_size) % stride != 0:
        for col in range(0, W - patch_size + 1, stride):
            row = H - patch_size
            patches.append(
                {
                    "data": data[row : row + patch_size, col : col + patch_size, :],
                    "labels": labels[row : row + patch_size, col : col + patch_size],
                    "row_start": row,
                    "col_start": col,
                }
            )
    # Right edge
    if (W - patch_size) % stride != 0:
        for row in range(0, H - patch_size + 1, stride):
            col = W - patch_size
            patches.append(
                {
                    "data": data[row : row + patch_size, col : col + patch_size, :],
                    "labels": labels[row : row + patch_size, col : col + patch_size],
                    "row_start": row,
                    "col_start": col,
                }
            )

    return patches
