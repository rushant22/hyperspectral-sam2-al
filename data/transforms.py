"""
data/transforms.py — Data augmentation and preprocessing transforms for HSI.

Unlike RGB augmentation (which can use off-the-shelf torchvision transforms),
hyperspectral data requires special care:
  - Spectral augmentation must preserve physically meaningful band relationships
    (e.g., you can't randomly permute bands like you can color-jitter RGB).
  - Spatial augmentations (flip, rotate) are safe and standard.
  - We normalize per-band, not per-channel like ImageNet normalization.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import torch.nn.functional as F


class HSITransform:
    """
    Composable transform for hyperspectral image patches.

    Applies spatial augmentations (safe for HSI) and normalization.
    All transforms operate on (data, label) pairs to ensure spatial
    augmentations are applied consistently to both.
    """

    def __init__(
        self,
        target_size: int = 256,
        normalize: bool = True,
        spatial_augment: bool = True,
        spectral_augment: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            target_size: Resize spatial dims to this size (for SAM2 input).
            normalize: Whether to apply per-band normalization.
            spatial_augment: Random flips and 90° rotations (training only).
            spectral_augment: Mild spectral noise (training only, experimental).
            mean: Pre-computed per-band means for normalization.
            std: Pre-computed per-band stds for normalization.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.spatial_augment = spatial_augment
        self.spectral_augment = spectral_augment
        self.mean = mean
        self.std = std

    def __call__(
        self, data: np.ndarray, label: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform an HSI patch and its corresponding label mask.

        Args:
            data: HSI patch, shape (H, W, B), float32.
            label: Label mask, shape (H, W), int.

        Returns:
            data_tensor: Shape (B, target_size, target_size), float32 torch tensor.
            label_tensor: Shape (target_size, target_size), long torch tensor.
        """
        # --- Per-band normalization ---
        # Done in numpy before conversion to torch for efficiency
        if self.normalize and self.mean is not None and self.std is not None:
            data = (data - self.mean) / self.std

        # --- Convert to torch tensors ---
        # Data: (H, W, B) → (B, H, W) to match PyTorch's (C, H, W) convention
        data_tensor = torch.from_numpy(data.copy()).float().permute(2, 0, 1)
        label_tensor = torch.from_numpy(label.copy()).long()

        # --- Spatial augmentations ---
        # These are safe for HSI because they only affect spatial dims, not
        # the spectral dimension. The same transform must be applied to both
        # the data and the label to maintain spatial alignment.
        if self.spatial_augment:
            # Random horizontal flip (50% chance)
            if torch.rand(1).item() > 0.5:
                data_tensor = torch.flip(data_tensor, dims=[2])  # flip W
                label_tensor = torch.flip(label_tensor, dims=[1])

            # Random vertical flip (50% chance)
            if torch.rand(1).item() > 0.5:
                data_tensor = torch.flip(data_tensor, dims=[1])  # flip H
                label_tensor = torch.flip(label_tensor, dims=[0])

            # Random 90° rotation (0°, 90°, 180°, or 270°)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                data_tensor = torch.rot90(data_tensor, k, dims=[1, 2])
                label_tensor = torch.rot90(label_tensor, k, dims=[0, 1])

        # --- Spectral augmentation (experimental, off by default) ---
        # Adds small Gaussian noise to each band independently. This simulates
        # sensor noise variation and can improve generalization, but must be
        # very mild to avoid destroying spectral signatures.
        if self.spectral_augment:
            noise_std = 0.01  # 1% noise — very conservative
            noise = torch.randn_like(data_tensor) * noise_std
            data_tensor = data_tensor + noise

        # --- Resize to target size ---
        # SAM2 expects inputs larger than typical HSI patches. We bilinearly
        # interpolate the spatial dims. This doesn't add information but
        # ensures the Hiera encoder's positional embeddings work correctly.
        if data_tensor.shape[1] != self.target_size or data_tensor.shape[2] != self.target_size:
            # Data: bilinear interpolation (smooth for continuous reflectance)
            data_tensor = F.interpolate(
                data_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # Labels: nearest-neighbor interpolation (preserves class indices)
            label_tensor = (
                F.interpolate(
                    label_tensor.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.target_size, self.target_size),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        return data_tensor, label_tensor


def get_train_transform(
    target_size: int = 256,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> HSITransform:
    """Create a training transform with augmentations enabled."""
    return HSITransform(
        target_size=target_size,
        normalize=True,
        spatial_augment=True,
        spectral_augment=False,  # Conservative default — enable after baseline works
        mean=mean,
        std=std,
    )


def get_eval_transform(
    target_size: int = 256,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> HSITransform:
    """Create an evaluation/test transform (no augmentations)."""
    return HSITransform(
        target_size=target_size,
        normalize=True,
        spatial_augment=False,
        spectral_augment=False,
        mean=mean,
        std=std,
    )
