"""
models/losses.py — Composite Focal-Dice loss for class-imbalanced segmentation.

Why this combination:
  - Focal Loss addresses class imbalance by down-weighting easy (well-classified)
    pixels and focusing on hard examples. Critical because invasive species
    (or minority vegetation classes) occupy tiny fractions of images.
  - Dice Loss directly optimizes the IoU-like metric, which is what we report.
    It's insensitive to class imbalance by design (normalizes by class area).
  - Together, they complement each other: Focal provides pixel-level gradients
    for all classes, Dice provides region-level gradients for coherent masks.

Both losses are computed per-class and averaged, ensuring that small classes
contribute equally to the gradient signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where p_t is the predicted probability for the true class.

    When γ=0, this reduces to standard cross-entropy.
    When γ>0, well-classified pixels (p_t close to 1) are down-weighted,
    focusing the loss on hard-to-classify pixels near decision boundaries.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
                γ=2 is standard in the literature (from Lin et al., 2017).
            alpha: Per-class weights, shape (num_classes,). If None, all classes
                are weighted equally. Typically set to inverse class frequency.
            ignore_index: Label value to ignore (background/unlabeled pixels).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw predictions, shape (batch, num_classes, H, W).
            targets: Ground truth class indices, shape (batch, H, W).
                     Pixels with value == ignore_index are excluded.

        Returns:
            Scalar loss value.
        """
        num_classes = logits.shape[1]

        # Standard cross-entropy per pixel (unreduced)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Get predicted probability for the true class
        # Softmax over class dimension, then gather the true class probability
        probs = F.softmax(logits, dim=1)  # (batch, C, H, W)

        # Create a mask for valid (non-ignored) pixels
        valid_mask = (targets != self.ignore_index)

        # For gathering, we need targets with no -1 values (clamp to 0 temporarily)
        safe_targets = targets.clamp(min=0)

        # Gather the probability of the true class at each pixel
        # (batch, C, H, W) → gather along C → (batch, 1, H, W) → squeeze
        p_t = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)  # (batch, H, W)

        # Focal modulation: (1 - p_t)^γ
        # When p_t is high (confident correct prediction), this is near 0
        # When p_t is low (uncertain/wrong), this is near 1
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply focal weight to cross-entropy
        focal_loss = focal_weight * ce_loss

        # Average over valid pixels only
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return focal_loss.sum() * 0.0  # No valid pixels — return zero


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.

    Dice = 2 · |P ∩ G| / (|P| + |G|)
    DiceLoss = 1 - Dice

    Computed per-class and averaged. Directly relates to the IoU metric
    we use for evaluation, so optimizing Dice approximately optimizes IoU.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        """
        Args:
            smooth: Smoothing term (ε) to prevent division by zero and
                stabilize gradients when a class has very few pixels.
            ignore_index: Label value to ignore.
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-class Dice loss, averaged over classes.

        Args:
            logits: (batch, num_classes, H, W)
            targets: (batch, H, W) with class indices

        Returns:
            Scalar loss value.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (batch, C, H, W)

        # Create valid pixel mask
        valid_mask = (targets != self.ignore_index)  # (batch, H, W)

        # One-hot encode targets: (batch, H, W) → (batch, C, H, W)
        # Set ignored pixels to class 0 temporarily (they'll be masked out)
        safe_targets = targets.clamp(min=0)
        targets_one_hot = F.one_hot(safe_targets, num_classes)  # (batch, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (batch, C, H, W)

        # Zero out ignored pixels in both predictions and targets
        valid_mask_expanded = valid_mask.unsqueeze(1).float()  # (batch, 1, H, W)
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        # Compute Dice per class
        # Flatten spatial dims for summation
        probs_flat = probs.reshape(probs.shape[0], num_classes, -1)       # (batch, C, N)
        targets_flat = targets_one_hot.reshape(targets_one_hot.shape[0], num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)  # (batch, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (batch, C)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (batch, C)

        # Average over classes and batch
        # Note: classes with no pixels in this batch will have dice ≈ smooth/(smooth)
        # ≈ 1.0, which is correct (no penalty for absent classes)
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalDiceLoss(nn.Module):
    """
    Composite Focal + Dice loss.

    L_total = λ₁ · L_focal + λ₂ · L_dice

    This is the main loss function for the project.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        ignore_index: int = -1,
    ):
        """
        Args:
            focal_gamma: Focal loss focusing parameter.
            focal_alpha: Per-class weights for focal loss.
            focal_weight: λ₁ — weight for focal loss term.
            dice_weight: λ₂ — weight for dice loss term.
            dice_smooth: Smoothing term for dice loss.
            ignore_index: Label value to ignore.
        """
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """
        Compute composite loss and return individual components for logging.

        Args:
            logits: (batch, num_classes, H, W)
            targets: (batch, H, W)

        Returns:
            Dict with keys:
              - "total": combined loss (for backprop)
              - "focal": focal component (for logging)
              - "dice": dice component (for logging)
        """
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        total = self.focal_weight * focal_loss + self.dice_weight * dice_loss

        return {
            "total": total,
            "focal": focal_loss.detach(),
            "dice": dice_loss.detach(),
        }
