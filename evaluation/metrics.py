"""
evaluation/metrics.py — Segmentation metrics: mIoU, per-class IoU, OA, Kappa.

All metrics operate on flattened predicted/ground-truth label arrays,
handling the ignore_index (-1) convention used throughout the codebase.

These functions are called both during training (validation) and final
evaluation. Every number in the paper comes from calling these functions.
"""

import torch
import numpy as np
from typing import Dict, Optional


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Compute the confusion matrix from predicted and target labels.

    Args:
        pred: Predicted class indices, shape (N,).
        target: Ground truth class indices, shape (N,).
        num_classes: Number of classes (C). Classes are indexed 0 to C-1.
            Note: class 0 may represent "background" depending on context.

    Returns:
        cm: Confusion matrix, shape (num_classes, num_classes).
            cm[i][j] = number of pixels with true class i predicted as class j.
    """
    assert pred.shape == target.shape, (
        f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    )

    # Ensure we're working with numpy arrays
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(pred_np)):
        t = int(target_np[i])
        p = int(pred_np[i])
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    return cm


def compute_per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """
    Compute per-class Intersection over Union (IoU).

    IoU for class c = TP_c / (TP_c + FP_c + FN_c)
    where:
      TP_c = correctly predicted as class c
      FP_c = incorrectly predicted as class c (was actually another class)
      FN_c = missed class c (was class c but predicted as something else)

    Args:
        pred: Predicted labels, shape (N,). Already filtered to valid pixels.
        target: Ground truth labels, shape (N,).
        num_classes: Number of classes.

    Returns:
        Dict mapping class_id → IoU value. Classes with no pixels get IoU = NaN.
    """
    cm = compute_confusion_matrix(pred, target, num_classes)

    per_class_iou = {}
    for c in range(num_classes):
        tp = cm[c][c]
        fp = cm[:, c].sum() - tp  # Column sum minus diagonal
        fn = cm[c, :].sum() - tp  # Row sum minus diagonal

        denominator = tp + fp + fn
        if denominator == 0:
            # This class has no pixels in either pred or target
            per_class_iou[c] = float("nan")
        else:
            per_class_iou[c] = float(tp) / float(denominator)

    return per_class_iou


def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_absent: bool = True,
) -> float:
    """
    Compute mean Intersection over Union (mIoU) across classes.

    This is the primary metric for the project.

    Args:
        pred: Predicted labels, shape (N,).
        target: Ground truth labels, shape (N,).
        num_classes: Number of classes.
        ignore_absent: If True, classes with no pixels in the ground truth
            are excluded from the mean. This prevents "free" IoU=NaN→0 from
            dragging down the mean when some classes don't appear in a patch.

    Returns:
        mIoU as a float in [0, 1].
    """
    per_class = compute_per_class_iou(pred, target, num_classes)

    ious = []
    for c, iou in per_class.items():
        if not np.isnan(iou):
            ious.append(iou)
        elif not ignore_absent:
            ious.append(0.0)

    if len(ious) == 0:
        return 0.0

    return float(np.mean(ious))


def compute_overall_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Overall pixel accuracy: fraction of correctly classified pixels.

    OA = correct / total

    Note: OA is misleading for imbalanced datasets (e.g., predicting all
    background gives >90% OA on Indian Pines). We report it for completeness
    but rely on mIoU as the primary metric.
    """
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target

    correct = (pred_np == target_np).sum()
    total = len(pred_np)

    return float(correct) / max(float(total), 1.0)


def compute_kappa(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Cohen's Kappa coefficient.

    κ = (p_o - p_e) / (1 - p_e)

    where p_o = observed agreement (OA), p_e = expected agreement by chance.

    Kappa adjusts for class imbalance by comparing observed accuracy to
    what would be expected by random agreement. Commonly reported alongside
    mIoU in HSI classification papers.

    Interpretation:
      κ < 0:     Worse than random
      κ = 0:     Random agreement
      0 < κ < 1: Better than random
      κ = 1:     Perfect agreement
    """
    cm = compute_confusion_matrix(pred, target, num_classes)
    total = cm.sum()

    if total == 0:
        return 0.0

    p_o = np.diag(cm).sum() / total  # Observed agreement

    # Expected agreement: sum of (row_total * col_total) / total²
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (total * total)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)
