"""
active_learning/uncertainty.py — Uncertainty estimation via MC-Dropout.

Implements two uncertainty measures:
  1. Shannon Entropy (total uncertainty): H(ŷ) = -Σ p_c · log(p_c)
     High when the model is uncertain about the class of a pixel.

  2. BALD (Bayesian Active Learning by Disagreement):
     I[y; θ | x] = H(ŷ) - E_θ[H(ŷ|θ)]
     = Total uncertainty - Expected aleatoric uncertainty
     = Epistemic uncertainty (model uncertainty, reducible with more data)

     High BALD means the model parameters disagree about the prediction,
     which indicates the model would benefit most from a label at that pixel.

MC-Dropout procedure:
  - Run T forward passes with dropout enabled (even during inference)
  - Each pass samples a different dropout mask → different predictions
  - Average the T predictions → predictive distribution (for entropy)
  - Variance across the T predictions → model uncertainty (for BALD)

This is a computationally cheap approximation to full Bayesian inference.
T=10 is standard — more passes give diminishing returns on uncertainty quality.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm


def mc_dropout_inference(
    model: torch.nn.Module,
    hsi: torch.Tensor,
    pca_rgb: torch.Tensor = None,
    num_passes: int = 10,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Run MC-Dropout inference to estimate per-pixel uncertainty.

    Args:
        model: The segmentation model with dropout layers. Must support
            enable_mc_dropout() method to keep dropout active during eval.
        hsi: Hyperspectral input, shape (1, B, H, W). Batch size must be 1
            for memory efficiency during uncertainty estimation.
        pca_rgb: PCA-reduced input for residual pathway, shape (1, 3, H, W).
        num_passes: T — number of stochastic forward passes.
        device: Device to run on.

    Returns:
        Dict with:
          - "mean_probs": (C, H, W) — averaged predictive probabilities
          - "entropy": (H, W) — Shannon entropy (total uncertainty)
          - "bald": (H, W) — BALD score (epistemic uncertainty)
          - "predicted_class": (H, W) — argmax of mean probabilities
          - "all_probs": (T, C, H, W) — all T probability maps (for analysis)
    """
    model.enable_mc_dropout()  # Dropout ON, BatchNorm in eval mode
    hsi = hsi.to(device)
    if pca_rgb is not None:
        pca_rgb = pca_rgb.to(device)

    all_probs = []

    with torch.no_grad():
        for t in range(num_passes):
            output = model(hsi, pca_rgb)
            logits = output["logits"]  # (1, C, H, W)

            # Softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (1, C, H, W)
            all_probs.append(probs.squeeze(0))  # (C, H, W)

    # Stack all passes: (T, C, H, W)
    all_probs = torch.stack(all_probs, dim=0)

    # === Predictive distribution (mean over T passes) ===
    mean_probs = all_probs.mean(dim=0)  # (C, H, W)

    # === Shannon Entropy: H(ŷ) = -Σ p_c · log(p_c) ===
    # Clamp probs to avoid log(0)
    entropy = -torch.sum(
        mean_probs * torch.log(mean_probs.clamp(min=1e-8)), dim=0
    )  # (H, W)

    # === BALD: I[y; θ | x] = H(ŷ) - E_θ[H(ŷ|θ)] ===
    # E_θ[H(ŷ|θ)] is the mean entropy of individual passes
    # (this captures aleatoric/irreducible uncertainty)
    per_pass_entropy = -torch.sum(
        all_probs * torch.log(all_probs.clamp(min=1e-8)), dim=1
    )  # (T, H, W)
    expected_entropy = per_pass_entropy.mean(dim=0)  # (H, W)

    # BALD = total entropy - expected entropy
    # High BALD = high disagreement between passes = high epistemic uncertainty
    bald = entropy - expected_entropy  # (H, W)

    # Clamp BALD to non-negative (numerical errors can cause tiny negatives)
    bald = bald.clamp(min=0)

    # === Predicted class ===
    predicted_class = mean_probs.argmax(dim=0)  # (H, W)

    return {
        "mean_probs": mean_probs.cpu(),        # (C, H, W)
        "entropy": entropy.cpu(),               # (H, W)
        "bald": bald.cpu(),                     # (H, W)
        "predicted_class": predicted_class.cpu(), # (H, W)
        "all_probs": all_probs.cpu(),           # (T, C, H, W)
    }


def compute_pixel_uncertainties(
    uncertainty_map: torch.Tensor,
    vegetation_mask: torch.Tensor = None,
    labeled_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank pixels by uncertainty, optionally filtering by vegetation and
    excluding already-labeled pixels.

    Args:
        uncertainty_map: Per-pixel uncertainty scores, shape (H, W).
            Can be either entropy or BALD scores.
        vegetation_mask: Boolean mask, shape (H, W). True = vegetation pixel.
            If provided, only vegetation pixels are ranked (non-vegetation
            are assigned uncertainty = -inf so they're never selected).
        labeled_mask: Boolean mask, shape (H, W). True = already labeled.
            These pixels are excluded from selection.

    Returns:
        sorted_indices: (N, 2) tensor of (row, col) pixel coordinates,
            sorted by descending uncertainty (most uncertain first).
        sorted_scores: (N,) tensor of uncertainty scores, sorted descending.
    """
    H, W = uncertainty_map.shape
    scores = uncertainty_map.clone()

    # Mask out non-vegetation pixels
    if vegetation_mask is not None:
        scores[~vegetation_mask] = -float("inf")

    # Mask out already-labeled pixels
    if labeled_mask is not None:
        scores[labeled_mask] = -float("inf")

    # Flatten, sort by descending uncertainty, get top pixel coordinates
    flat_scores = scores.reshape(-1)
    sorted_idx = torch.argsort(flat_scores, descending=True)

    # Convert flat indices to (row, col) coordinates
    rows = sorted_idx // W
    cols = sorted_idx % W
    sorted_coords = torch.stack([rows, cols], dim=1)  # (H*W, 2)

    # Filter out masked pixels (those with -inf score)
    valid_mask = flat_scores[sorted_idx] > -float("inf")
    sorted_coords = sorted_coords[valid_mask]
    sorted_scores = flat_scores[sorted_idx][valid_mask]

    return sorted_coords, sorted_scores
