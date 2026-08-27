"""
active_learning/query_strategies.py — Pixel/region selection strategies for AL.

Implements three strategies:
  1. Random: Uniform random selection (baseline for comparison).
  2. Entropy/BALD: Select the K most uncertain pixels (greedy).
  3. BADGE-inspired: Select uncertain pixels with spatial diversity via
     K-means clustering on feature embeddings.

Strategy (3) is important because greedy uncertainty selection tends to
pick spatially clustered pixels (uncertain regions are often contiguous).
This wastes the annotation budget — labeling adjacent pixels provides
redundant information. BADGE-style diversity ensures spatial spread.

Implementation note: True BADGE uses gradient embeddings (gradient of loss
w.r.t. last-layer params per sample), which requires one backward pass per
unlabeled pixel. That's prohibitively expensive for pixel-level AL on large
images. We use feature embeddings (penultimate layer activations) instead,
weighted by uncertainty. This is often called "CoreSet + Uncertainty" and
achieves similar diversity benefits at much lower compute cost.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional


def random_query(
    num_pixels: int,
    total_pixels: int,
    labeled_mask: torch.Tensor = None,
    vegetation_mask: torch.Tensor = None,
    seed: int = None,
) -> torch.Tensor:
    """
    Uniform random pixel selection (baseline strategy).

    Args:
        num_pixels: Number of pixels to select (K).
        total_pixels: Total number of candidate pixels.
        labeled_mask: (H, W) boolean — True = already labeled (excluded).
        vegetation_mask: (H, W) boolean — True = vegetation (included).
        seed: Random seed for reproducibility.

    Returns:
        selected_coords: (K, 2) tensor of (row, col) pixel coordinates.
    """
    rng = np.random.RandomState(seed)

    if labeled_mask is not None and vegetation_mask is not None:
        H, W = labeled_mask.shape
        # Valid pixels: vegetation AND not yet labeled
        valid = vegetation_mask & (~labeled_mask)
        valid_coords = torch.nonzero(valid, as_tuple=False)  # (N_valid, 2)
    elif labeled_mask is not None:
        valid = ~labeled_mask
        valid_coords = torch.nonzero(valid, as_tuple=False)
    elif vegetation_mask is not None:
        valid_coords = torch.nonzero(vegetation_mask, as_tuple=False)
    else:
        # All pixels are candidates — create a grid of coordinates
        H = int(np.sqrt(total_pixels))
        W = total_pixels // H
        rows, cols = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        valid_coords = torch.stack([rows.flatten(), cols.flatten()], dim=1)

    n_valid = len(valid_coords)
    if n_valid == 0:
        return torch.zeros(0, 2, dtype=torch.long)

    # Randomly select K pixels from valid candidates
    k = min(num_pixels, n_valid)
    indices = rng.choice(n_valid, size=k, replace=False)
    selected = valid_coords[indices]

    return selected


def uncertainty_query(
    uncertainty_scores: torch.Tensor,
    num_pixels: int,
    labeled_mask: torch.Tensor = None,
    vegetation_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Greedy uncertainty-based selection (entropy or BALD scores).

    Simply picks the K most uncertain unlabeled vegetation pixels.
    Fast and effective, but tends to cluster spatially.

    Args:
        uncertainty_scores: (H, W) tensor of per-pixel uncertainty.
        num_pixels: Number of pixels to select (K).
        labeled_mask: (H, W) boolean — already labeled pixels.
        vegetation_mask: (H, W) boolean — vegetation pixels.

    Returns:
        selected_coords: (K, 2) tensor of (row, col) pixel coordinates.
    """
    H, W = uncertainty_scores.shape
    scores = uncertainty_scores.clone()

    # Mask out ineligible pixels
    if vegetation_mask is not None:
        scores[~vegetation_mask] = -float("inf")
    if labeled_mask is not None:
        scores[labeled_mask] = -float("inf")

    # Flatten and get top-K indices
    flat = scores.reshape(-1)
    k = min(num_pixels, (flat > -float("inf")).sum().item())
    if k == 0:
        return torch.zeros(0, 2, dtype=torch.long)

    _, top_indices = torch.topk(flat, k)

    # Convert flat indices to (row, col)
    rows = top_indices // W
    cols = top_indices % W
    return torch.stack([rows, cols], dim=1)


def badge_inspired_query(
    uncertainty_scores: torch.Tensor,
    features: torch.Tensor,
    num_pixels: int,
    num_clusters: int = 50,
    labeled_mask: torch.Tensor = None,
    vegetation_mask: torch.Tensor = None,
    top_fraction: float = 0.2,
) -> torch.Tensor:
    """
    BADGE-inspired query: uncertainty-filtered, diversity-selected.

    Algorithm:
    1. Filter to vegetation + unlabeled pixels.
    2. Select the top `top_fraction` most uncertain pixels as candidates.
    3. Extract their feature embeddings from the model's penultimate layer.
    4. Weight features by uncertainty (so K-means is pulled toward uncertain regions).
    5. Run K-means clustering on weighted features.
    6. Select the pixel closest to each cluster center (most representative).

    This gives spatially and semantically diverse queries that are also uncertain —
    the best of both worlds.

    Args:
        uncertainty_scores: (H, W) per-pixel uncertainty (entropy or BALD).
        features: (D, H, W) feature maps from the model's encoder.
        num_pixels: Number of pixels to select (K).
        num_clusters: Number of K-means clusters. Should be >= num_pixels.
            More clusters → finer spatial diversity.
        labeled_mask: (H, W) boolean — already labeled.
        vegetation_mask: (H, W) boolean — vegetation pixels.
        top_fraction: Fraction of uncertain pixels to consider as candidates
            before clustering. E.g., 0.2 = top 20% most uncertain pixels.

    Returns:
        selected_coords: (K, 2) tensor of (row, col) pixel coordinates.
    """
    H, W = uncertainty_scores.shape
    D = features.shape[0]

    # --- Step 1: Filter eligible pixels ---
    eligible = torch.ones(H, W, dtype=torch.bool)
    if vegetation_mask is not None:
        eligible = eligible & vegetation_mask
    if labeled_mask is not None:
        eligible = eligible & (~labeled_mask)

    eligible_coords = torch.nonzero(eligible, as_tuple=False)  # (N, 2)
    n_eligible = len(eligible_coords)

    if n_eligible == 0:
        return torch.zeros(0, 2, dtype=torch.long)

    # --- Step 2: Get uncertainty scores for eligible pixels ---
    eligible_scores = uncertainty_scores[eligible_coords[:, 0], eligible_coords[:, 1]]

    # --- Step 3: Select top uncertain candidates ---
    n_candidates = max(num_pixels, int(n_eligible * top_fraction))
    n_candidates = min(n_candidates, n_eligible)

    _, top_idx = torch.topk(eligible_scores, n_candidates)
    candidate_coords = eligible_coords[top_idx]  # (n_candidates, 2)
    candidate_scores = eligible_scores[top_idx]

    # --- Step 4: Extract feature embeddings for candidates ---
    # features: (D, H, W) → gather at candidate positions
    candidate_features = features[
        :, candidate_coords[:, 0], candidate_coords[:, 1]
    ].T  # (n_candidates, D)

    # --- Step 5: Weight features by uncertainty ---
    # Normalize scores to [0, 1] range for weighting
    score_min = candidate_scores.min()
    score_range = candidate_scores.max() - score_min + 1e-8
    normalized_scores = (candidate_scores - score_min) / score_range

    # Weight each feature vector by its uncertainty score
    # This biases K-means toward uncertain regions without ignoring diversity
    weighted_features = candidate_features * normalized_scores.unsqueeze(1)

    # --- Step 6: K-means clustering ---
    n_clusters = min(num_clusters, n_candidates, num_pixels)
    if n_clusters < 1:
        return candidate_coords[:num_pixels]

    # Move to numpy for sklearn K-means
    features_np = weighted_features.cpu().numpy()

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        cluster_labels = kmeans.fit_predict(features_np)
    except Exception:
        # Fallback: if K-means fails (e.g., too few samples), return top uncertain
        return candidate_coords[:num_pixels]

    # --- Step 7: Select the most uncertain pixel from each cluster ---
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_scores = candidate_scores[cluster_mask]

        if len(cluster_scores) == 0:
            continue

        # Within this cluster, pick the most uncertain pixel
        best_in_cluster = cluster_scores.argmax()
        # Map back to the index in candidate_coords
        cluster_indices = torch.where(torch.from_numpy(cluster_mask))[0]
        selected_indices.append(cluster_indices[best_in_cluster].item())

    if len(selected_indices) == 0:
        return candidate_coords[:num_pixels]

    selected = candidate_coords[selected_indices]

    # If we need more pixels than clusters, fill with remaining top uncertain
    if len(selected) < num_pixels:
        remaining_mask = torch.ones(n_candidates, dtype=torch.bool)
        for idx in selected_indices:
            remaining_mask[idx] = False
        remaining = candidate_coords[remaining_mask]
        remaining_scores = candidate_scores[remaining_mask]
        _, fill_idx = torch.topk(
            remaining_scores, min(num_pixels - len(selected), len(remaining_scores))
        )
        selected = torch.cat([selected, remaining[fill_idx]], dim=0)

    return selected[:num_pixels]
