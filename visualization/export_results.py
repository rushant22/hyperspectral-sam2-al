"""
visualization/export_results.py — Export experiment results for the web dashboard.

Converts internal result formats (PyTorch tensors, numpy arrays, JSON logs)
into the structure expected by the dashboard's JavaScript frontend:
  - uncertainty_map.json: per-pixel BALD/entropy scores
  - query_history.json: AL query locations per round with entropy coloring
  - segmentation_map.json: predicted class map
  - false_color.json: PCA-derived false-color image for display
  - metrics_summary.json: per-round mIoU and annotation counts

All data is exported as JSON arrays (not binary) for easy consumption
by vanilla JavaScript without any build tooling.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Optional
from data.utils import apply_pca


def export_for_dashboard(
    full_data: np.ndarray,
    full_labels: np.ndarray,
    predictions: np.ndarray,
    uncertainty_map: np.ndarray,
    query_history: list,
    al_results: dict,
    output_dir: str = "./visualization/dashboard/data",
    max_resolution: int = 256,
):
    """
    Export all data needed by the web dashboard.

    Args:
        full_data: HSI cube, shape (H, W, B).
        full_labels: Ground truth, shape (H, W).
        predictions: Predicted class map, shape (H, W).
        uncertainty_map: BALD/entropy scores, shape (H, W).
        query_history: List of dicts from SimulatedOracle.history.
        al_results: Results dict from ActiveLearningLoop.
        output_dir: Where to write the JSON files.
        max_resolution: Downsample to this size if image is larger
            (keeps JSON files small enough for the browser).
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W = full_data.shape[:2]

    # Downsample if needed (dashboard doesn't need full resolution)
    scale = min(1.0, max_resolution / max(H, W))
    if scale < 1.0:
        new_H, new_W = int(H * scale), int(W * scale)
        # Simple nearest-neighbor downsampling for labels/predictions
        from skimage.transform import resize
        full_data_small = resize(full_data, (new_H, new_W, full_data.shape[2]),
                                  anti_aliasing=True, preserve_range=True)
        full_labels_small = resize(full_labels.astype(float), (new_H, new_W),
                                    order=0, preserve_range=True).astype(int)
        predictions_small = resize(predictions.astype(float), (new_H, new_W),
                                    order=0, preserve_range=True).astype(int)
        uncertainty_small = resize(uncertainty_map, (new_H, new_W),
                                    anti_aliasing=True, preserve_range=True)
    else:
        new_H, new_W = H, W
        full_data_small = full_data
        full_labels_small = full_labels
        predictions_small = predictions
        uncertainty_small = uncertainty_map

    # --- 1. False-color image (PCA → RGB) ---
    pca_rgb, _ = apply_pca(full_data_small, n_components=3)
    # Normalize to 0-255 for display
    for c in range(3):
        channel = pca_rgb[:, :, c]
        cmin, cmax = np.percentile(channel, [2, 98])
        pca_rgb[:, :, c] = np.clip((channel - cmin) / (cmax - cmin + 1e-8) * 255, 0, 255)
    pca_rgb = pca_rgb.astype(np.uint8)

    false_color_data = {
        "width": new_W,
        "height": new_H,
        "pixels": pca_rgb.tolist(),
    }
    with open(os.path.join(output_dir, "false_color.json"), "w") as f:
        json.dump(false_color_data, f)

    # --- 2. Uncertainty map ---
    # Normalize to 0-1 for consistent coloring
    u_min, u_max = uncertainty_small.min(), uncertainty_small.max()
    if u_max > u_min:
        uncertainty_norm = ((uncertainty_small - u_min) / (u_max - u_min)).tolist()
    else:
        uncertainty_norm = np.zeros_like(uncertainty_small).tolist()

    uncertainty_data = {
        "width": new_W,
        "height": new_H,
        "values": uncertainty_norm,
        "min_raw": float(u_min),
        "max_raw": float(u_max),
    }
    with open(os.path.join(output_dir, "uncertainty_map.json"), "w") as f:
        json.dump(uncertainty_data, f)

    # --- 3. Segmentation map ---
    seg_data = {
        "width": new_W,
        "height": new_H,
        "ground_truth": full_labels_small.tolist(),
        "predictions": predictions_small.tolist(),
    }
    with open(os.path.join(output_dir, "segmentation_map.json"), "w") as f:
        json.dump(seg_data, f)

    # --- 4. Query history ---
    query_data = []
    for entry in query_history:
        coords = entry["coords"]
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        # Scale coordinates to match downsampled image
        scaled_coords = (coords * scale).astype(int).tolist()
        query_data.append({
            "round": entry["round"],
            "coordinates": scaled_coords,
            "num_new": entry["num_new"],
        })

    with open(os.path.join(output_dir, "query_history.json"), "w") as f:
        json.dump(query_data, f)

    # --- 5. Metrics summary ---
    metrics_data = {
        "initial": al_results.get("initial", {}),
        "rounds": al_results.get("rounds", []),
        "strategy": al_results.get("strategy", "unknown"),
    }
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_data, f, default=str)

    print(f"[Export] Dashboard data written to {output_dir}/")
    print(f"  Resolution: {new_W}×{new_H} (scale={scale:.2f})")
    print(f"  Files: false_color.json, uncertainty_map.json, "
          f"segmentation_map.json, query_history.json, metrics_summary.json")
