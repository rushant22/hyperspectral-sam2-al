"""
evaluation/plots.py — Generate publication-quality plots for the paper.

All plots are generated from actual experiment results (JSON files from
the AL loop), never from fabricated data. If a result isn't available,
the plot function will print a warning and skip, not invent a placeholder.

Primary plots:
  1. Annotation-efficiency curve: mIoU vs. % labeled pixels
  2. Per-class IoU bar chart
  3. Uncertainty heatmap
  4. Active learning query locations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional

# Use non-interactive backend for server/headless environments
matplotlib.use("Agg")

# Publication-quality plot settings
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})


def plot_annotation_efficiency(
    results_files: Dict[str, str],
    output_path: str,
    title: str = "Annotation Efficiency: mIoU vs. Labeled Pixels",
    full_supervision_miou: Optional[float] = None,
):
    """
    Plot the annotation-efficiency curve comparing AL strategies.

    This is the KEY FIGURE for the paper — shows that BALD-based AL achieves
    higher mIoU with fewer labels compared to random sampling.

    Args:
        results_files: Dict mapping strategy name → path to JSON results file.
            E.g., {"BALD": "results/al_results_bald.json",
                   "Random": "results/al_results_random.json"}
        output_path: Where to save the plot (e.g., "paper/figures/al_curve.pdf").
        title: Plot title.
        full_supervision_miou: If available, draw as a horizontal dashed line.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color scheme for different strategies
    colors = {
        "BALD": "#2196F3",          # Blue
        "Random": "#9E9E9E",        # Gray
        "Entropy": "#FF9800",       # Orange
        "BADGE-inspired": "#4CAF50", # Green
    }

    markers = {
        "BALD": "o",
        "Random": "s",
        "Entropy": "^",
        "BADGE-inspired": "D",
    }

    for strategy_name, filepath in results_files.items():
        if not os.path.exists(filepath):
            print(f"[WARNING] Results file not found: {filepath} — skipping {strategy_name}")
            continue

        with open(filepath, "r") as f:
            results = json.load(f)

        # Extract data points
        labeled_counts = [results.get("initial", {}).get("labeled_count", 0)]
        mious = [results.get("initial", {}).get("miou", 0)]

        for round_data in results.get("rounds", []):
            labeled_counts.append(round_data["labeled_count"])
            mious.append(round_data["miou"])

        color = colors.get(strategy_name, "#000000")
        marker = markers.get(strategy_name, "o")

        ax.plot(
            labeled_counts, mious,
            color=color, marker=marker,
            label=strategy_name, linewidth=2.5, markersize=8,
            markeredgecolor="white", markeredgewidth=1,
        )

    # Full supervision upper bound
    if full_supervision_miou is not None:
        ax.axhline(
            y=full_supervision_miou, color="#F44336", linestyle="--",
            linewidth=2, alpha=0.7, label=f"Full Supervision ({full_supervision_miou:.3f})",
        )

    ax.set_xlabel("Number of Labeled Pixels")
    ax.set_ylabel("mIoU")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved annotation-efficiency curve to {output_path}")


def plot_per_class_iou(
    per_class_iou: Dict[int, float],
    class_names: Dict[int, str],
    output_path: str,
    title: str = "Per-Class IoU",
):
    """
    Bar chart of per-class IoU values.

    Shows class-level performance, highlighting which vegetation classes
    the model handles well and which are challenging.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    classes = sorted(per_class_iou.keys())
    ious = [per_class_iou[c] for c in classes]
    names = [class_names.get(c, f"Class {c}") for c in classes]

    # Color bars by IoU value (red = low, green = high)
    colors = []
    for iou in ious:
        if np.isnan(iou):
            colors.append("#BDBDBD")  # Gray for absent classes
        elif iou < 0.3:
            colors.append("#F44336")  # Red
        elif iou < 0.6:
            colors.append("#FF9800")  # Orange
        else:
            colors.append("#4CAF50")  # Green

    bars = ax.bar(range(len(classes)), ious, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, iou in zip(bars, ious):
        if not np.isnan(iou):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{iou:.2f}", ha="center", va="bottom", fontsize=10,
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved per-class IoU to {output_path}")


def plot_uncertainty_map(
    uncertainty: np.ndarray,
    output_path: str,
    title: str = "Pixel Uncertainty (BALD Score)",
    cmap: str = "inferno",
):
    """
    Heatmap of per-pixel uncertainty scores.

    Used for the visualization dashboard and paper figures.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(uncertainty, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Uncertainty Score")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved uncertainty map to {output_path}")


def plot_query_locations(
    image_rgb: np.ndarray,
    query_coords: np.ndarray,
    uncertainty_scores: np.ndarray,
    output_path: str,
    title: str = "Active Learning Query Locations",
    entropy_thresholds: tuple = (0.3, 0.5, 0.8),
):
    """
    Overlay AL query markers on an RGB/false-color image.

    Markers are color-coded by uncertainty level:
      - Red: high uncertainty (> threshold_high)
      - Yellow: medium uncertainty
      - Green: low uncertainty (< threshold_low)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display the background image
    ax.imshow(image_rgb, interpolation="nearest")

    low_thresh, mid_thresh, high_thresh = entropy_thresholds

    # Color-code markers by uncertainty
    for i, (coord, score) in enumerate(zip(query_coords, uncertainty_scores)):
        if score > high_thresh:
            color = "#F44336"  # Red
            marker = "^"
        elif score > mid_thresh:
            color = "#FFEB3B"  # Yellow
            marker = "o"
        else:
            color = "#4CAF50"  # Green
            marker = "s"

        ax.plot(
            coord[1], coord[0],  # col, row for matplotlib
            marker=marker, color=color, markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
        )

    ax.set_title(title)
    ax.axis("off")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#F44336",
               markersize=10, label=f"High (>{high_thresh})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFEB3B",
               markersize=10, label=f"Medium ({mid_thresh}-{high_thresh})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
               markersize=10, label=f"Low (<{mid_thresh})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", title="Uncertainty")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved query locations to {output_path}")
