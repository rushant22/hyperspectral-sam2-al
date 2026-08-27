"""
scripts/run_al_loop.py — Run the full active learning experiment.

Executes the AL loop for multiple strategies (BALD, random, entropy) and
generates annotation-efficiency curves comparing them.

Usage:
    # Run with default config (BALD strategy)
    python scripts/run_al_loop.py --config configs/default.yaml

    # Run specific strategy
    python scripts/run_al_loop.py --config configs/default.yaml --strategy bald

    # Run all strategies for comparison
    python scripts/run_al_loop.py --config configs/default.yaml --all-strategies
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indian_pines import IndianPinesDataset
from data.pavia import PaviaDataset
from data.transforms import get_eval_transform
from data.utils import apply_pca, compute_ndvi, get_ndvi_band_indices
from models.sam2_wrapper import build_model
from active_learning.loop import ActiveLearningLoop
from evaluation.plots import plot_annotation_efficiency


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(cfg: dict):
    """
    Load dataset and prepare all tensors needed for the AL loop.

    Returns a dict with:
      - full_data: (B, H, W) HSI tensor
      - full_labels: (H, W) ground truth
      - test_labels: (H, W) test split labels (-1 for non-test pixels)
      - pca_rgb: (3, H, W) PCA-reduced image
      - vegetation_mask: (H, W) boolean
      - num_bands: int
      - num_classes: int
      - class_names: dict
      - pca_model: fitted PCA model
    """
    dataset_name = cfg["dataset"]["name"]
    root_dir = os.path.join(cfg["dataset"]["root_dir"], dataset_name)

    print(f"\n=== Loading {dataset_name} dataset ===")

    if dataset_name == "indian_pines":
        # Indian Pines: use the entire image as one "patch"
        dataset = IndianPinesDataset(root_dir=root_dir, seed=cfg["seed"])
        full_data = dataset.data  # (145, 145, 200)
        full_labels = dataset.labels  # (145, 145)
        num_classes = dataset.num_classes + 1
        class_names = dataset.class_names
        band_mean = dataset.band_mean
        band_std = dataset.band_std

    elif dataset_name == "pavia":
        # Pavia: use the full image (not patched) for AL
        dataset = PaviaDataset(root_dir=root_dir, seed=cfg["seed"])
        full_data = dataset.full_data  # (610, 340, 103)
        full_labels = dataset.full_labels  # (610, 340)
        num_classes = dataset.num_classes + 1
        class_names = dataset.class_names
        band_mean = dataset.band_mean
        band_std = dataset.band_std
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Normalize ---
    from data.utils import normalize_per_band
    full_data_norm, _, _ = normalize_per_band(full_data, band_mean, band_std)

    # --- PCA for residual pathway ---
    pca_data, pca_model = apply_pca(full_data_norm, n_components=3)

    # --- NDVI vegetation mask ---
    red_idx, nir_idx = get_ndvi_band_indices(dataset_name)
    ndvi = compute_ndvi(full_data, red_idx, nir_idx)
    ndvi_threshold = cfg["active_learning"].get("ndvi_threshold", 0.2)
    vegetation_mask = torch.from_numpy(ndvi > ndvi_threshold)
    print(f"  Vegetation mask: {vegetation_mask.sum().item()} / {vegetation_mask.numel()} pixels "
          f"({100*vegetation_mask.float().mean():.1f}%) above NDVI threshold {ndvi_threshold}")

    # --- Create test split labels ---
    # Use the dataset's split mechanism to get test pixel mask
    rng = np.random.RandomState(cfg["seed"])
    test_labels = np.full(full_labels.shape, -1, dtype=np.int64)
    ratios = (
        cfg["dataset"]["train_ratio"],
        cfg["dataset"]["val_ratio"],
        cfg["dataset"]["test_ratio"],
    )
    # Stratified test split
    for class_id in range(1, num_classes):
        class_pixels = np.argwhere(full_labels == class_id)
        n = len(class_pixels)
        if n == 0:
            continue
        indices = rng.permutation(n)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        test_idx = indices[n_train + n_val:]
        for idx in test_idx:
            r, c = class_pixels[idx]
            test_labels[r, c] = full_labels[r, c]

    # --- Convert to tensors ---
    # Data: (H, W, B) → (B, H, W)
    full_data_tensor = torch.from_numpy(
        full_data_norm.transpose(2, 0, 1)
    ).float()

    # PCA: (H, W, 3) → (3, H, W)
    pca_tensor = torch.from_numpy(pca_data.transpose(2, 0, 1)).float()

    full_labels_tensor = torch.from_numpy(full_labels).long()
    test_labels_tensor = torch.from_numpy(test_labels).long()

    return {
        "full_data": full_data_tensor,
        "full_labels": full_labels_tensor,
        "test_labels": test_labels_tensor,
        "pca_rgb": pca_tensor,
        "vegetation_mask": vegetation_mask,
        "num_bands": full_data.shape[2],
        "num_classes": num_classes,
        "class_names": class_names,
        "pca_model": pca_model,
    }


def run_al_experiment(cfg: dict, strategy: str, data: dict) -> dict:
    """
    Run one AL experiment with a specific strategy.

    Args:
        cfg: Full config dict.
        strategy: One of "bald", "entropy", "random", "badge_inspired".
        data: Dict from prepare_data().

    Returns:
        Results dict from the AL loop.
    """
    set_seed(cfg["seed"])
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Build a fresh model for each strategy (independent experiments)
    model = build_model(
        cfg,
        num_bands=data["num_bands"],
        num_classes=data["num_classes"],
        pca_model=data["pca_model"],
    )

    # Override strategy in config
    al_cfg = cfg.copy()
    al_cfg["active_learning"] = cfg.get("active_learning", {}).copy()
    al_cfg["active_learning"]["strategy"] = strategy

    # Create and run the AL loop
    al_loop = ActiveLearningLoop(
        model=model,
        full_data=data["full_data"],
        full_labels=data["full_labels"],
        test_labels=data["test_labels"],
        pca_rgb=data["pca_rgb"],
        vegetation_mask=data["vegetation_mask"],
        cfg=al_cfg,
        device=device,
        output_dir=cfg["evaluation"]["output_dir"],
    )

    results = al_loop.run()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run active learning experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["bald", "entropy", "random", "badge_inspired"])
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run all strategies for comparison")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Prepare data (shared across strategies) ---
    data = prepare_data(cfg)

    # --- Determine which strategies to run ---
    if args.all_strategies:
        strategies = ["bald", "random", "entropy"]
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = [cfg["active_learning"]["strategy"]]

    # --- Run experiments ---
    results_files = {}
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running AL experiment: {strategy.upper()}")
        print(f"{'='*60}")

        run_al_experiment(cfg, strategy, data)
        results_files[strategy.upper()] = os.path.join(
            cfg["evaluation"]["output_dir"],
            f"al_results_{strategy}.json",
        )

    # --- Generate comparison plot ---
    if len(strategies) > 1:
        print("\n=== Generating comparison plot ===")
        plot_annotation_efficiency(
            results_files=results_files,
            output_path=os.path.join(
                cfg["evaluation"]["output_dir"],
                "annotation_efficiency_comparison.pdf",
            ),
            title=f"Annotation Efficiency — {cfg['dataset']['name'].title()}",
        )

    print("\n=== All experiments complete ===")


if __name__ == "__main__":
    main()
