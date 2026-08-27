"""
scripts/run_ablations.py — Run all ablation experiments for the paper.

Ablations:
  1. Adapter contribution: adapter+LoRA vs. no-adapter baseline
  2. LoRA rank: r=4 vs. r=8 vs. r=16
  3. Number of spectral queries: M=4 vs. M=8 vs. M=12 vs. M=16
  4. AL strategy comparison: BALD vs. entropy vs. random
  5. AL budget sensitivity: 1% vs. 2% vs. 5% per round

Each ablation retrains from scratch to avoid contamination.
Results are saved as JSON and compiled into a summary table.

Usage:
    python scripts/run_ablations.py --config configs/default.yaml
    python scripts/run_ablations.py --config configs/default.yaml --ablation lora_rank
"""

import os
import sys
import argparse
import yaml
import json
import copy
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indian_pines import IndianPinesDataset
from data.pavia import PaviaDataset
from data.transforms import get_train_transform, get_eval_transform
from data.utils import apply_pca
from models.sam2_wrapper import build_model
from models.losses import FocalDiceLoss
from evaluation.metrics import compute_miou, compute_per_class_iou, compute_overall_accuracy


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quick_train_eval(cfg: dict, label: str) -> dict:
    """
    Train a model with the given config and return test metrics.

    Simplified version of train_adapter.py for ablation runs — uses fewer
    epochs and simpler logging to keep the total ablation time manageable.
    """
    set_seed(cfg["seed"])
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset_name = cfg["dataset"]["name"]
    root_dir = os.path.join(cfg["dataset"]["root_dir"], dataset_name)
    target_size = cfg["dataset"]["sam2_input_size"]

    # Load datasets
    if dataset_name == "indian_pines":
        train_ds = IndianPinesDataset(root_dir=root_dir, split="train", seed=cfg["seed"])
        test_ds = IndianPinesDataset(root_dir=root_dir, split="test", seed=cfg["seed"])
        full_data = train_ds.data
    elif dataset_name == "pavia":
        train_ds = PaviaDataset(root_dir=root_dir, split="train", seed=cfg["seed"],
                                patch_size=cfg["dataset"]["patch_size"],
                                patch_overlap=cfg["dataset"]["patch_overlap"])
        test_ds = PaviaDataset(root_dir=root_dir, split="test", seed=cfg["seed"],
                               patch_size=cfg["dataset"]["patch_size"],
                               patch_overlap=cfg["dataset"]["patch_overlap"])
        full_data = train_ds.full_data
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_ds.transform = get_train_transform(target_size, train_ds.band_mean, train_ds.band_std)
    test_ds.transform = get_eval_transform(target_size, test_ds.band_mean, test_ds.band_std)

    _, pca_model = apply_pca(full_data, n_components=cfg["dataset"]["pca_components"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    num_classes = train_ds.num_classes + 1
    model = build_model(cfg, train_ds.num_bands, num_classes, pca_model).to(device)

    loss_fn = FocalDiceLoss(
        focal_gamma=cfg["loss"]["focal_gamma"],
        focal_alpha=train_ds.get_class_weights(),
        ignore_index=-1,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scaler = torch.amp.GradScaler('cuda', enabled=cfg["training"]["use_amp"] and device == "cuda")

    # Ablation training: use fewer epochs for speed
    ablation_epochs = min(cfg["training"]["epochs"], 60)
    print(f"  [{label}] Training for {ablation_epochs} epochs...")

    model.train()
    for epoch in range(ablation_epochs):
        for batch in train_loader:
            data = batch["data"].to(device)
            labels = batch["labels"].to(device)

            # PCA
            pca_list = []
            for i in range(data.shape[0]):
                hsi_np = data[i].cpu().numpy().transpose(1, 2, 0)
                pca_np, _ = apply_pca(hsi_np, n_components=3, pca_model=pca_model)
                pca_list.append(torch.from_numpy(pca_np.transpose(2, 0, 1)).float())
            pca_rgb = torch.stack(pca_list).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=cfg["training"]["use_amp"] and device == "cuda"):
                output = model(data, pca_rgb)
                logits = output["logits"]
                if logits.shape[2:] != labels.shape[1:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
                loss_dict = loss_fn(logits, labels)

            scaler.scale(loss_dict["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            data = batch["data"].to(device)
            labels = batch["labels"]
            pca_list = []
            for i in range(data.shape[0]):
                hsi_np = data[i].cpu().numpy().transpose(1, 2, 0)
                pca_np, _ = apply_pca(hsi_np, n_components=3, pca_model=pca_model)
                pca_list.append(torch.from_numpy(pca_np.transpose(2, 0, 1)).float())
            pca_rgb = torch.stack(pca_list).to(device)

            output = model(data, pca_rgb)
            logits = output["logits"]
            if logits.shape[2:] != labels.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).cpu()
            for i in range(pred.shape[0]):
                valid = labels[i] >= 0
                if valid.sum() > 0:
                    all_preds.append(pred[i][valid])
                    all_targets.append(labels[i][valid])

    if len(all_preds) == 0:
        return {"miou": 0.0, "oa": 0.0, "per_class_iou": {}}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {
        "miou": compute_miou(all_preds, all_targets, num_classes),
        "oa": compute_overall_accuracy(all_preds, all_targets),
        "per_class_iou": compute_per_class_iou(all_preds, all_targets, num_classes),
    }
    print(f"  [{label}] mIoU: {metrics['miou']:.4f}, OA: {metrics['oa']:.4f}")
    return metrics


# =============================================================================
# Ablation Definitions
# =============================================================================

def ablation_adapter_contribution(base_cfg: dict) -> dict:
    """Ablation 1: With vs. without spectral adapter."""
    print("\n=== Ablation: Adapter Contribution ===")
    results = {}

    # Without adapter (baseline)
    cfg = copy.deepcopy(base_cfg)
    cfg["adapter"]["enabled"] = False
    results["no_adapter"] = quick_train_eval(cfg, "No Adapter")

    # With adapter (full model)
    cfg = copy.deepcopy(base_cfg)
    cfg["adapter"]["enabled"] = True
    results["with_adapter"] = quick_train_eval(cfg, "With Adapter")

    delta = results["with_adapter"]["miou"] - results["no_adapter"]["miou"]
    print(f"\n  Adapter improvement: {delta:+.4f} mIoU")
    return results


def ablation_lora_rank(base_cfg: dict) -> dict:
    """Ablation 2: LoRA rank sensitivity."""
    print("\n=== Ablation: LoRA Rank ===")
    results = {}

    for rank in [4, 8, 16]:
        cfg = copy.deepcopy(base_cfg)
        cfg["lora"]["rank"] = rank
        cfg["lora"]["alpha"] = rank * 2  # Keep alpha/rank = 2
        results[f"rank_{rank}"] = quick_train_eval(cfg, f"r={rank}")

    return results


def ablation_num_queries(base_cfg: dict) -> dict:
    """Ablation 3: Number of spectral queries (M)."""
    print("\n=== Ablation: Number of Queries (M) ===")
    results = {}

    for M in [4, 8, 12, 16]:
        cfg = copy.deepcopy(base_cfg)
        cfg["adapter"]["num_queries"] = M
        results[f"M_{M}"] = quick_train_eval(cfg, f"M={M}")

    return results


def ablation_seg_head(base_cfg: dict) -> dict:
    """Ablation 4: Segmentation head type."""
    print("\n=== Ablation: Segmentation Head ===")
    results = {}

    for head_type in ["conv1x1", "mlp_2layer"]:
        cfg = copy.deepcopy(base_cfg)
        cfg["seg_head"]["type"] = head_type
        results[head_type] = quick_train_eval(cfg, head_type)

    return results


ABLATIONS = {
    "adapter_contribution": ablation_adapter_contribution,
    "lora_rank": ablation_lora_rank,
    "num_queries": ablation_num_queries,
    "seg_head": ablation_seg_head,
}


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=list(ABLATIONS.keys()),
                        help="Run a specific ablation. Omit to run all.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    output_dir = base_cfg["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Determine which ablations to run
    if args.ablation:
        ablations_to_run = {args.ablation: ABLATIONS[args.ablation]}
    else:
        ablations_to_run = ABLATIONS

    all_results = {}
    for name, fn in ablations_to_run.items():
        results = fn(base_cfg)
        all_results[name] = results

    # Save all results
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for ablation_name, results in all_results.items():
        print(f"\n  {ablation_name}:")
        for variant, metrics in results.items():
            print(f"    {variant:20s} -> mIoU: {metrics['miou']:.4f}, OA: {metrics['oa']:.4f}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
