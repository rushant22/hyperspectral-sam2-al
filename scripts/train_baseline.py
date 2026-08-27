"""
scripts/train_baseline.py — Train baseline model (PCA → encoder, no spectral adapter).

This establishes the performance FLOOR that the spectral adapter must beat.
Without the adapter, the model reduces B bands to 3 via PCA and processes
them through the encoder + segmentation head. This tests whether the spatial
processing alone (without spectral awareness) can segment vegetation classes.

Usage:
    python scripts/train_baseline.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indian_pines import IndianPinesDataset
from data.pavia import PaviaDataset
from data.transforms import get_train_transform, get_eval_transform
from data.utils import apply_pca
from models.sam2_wrapper import build_model
from models.losses import FocalDiceLoss
from evaluation.metrics import compute_miou, compute_per_class_iou, compute_overall_accuracy


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(cfg: dict, split: str):
    """Load the configured dataset with appropriate transforms."""
    dataset_name = cfg["dataset"]["name"]
    root_dir = os.path.join(cfg["dataset"]["root_dir"], dataset_name)
    target_size = cfg["dataset"]["sam2_input_size"]

    if dataset_name == "indian_pines":
        dataset = IndianPinesDataset(
            root_dir=root_dir,
            split=split,
            seed=cfg["seed"],
        )
    elif dataset_name == "pavia":
        dataset = PaviaDataset(
            root_dir=root_dir,
            patch_size=cfg["dataset"]["patch_size"],
            patch_overlap=cfg["dataset"]["patch_overlap"],
            split=split,
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create transforms with the dataset's normalization stats
    if split == "train":
        transform = get_train_transform(target_size, dataset.band_mean, dataset.band_std)
    else:
        transform = get_eval_transform(target_size, dataset.band_mean, dataset.band_std)

    dataset.transform = transform

    # Compute PCA model on the full data for the baseline
    if dataset_name == "indian_pines":
        full_data = dataset.data
    else:
        full_data = dataset.full_data

    _, pca_model = apply_pca(full_data, n_components=cfg["dataset"]["pca_components"])

    return dataset, pca_model


def train_one_epoch(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    pca_model,
) -> dict:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        data = batch["data"].to(device)        # (B_batch, bands, H, W)
        labels = batch["labels"].to(device)    # (B_batch, H, W)

        # Compute PCA for the residual pathway
        with torch.no_grad():
            # PCA per sample in the batch
            pca_rgbs = []
            for i in range(data.shape[0]):
                hsi_np = data[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, B)
                pca_np, _ = apply_pca(hsi_np, n_components=3, pca_model=pca_model)
                pca_tensor = torch.from_numpy(pca_np.transpose(2, 0, 1)).float()
                pca_rgbs.append(pca_tensor)
            pca_rgb = torch.stack(pca_rgbs).to(device)  # (B_batch, 3, H, W)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(data, pca_rgb)
            logits = output["logits"]

            # Resize logits to match label size if needed
            if logits.shape[2:] != labels.shape[1:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[1:],
                    mode="bilinear", align_corners=False,
                )

            loss_dict = loss_fn(logits, labels)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    num_classes: int,
    device: str,
    pca_model,
) -> dict:
    """Evaluate on validation/test set, return metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["labels"]

        # PCA for residual pathway
        pca_rgbs = []
        for i in range(data.shape[0]):
            hsi_np = data[i].cpu().numpy().transpose(1, 2, 0)
            pca_np, _ = apply_pca(hsi_np, n_components=3, pca_model=pca_model)
            pca_tensor = torch.from_numpy(pca_np.transpose(2, 0, 1)).float()
            pca_rgbs.append(pca_tensor)
        pca_rgb = torch.stack(pca_rgbs).to(device)

        output = model(data, pca_rgb)
        logits = output["logits"]

        if logits.shape[2:] != labels.shape[1:]:
            logits = nn.functional.interpolate(
                logits, size=labels.shape[1:],
                mode="bilinear", align_corners=False,
            )

        pred = logits.argmax(dim=1).cpu()  # (B, H, W)

        # Collect valid (non-ignored) pixels
        for i in range(pred.shape[0]):
            valid = labels[i] >= 0
            if valid.sum() > 0:
                all_preds.append(pred[i][valid])
                all_targets.append(labels[i][valid])

    if len(all_preds) == 0:
        return {"miou": 0.0, "oa": 0.0, "per_class_iou": {}}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    miou = compute_miou(all_preds, all_targets, num_classes)
    oa = compute_overall_accuracy(all_preds, all_targets)
    per_class = compute_per_class_iou(all_preds, all_targets, num_classes)

    return {"miou": miou, "oa": oa, "per_class_iou": per_class}


def main():
    parser = argparse.ArgumentParser(description="Train baseline model (no adapter)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        device = "cpu"

    # --- Load data ---
    print("\n=== Loading dataset ===")
    train_dataset, pca_model = load_dataset(cfg, "train")
    val_dataset, _ = load_dataset(cfg, "val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # 0 for Windows compatibility
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --- Build model (baseline: adapter disabled) ---
    print("\n=== Building baseline model ===")
    # Override config to disable adapter for baseline
    baseline_cfg = cfg.copy()
    baseline_cfg["adapter"] = {"enabled": False, "use_residual": True}

    num_classes = train_dataset.num_classes + 1  # +1 because classes are 1-indexed
    model = build_model(
        baseline_cfg,
        num_bands=train_dataset.num_bands,
        num_classes=num_classes,
        pca_model=pca_model,
    )
    model = model.to(device)

    # --- Loss & optimizer ---
    class_weights = train_dataset.get_class_weights()
    loss_fn = FocalDiceLoss(
        focal_gamma=cfg["loss"]["focal_gamma"],
        focal_alpha=class_weights,
        focal_weight=cfg["loss"]["focal_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        dice_smooth=cfg["loss"]["dice_smooth"],
        ignore_index=-1,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scaler = torch.amp.GradScaler('cuda', enabled=cfg["training"]["use_amp"])

    # --- Training loop ---
    print(f"\n=== Training baseline for {cfg['training']['epochs']} epochs ===")
    best_miou = 0.0
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler,
            cfg["training"]["use_amp"], pca_model,
        )

        # Validate every 5 epochs (or every epoch for small datasets)
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(
                model, val_loader, num_classes, device, pca_model,
            )
            print(
                f"Epoch {epoch:3d} | Loss: {train_metrics['loss']:.4f} | "
                f"Val mIoU: {val_metrics['miou']:.4f} | Val OA: {val_metrics['oa']:.4f}"
            )

            # Early stopping check
            if val_metrics["miou"] > best_miou + 1e-4:
                best_miou = val_metrics["miou"]
                patience_counter = 0
                # Save best model
                save_path = os.path.join(cfg["evaluation"]["output_dir"], "baseline_best.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 5  # We validate every 5 epochs
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (best mIoU: {best_miou:.4f})")
                    break

    # --- Final evaluation on test set ---
    print("\n=== Final test evaluation ===")
    test_dataset, _ = load_dataset(cfg, "test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    # Load best model
    best_path = os.path.join(cfg["evaluation"]["output_dir"], "baseline_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, test_loader, num_classes, device, pca_model)
    print(f"\n{'='*60}")
    print(f"BASELINE TEST RESULTS")
    print(f"  mIoU: {test_metrics['miou']:.4f}")
    print(f"  OA:   {test_metrics['oa']:.4f}")
    print(f"  Per-class IoU:")
    for c, iou in test_metrics['per_class_iou'].items():
        class_name = train_dataset.class_names.get(c, f"Class {c}")
        print(f"    {class_name}: {iou:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
