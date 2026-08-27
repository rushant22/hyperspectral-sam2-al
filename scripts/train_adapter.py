"""
scripts/train_adapter.py — Train with Spectral Cross-Attention Adapter + LoRA.

This is the MAIN training script — trains the full model with the spectral
adapter enabled. Compare results against train_baseline.py to quantify the
adapter's contribution.

Usage:
    python scripts/train_adapter.py --config configs/default.yaml
    python scripts/train_adapter.py --config configs/default.yaml --dataset indian_pines
"""

import os
import sys
import argparse
import yaml
import time
import json
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
from data.utils import apply_pca, normalize_per_band
from models.sam2_wrapper import build_model
from models.losses import FocalDiceLoss
from models.lora import count_trainable_params
from evaluation.metrics import (
    compute_miou, compute_per_class_iou,
    compute_overall_accuracy, compute_kappa,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Note: torch.use_deterministic_algorithms(True) can cause errors
        # with some ops. Enable only for final reproducibility validation.


def load_dataset(cfg: dict, split: str):
    """Load dataset and return (dataset, pca_model)."""
    dataset_name = cfg["dataset"]["name"]
    root_dir = os.path.join(cfg["dataset"]["root_dir"], dataset_name)

    if dataset_name == "indian_pines":
        dataset = IndianPinesDataset(root_dir=root_dir, split=split, seed=cfg["seed"])
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

    target_size = cfg["dataset"]["sam2_input_size"]
    if split == "train":
        transform = get_train_transform(target_size, dataset.band_mean, dataset.band_std)
    else:
        transform = get_eval_transform(target_size, dataset.band_mean, dataset.band_std)
    dataset.transform = transform

    # PCA model for the residual pathway
    if dataset_name == "indian_pines":
        full_data = dataset.data
    else:
        full_data = dataset.full_data
    _, pca_model = apply_pca(full_data, n_components=cfg["dataset"]["pca_components"])

    return dataset, pca_model


def compute_pca_batch(data_batch: torch.Tensor, pca_model) -> torch.Tensor:
    """
    Compute PCA-reduced 3-channel images for a batch.
    data_batch: (batch, bands, H, W) tensor
    Returns: (batch, 3, H, W) tensor
    """
    pca_list = []
    for i in range(data_batch.shape[0]):
        hsi_np = data_batch[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, B)
        pca_np, _ = apply_pca(hsi_np, n_components=3, pca_model=pca_model)
        pca_list.append(torch.from_numpy(pca_np.transpose(2, 0, 1)).float())
    return torch.stack(pca_list)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler,
                    device, scaler, use_amp, pca_model, grad_accum_steps):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        data = batch["data"].to(device)
        labels = batch["labels"].to(device)

        # PCA for residual pathway
        with torch.no_grad():
            pca_rgb = compute_pca_batch(data, pca_model).to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(data, pca_rgb)
            logits = output["logits"]

            if logits.shape[2:] != labels.shape[1:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[1:],
                    mode="bilinear", align_corners=False,
                )

            loss_dict = loss_fn(logits, labels)
            loss = loss_dict["total"] / grad_accum_steps

        scaler.scale(loss).backward()

        # Gradient accumulation: step optimizer every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["total"].item()
        total_focal += loss_dict["focal"].item()
        total_dice += loss_dict["dice"].item()
        num_batches += 1

    if scheduler is not None:
        scheduler.step()

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "focal": total_focal / n,
        "dice": total_dice / n,
    }


@torch.no_grad()
def evaluate(model, dataloader, num_classes, device, pca_model):
    """Evaluate on val/test set, return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["labels"]

        pca_rgb = compute_pca_batch(data, pca_model).to(device)

        output = model(data, pca_rgb)
        logits = output["logits"]

        if logits.shape[2:] != labels.shape[1:]:
            logits = nn.functional.interpolate(
                logits, size=labels.shape[1:],
                mode="bilinear", align_corners=False,
            )

        pred = logits.argmax(dim=1).cpu()

        for i in range(pred.shape[0]):
            valid = labels[i] >= 0
            if valid.sum() > 0:
                all_preds.append(pred[i][valid])
                all_targets.append(labels[i][valid])

    if len(all_preds) == 0:
        return {"miou": 0.0, "oa": 0.0, "kappa": 0.0, "per_class_iou": {}}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "miou": compute_miou(all_preds, all_targets, num_classes),
        "oa": compute_overall_accuracy(all_preds, all_targets),
        "kappa": compute_kappa(all_preds, all_targets, num_classes),
        "per_class_iou": compute_per_class_iou(all_preds, all_targets, num_classes),
    }


def main():
    parser = argparse.ArgumentParser(description="Train with Spectral Adapter + LoRA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset name from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
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
        shuffle=True, num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0, pin_memory=(device == "cuda"),
    )

    # --- Build model (adapter ENABLED) ---
    print("\n=== Building model with Spectral Adapter + LoRA ===")
    num_classes = train_dataset.num_classes + 1
    model = build_model(cfg, train_dataset.num_bands, num_classes, pca_model)
    model = model.to(device)

    # Log parameter budget
    param_info = count_trainable_params(model)
    print(f"  Total params:     {param_info['total']:>12,}")
    print(f"  Trainable params: {param_info['trainable']:>12,} ({param_info['trainable_pct']:.1f}%)")
    print(f"  Frozen params:    {param_info['frozen']:>12,}")

    # Log GPU memory if available
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory (model): {allocated:.0f} MB")

    # --- Loss function ---
    class_weights = train_dataset.get_class_weights()
    loss_fn = FocalDiceLoss(
        focal_gamma=cfg["loss"]["focal_gamma"],
        focal_alpha=class_weights,
        focal_weight=cfg["loss"]["focal_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        dice_smooth=cfg["loss"]["dice_smooth"],
        ignore_index=-1,
    )

    # --- Optimizer with separate LR for decoder ---
    # The adapter and LoRA params get the main LR.
    # The mask decoder (fully trainable) gets a lower LR since it's pretrained.
    adapter_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "spectral_adapter" in name:
            adapter_params.append(param)
        elif "seg_head" in name or "pca_stem" in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {"params": adapter_params, "lr": cfg["training"]["learning_rate"]},
        {"params": other_params, "lr": cfg["training"]["learning_rate"]},
        {"params": decoder_params, "lr": cfg["training"].get("decoder_lr", 5e-5)},
    ], weight_decay=cfg["training"]["weight_decay"])

    # --- Learning rate scheduler ---
    scheduler_type = cfg["training"].get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"],
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler('cuda', enabled=cfg["training"]["use_amp"] and device == "cuda")

    # --- Training loop ---
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    grad_accum = cfg["training"].get("grad_accumulation_steps", 1)
    warmup_epochs = cfg["training"].get("warmup_epochs", 5)

    print(f"\n=== Training for {epochs} epochs ===")
    print(f"  Batch size: {cfg['training']['batch_size']} × {grad_accum} (grad accum)")
    print(f"  LR: {cfg['training']['learning_rate']} (adapter/LoRA), "
          f"{cfg['training'].get('decoder_lr', 5e-5)} (decoder)")

    best_miou = 0.0
    patience_counter = 0
    training_log = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Warmup: linearly increase LR for the first few epochs
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * warmup_factor if "initial_lr" in pg else pg["lr"]

        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            device, scaler, cfg["training"]["use_amp"] and device == "cuda",
            pca_model, grad_accum,
        )
        epoch_time = time.time() - epoch_start

        # Validate
        val_metrics = {}
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate(model, val_loader, num_classes, device, pca_model)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss: {train_metrics['loss']:.4f} "
                f"(F:{train_metrics['focal']:.3f} D:{train_metrics['dice']:.3f}) | "
                f"Val mIoU: {val_metrics['miou']:.4f} | "
                f"OA: {val_metrics['oa']:.4f} | "
                f"Kappa: {val_metrics['kappa']:.4f} | "
                f"{epoch_time:.1f}s"
            )

            # Early stopping
            if val_metrics["miou"] > best_miou + 1e-4:
                best_miou = val_metrics["miou"]
                patience_counter = 0
                save_dir = cfg["evaluation"]["output_dir"]
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "adapter_best.pt"))
            else:
                patience_counter += 5
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (best mIoU: {best_miou:.4f})")
                    break

        # Log
        log_entry = {
            "epoch": epoch, "time": epoch_time,
            **train_metrics, **val_metrics,
        }
        training_log.append(log_entry)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # --- Final test evaluation ---
    print("\n=== Final test evaluation ===")
    test_dataset, _ = load_dataset(cfg, "test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    best_path = os.path.join(cfg["evaluation"]["output_dir"], "adapter_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    test_metrics = evaluate(model, test_loader, num_classes, device, pca_model)

    print(f"\n{'='*60}")
    print(f"ADAPTER MODEL TEST RESULTS")
    print(f"  mIoU:  {test_metrics['miou']:.4f}")
    print(f"  OA:    {test_metrics['oa']:.4f}")
    print(f"  Kappa: {test_metrics['kappa']:.4f}")
    print(f"  Per-class IoU:")
    for c, iou in sorted(test_metrics["per_class_iou"].items()):
        class_name = train_dataset.class_names.get(c, f"Class {c}")
        if not np.isnan(iou):
            print(f"    {class_name}: {iou:.4f}")
    print(f"{'='*60}")

    # --- Save training log ---
    log_path = os.path.join(cfg["evaluation"]["output_dir"], "adapter_training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": cfg,
            "param_info": param_info,
            "training_log": training_log,
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "per_class_iou"},
            "per_class_iou": {str(k): v for k, v in test_metrics["per_class_iou"].items()},
            "total_training_time_seconds": total_time,
        }, f, indent=2, default=str)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
