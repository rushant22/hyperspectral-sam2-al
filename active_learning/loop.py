"""
active_learning/loop.py — Main active learning loop orchestration.

This is the top-level AL pipeline that ties together:
  - Model training (with growing labeled set)
  - MC-Dropout uncertainty estimation
  - Query strategy (BALD + spatial diversity)
  - Simulated oracle (ground truth lookup)
  - Metric logging at each round

The loop runs for T rounds, each round:
  1. Train/fine-tune model on current labeled set Lₜ
  2. Run MC-Dropout inference on unlabeled set → uncertainty maps
  3. Apply NDVI mask (vegetation-only queries)
  4. Select K pixels using the chosen strategy
  5. "Label" them via simulated oracle → Lₜ₊₁
  6. Log metrics (mIoU, annotation count, etc.)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from active_learning.uncertainty import mc_dropout_inference
from active_learning.query_strategies import (
    random_query,
    uncertainty_query,
    badge_inspired_query,
)
from active_learning.simulated_oracle import SimulatedOracle
from evaluation.metrics import compute_miou, compute_per_class_iou


class ActiveLearningLoop:
    """
    Orchestrates the full active learning experiment.

    Designed to be run as a complete experiment — outputs all metrics
    needed for the annotation-efficiency curves in the paper.
    """

    def __init__(
        self,
        model: nn.Module,
        full_data: torch.Tensor,
        full_labels: torch.Tensor,
        test_labels: torch.Tensor,
        pca_rgb: Optional[torch.Tensor],
        vegetation_mask: torch.Tensor,
        cfg: dict,
        device: str = "cuda",
        output_dir: str = "./results",
    ):
        """
        Args:
            model: The segmentation model (AdaptedSAM2 instance).
            full_data: Full HSI image, shape (B, H, W).
            full_labels: Full ground truth, shape (H, W).
            test_labels: Test split labels, shape (H, W), -1 for non-test pixels.
            pca_rgb: PCA-reduced 3-channel image, shape (3, H, W).
            vegetation_mask: (H, W) boolean — True = vegetation pixel.
            cfg: Config dict (active_learning section).
            device: Training device.
            output_dir: Directory to save results.
        """
        self.model = model
        self.full_data = full_data
        self.full_labels = full_labels
        self.test_labels = test_labels
        self.pca_rgb = pca_rgb
        self.vegetation_mask = vegetation_mask
        self.device = device
        self.output_dir = output_dir

        # AL parameters from config
        al_cfg = cfg.get("active_learning", {})
        self.num_rounds = al_cfg.get("num_rounds", 10)
        self.initial_fraction = al_cfg.get("initial_labeled_fraction", 0.05)
        self.query_fraction = al_cfg.get("query_fraction_per_round", 0.02)
        self.mc_passes = al_cfg.get("mc_dropout_passes", 10)
        self.strategy = al_cfg.get("strategy", "bald")
        self.use_diversity = al_cfg.get("use_spatial_diversity", True)
        self.diversity_clusters = al_cfg.get("diversity_clusters", 50)
        self.retrain_epochs = al_cfg.get("retrain_epochs", 50)
        self.ndvi_threshold = al_cfg.get("ndvi_threshold", 0.2)

        # Training parameters
        train_cfg = cfg.get("training", {})
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.use_amp = train_cfg.get("use_amp", True) and device == "cuda"

        # Simulated oracle
        self.oracle = SimulatedOracle(full_labels)

        # Results storage
        self.results = {
            "rounds": [],
            "strategy": self.strategy,
        }

        # Compute number of query pixels per round
        n_non_bg = (full_labels > 0).sum().item()
        self.query_budget = max(1, int(n_non_bg * self.query_fraction))

        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict:
        """
        Execute the full active learning loop.

        Returns:
            Dict with results for all rounds (for plotting).
        """
        print(f"\n{'='*60}")
        print(f"Active Learning Loop — Strategy: {self.strategy}")
        print(f"Rounds: {self.num_rounds}, Budget/round: {self.query_budget} pixels")
        print(f"{'='*60}\n")

        # --- Initialize labeled pool ---
        labeled_mask = self.oracle.initialize_random_labels(
            self.initial_fraction, seed=42
        )

        # --- Evaluate initial model (before any AL) ---
        initial_metrics = self._evaluate_on_test()
        self.results["initial"] = {
            "labeled_count": self.oracle.total_labeled,
            "miou": initial_metrics["miou"],
            "per_class_iou": initial_metrics["per_class_iou"],
        }
        print(f"[Initial] mIoU: {initial_metrics['miou']:.4f}, "
              f"labeled: {self.oracle.total_labeled}")

        # --- AL rounds ---
        for round_idx in range(1, self.num_rounds + 1):
            print(f"\n--- AL Round {round_idx}/{self.num_rounds} ---")

            # Step 1: Train on current labeled set
            training_labels, current_mask = self.oracle.get_labeled_data()
            self._train_round(training_labels, round_idx)

            # Step 2: MC-Dropout uncertainty estimation
            uncertainty = self._compute_uncertainty()

            # Step 3: Select query pixels
            query_coords = self._select_queries(
                uncertainty, current_mask, round_idx
            )

            # Step 4: "Label" via oracle
            oracle_result = self.oracle.label_pixels(query_coords)

            # Step 5: Evaluate on test set
            metrics = self._evaluate_on_test()

            # Step 6: Log results
            round_result = {
                "round": round_idx,
                "labeled_count": self.oracle.total_labeled,
                "num_new_labels": oracle_result["num_new"],
                "miou": metrics["miou"],
                "per_class_iou": metrics["per_class_iou"],
                "mean_entropy": uncertainty["entropy"].mean().item(),
                "mean_bald": uncertainty["bald"].mean().item(),
            }
            self.results["rounds"].append(round_result)

            print(f"[Round {round_idx}] mIoU: {metrics['miou']:.4f}, "
                  f"labeled: {self.oracle.total_labeled}")

        # --- Save results ---
        self._save_results()

        return self.results

    def _train_round(self, training_labels: torch.Tensor, round_idx: int):
        """
        Train the model on the current labeled set for one AL round.

        Uses the training labels where unlabeled pixels have label = -1
        (ignored by the loss function).
        """
        from models.losses import FocalDiceLoss

        self.model.train()
        self.model.to(self.device)

        # Prepare input tensors
        hsi = self.full_data.unsqueeze(0).to(self.device)  # (1, B, H, W)
        labels = training_labels.unsqueeze(0).to(self.device)  # (1, H, W)
        pca = None
        if self.pca_rgb is not None:
            pca = self.pca_rgb.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Loss and optimizer
        loss_fn = FocalDiceLoss(ignore_index=-1)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # AMP scaler for mixed precision
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        best_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(self.retrain_epochs):
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                output = self.model(hsi, pca)
                logits = output["logits"]  # (1, C, H, W)

                # Resize logits to match label size if needed
                if logits.shape[2:] != labels.shape[1:]:
                    logits = nn.functional.interpolate(
                        logits, size=labels.shape[1:], mode="bilinear",
                        align_corners=False,
                    )

                loss_dict = loss_fn(logits, labels)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            # Simple early stopping
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        print(f"  Trained for {epoch+1} epochs, final loss: {loss.item():.4f}")

    def _compute_uncertainty(self) -> Dict[str, torch.Tensor]:
        """Run MC-Dropout inference to get uncertainty maps."""
        hsi = self.full_data.unsqueeze(0)  # (1, B, H, W)
        pca = self.pca_rgb.unsqueeze(0) if self.pca_rgb is not None else None

        uncertainty = mc_dropout_inference(
            self.model, hsi, pca,
            num_passes=self.mc_passes,
            device=self.device,
        )
        return uncertainty

    def _select_queries(
        self,
        uncertainty: Dict[str, torch.Tensor],
        labeled_mask: torch.Tensor,
        round_idx: int,
    ) -> torch.Tensor:
        """Select pixels to query based on the configured strategy."""

        if self.strategy == "random":
            return random_query(
                num_pixels=self.query_budget,
                total_pixels=labeled_mask.numel(),
                labeled_mask=labeled_mask,
                vegetation_mask=self.vegetation_mask,
                seed=42 + round_idx,
            )

        # For entropy/bald strategies, get the appropriate score map
        if self.strategy == "entropy":
            scores = uncertainty["entropy"]
        elif self.strategy in ("bald", "badge_inspired"):
            scores = uncertainty["bald"]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if self.strategy == "badge_inspired" and self.use_diversity:
            # Need features for clustering — re-run a single forward pass
            self.model.eval()
            with torch.no_grad():
                hsi = self.full_data.unsqueeze(0).to(self.device)
                pca = self.pca_rgb.unsqueeze(0).to(self.device) if self.pca_rgb is not None else None
                output = self.model(hsi, pca)
                features = output["features"].squeeze(0).cpu()  # (D, H, W)

            return badge_inspired_query(
                uncertainty_scores=scores,
                features=features,
                num_pixels=self.query_budget,
                num_clusters=self.diversity_clusters,
                labeled_mask=labeled_mask,
                vegetation_mask=self.vegetation_mask,
            )
        else:
            return uncertainty_query(
                uncertainty_scores=scores,
                num_pixels=self.query_budget,
                labeled_mask=labeled_mask,
                vegetation_mask=self.vegetation_mask,
            )

    def _evaluate_on_test(self) -> Dict:
        """Evaluate model on the test set."""
        self.model.eval()

        with torch.no_grad():
            hsi = self.full_data.unsqueeze(0).to(self.device)
            pca = self.pca_rgb.unsqueeze(0).to(self.device) if self.pca_rgb is not None else None
            output = self.model(hsi, pca)
            logits = output["logits"]

            # Resize if needed
            if logits.shape[2:] != self.test_labels.shape:
                logits = nn.functional.interpolate(
                    logits, size=self.test_labels.shape, mode="bilinear",
                    align_corners=False,
                )

            pred = logits.squeeze(0).argmax(dim=0).cpu()  # (H, W)

        # Compute metrics only on test pixels (test_labels != -1)
        valid = self.test_labels >= 0
        if valid.sum() == 0:
            return {"miou": 0.0, "per_class_iou": {}}

        pred_valid = pred[valid]
        gt_valid = self.test_labels[valid]

        miou = compute_miou(pred_valid, gt_valid, self.model.num_classes)
        per_class = compute_per_class_iou(pred_valid, gt_valid, self.model.num_classes)

        return {"miou": miou, "per_class_iou": per_class}

    def _save_results(self):
        """Save AL results to JSON for later plotting."""
        output_path = os.path.join(
            self.output_dir, f"al_results_{self.strategy}.json"
        )

        # Convert tensors/numpy to Python types for JSON serialization
        serializable = json.loads(json.dumps(self.results, default=str))

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"\n[AL Loop] Results saved to {output_path}")
