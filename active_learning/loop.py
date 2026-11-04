"""
active_learning/loop.py — Main active learning loop orchestration (patched).

Changes vs. the previous version (all aimed at round-to-round stability and
honest accounting; splits, pools, metrics and query strategies are untouched):

  1. BatchNorm recalibration after every training round. With batch size 1 and
     a random patch order, BN running stats used to depend on whichever ~10
     patches came last. We now recompute them over ALL train-pool patches
     (unlabeled HSI only, no labels, no test patches).
  2. Best-weights restore. Training-loss early stopping used to keep the LAST
     epoch's weights; we now restore the weights from the best-loss epoch.
  3. Lower, configurable AL learning rate (`retrain_lr`, default 3e-5) since a
     fresh AdamW with lr 1e-4 and no warmup kicked a converged model each round.
  4. Class weights computed from the CURRENTLY LABELED pixels only (the labels
     the oracle has already revealed) and passed to the focal loss, matching
     what supervised training does.
  5. Correct accounting. Flow is now:
         train on L0 -> evaluate ("initial", a real trained model)
         each round: query with model trained on L_{t-1} -> label -> train on
         L_t -> evaluate. The logged labeled_count is exactly what the
         evaluated model was trained on, and the last round's labels are used.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
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
from models.losses import FocalDiceLoss


class ActiveLearningLoop:
    """
    Orchestrates the full active learning experiment.
    """

    def __init__(
        self,
        model: nn.Module,
        full_data: torch.Tensor,
        full_labels: torch.Tensor,
        test_labels: torch.Tensor,
        pca_rgb: Optional[torch.Tensor],
        vegetation_mask: torch.Tensor,
        train_pool_mask: torch.Tensor,
        cfg: dict,
        device: str = "cuda",
        output_dir: str = "./results",
    ):
        self.model = model
        self.full_data = full_data
        self.full_labels = full_labels
        self.test_labels = test_labels
        self.pca_rgb = pca_rgb
        self.vegetation_mask = vegetation_mask
        self.train_pool_mask = train_pool_mask.bool()
        self.device = device
        self.output_dir = output_dir

        # Tallgrass AL composite is a horizontal strip of 128x128 patches.
        self.patch_h = self.full_data.shape[1]
        self.patch_w = 128

        if self.patch_h != 128:
            raise ValueError(
                f"Expected Tallgrass AL composite height 128, "
                f"got {self.patch_h}"
            )

        if self.full_data.shape[2] % self.patch_w != 0:
            raise ValueError(
                f"Composite width {self.full_data.shape[2]} "
                f"is not divisible by patch width {self.patch_w}."
            )

        self.num_patches = self.full_data.shape[2] // self.patch_w

        if self.train_pool_mask.shape != self.full_labels.shape:
            raise ValueError(
                "train_pool_mask shape must match full_labels shape."
            )

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
        base_lr = train_cfg.get("learning_rate", 1e-4)
        # AL fine-tunes an already-trained model every round, so use a gentler LR.
        self.lr = al_cfg.get("retrain_lr", base_lr * 0.3)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.use_amp = train_cfg.get("use_amp", True) and str(device) == "cuda"
        self.train_patience = al_cfg.get("retrain_patience", 10)

        # Simulated oracle
        self.oracle = SimulatedOracle(full_labels)

        # Results storage
        self.results = {
            "rounds": [],
            "strategy": self.strategy,
        }

        # Query budget computed ONLY from the eligible geographic train pool.
        # Background (0) is a valid class; -1 is ignore.
        eligible_train_pixels = (
            self.train_pool_mask & (self.full_labels >= 0)
        )

        n_train_pool = int(eligible_train_pixels.sum().item())

        if n_train_pool == 0:
            raise RuntimeError(
                "No eligible pixels found in the geographic training pool."
            )

        self.n_train_pool = n_train_pool
        self.query_budget = max(10, int(n_train_pool * self.query_fraction))

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        print(f"\n{'='*60}")
        print(f"Active Learning Loop — Strategy: {self.strategy}")
        print(f"Rounds: {self.num_rounds}, Budget/round: {self.query_budget} pixels")
        print(f"AL learning rate: {self.lr:.2e}")
        print(f"{'='*60}\n")

        self.model.to(self.device)

        # --- Initialize labeled pool (geographic train pool only) ---
        eligible_train_pixels = (
            self.train_pool_mask & (self.full_labels >= 0)
        )
        eligible_indices = torch.nonzero(eligible_train_pixels, as_tuple=False)

        n_initial = max(1, int(len(eligible_indices) * self.initial_fraction))

        generator = torch.Generator().manual_seed(42)
        permutation = torch.randperm(
            len(eligible_indices), generator=generator
        )[:n_initial]
        selected = eligible_indices[permutation]

        labeled_mask = torch.zeros_like(self.full_labels, dtype=torch.bool)
        labeled_mask[selected[:, 0], selected[:, 1]] = True

        self.oracle.labeled_mask = labeled_mask.clone()
        self.oracle.total_labeled = int(labeled_mask.sum().item())

        # --- Train on L0, then evaluate (a real "initial" point) ---
        training_labels, _ = self.oracle.get_labeled_data()
        self._train_round(training_labels, round_idx=0)
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

            # Step 1: uncertainty from the model trained on L_{t-1}
            current_mask = self.oracle.labeled_mask.clone()
            uncertainty = self._compute_uncertainty()

            # Step 2: select queries
            query_coords = self._select_queries(
                uncertainty, current_mask, round_idx
            )

            # Step 3: "label" via oracle
            oracle_result = self.oracle.label_pixels(query_coords)

            # Emergency fallback: never end a round with 0 new labels
            if oracle_result["num_new"] == 0:
                valid_pool_mask = (
                    self.train_pool_mask
                    & (self.oracle.ground_truth >= 0)
                    & (~self.oracle.labeled_mask)
                )
                pool_coords = torch.nonzero(valid_pool_mask)
                if len(pool_coords) > 0:
                    min_budget = max(10, self.query_budget // 5)
                    n_emerg = min(min_budget, len(pool_coords))
                    g = torch.Generator().manual_seed(999 + round_idx)
                    perm = torch.randperm(len(pool_coords), generator=g)[:n_emerg]
                    extra = pool_coords[perm]
                    for idx in range(len(extra)):
                        er, ec = extra[idx, 0].item(), extra[idx, 1].item()
                        self.oracle.labeled_mask[er, ec] = True
                    self.oracle.total_labeled = int(
                        self.oracle.labeled_mask.sum().item()
                    )
                    oracle_result["num_new"] += n_emerg
                    query_coords = (
                        torch.cat([query_coords, extra], dim=0)
                        if len(query_coords) > 0 else extra
                    )
                    print(
                        f"  [AL] Emergency top-up: added "
                        f"{n_emerg} valid training-pool pixels."
                    )

            # Step 4: train on L_t (includes the pixels just labeled)
            training_labels, _ = self.oracle.get_labeled_data()
            self._train_round(training_labels, round_idx)

            # Step 5: evaluate the model that was trained on L_t
            metrics = self._evaluate_on_test()

            # Step 6: per-round checkpoint
            ckpt_path = os.path.join(
                self.output_dir,
                f"al_{self.strategy}_round{round_idx:02d}.pt"
            )
            torch.save(
                {"model_state_dict": self.model.state_dict(),
                 "round": round_idx,
                 "labeled_count": self.oracle.total_labeled},
                ckpt_path,
            )

            # Step 7: log
            coords_list = query_coords.tolist() if len(query_coords) > 0 else []
            round_result = {
                "round": round_idx,
                "labeled_count": self.oracle.total_labeled,
                "num_new_labels": oracle_result["num_new"],
                "miou": metrics["miou"],
                "per_class_iou": metrics["per_class_iou"],
                "mean_entropy": float(uncertainty["entropy"].mean()),
                "mean_bald": float(uncertainty["bald"].mean()),
                "query_coords": coords_list,
            }
            self.results["rounds"].append(round_result)
            self._save_results()

            print(f"[Round {round_idx}] mIoU: {metrics['miou']:.4f}, "
                  f"labeled: {self.oracle.total_labeled}")

        self._save_results()
        return self.results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patch_inputs(self, patch_idx: int):
        """Return (hsi, pca) CPU tensors with batch dim for one patch."""
        cs = patch_idx * self.patch_w
        ce = cs + self.patch_w
        hsi = self.full_data[:, :, cs:ce].unsqueeze(0)
        pca = (
            self.pca_rgb[:, :, cs:ce].unsqueeze(0)
            if self.pca_rgb is not None else None
        )
        return hsi, pca

    def _labeled_class_weights(self, training_labels: torch.Tensor):
        """
        Inverse-frequency class weights from the currently labeled pixels only.
        Normalized to mean 1. Returns None if a class is entirely absent.
        """
        num_classes = self.model.num_classes
        valid = training_labels[training_labels >= 0]
        if valid.numel() == 0:
            return None
        counts = torch.bincount(valid, minlength=num_classes).float()
        if (counts == 0).any():
            return None
        weights = 1.0 / counts
        weights = weights / weights.mean()
        return weights

    @torch.no_grad()
    def _recalibrate_bn(self):
        """
        Recompute BatchNorm running statistics as a cumulative average over
        ALL train-pool patches (unlabeled HSI only; no labels, no test data).
        """
        bns = [
            m for m in self.model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]
        if not bns:
            return

        for m in bns:
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average

        self.model.train()
        used = 0
        for patch_idx in range(self.num_patches):
            cs = patch_idx * self.patch_w
            ce = cs + self.patch_w
            if not self.train_pool_mask[:, cs:ce].any():
                continue
            hsi, pca = self._patch_inputs(patch_idx)
            hsi = hsi.to(self.device)
            pca = pca.to(self.device) if pca is not None else None
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                self.model(hsi, pca)
            used += 1

        for m in bns:
            m.momentum = 0.1

        self.model.eval()
        print(f"  [BN] Recalibrated {len(bns)} BatchNorm layers "
              f"over {used} train-pool patches.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_round(self, training_labels: torch.Tensor, round_idx: int):
        """
        Train on the current labeled set (patch-by-patch), restore the
        best-loss weights, then recalibrate BatchNorm statistics.
        """
        self.model.train()
        self.model.to(self.device)

        class_weights = self._labeled_class_weights(training_labels)
        loss_fn = FocalDiceLoss(
            focal_alpha=class_weights,
            ignore_index=-1,
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        epoch = -1
        epoch_loss = float("nan")

        for epoch in range(self.retrain_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_used = 0

            patch_order = torch.randperm(self.num_patches)

            for patch_idx in patch_order.tolist():
                cs = patch_idx * self.patch_w
                ce = cs + self.patch_w

                label_patch = training_labels[:, cs:ce]
                if not torch.any(label_patch >= 0):
                    continue

                hsi, pca = self._patch_inputs(patch_idx)
                hsi = hsi.to(self.device)
                pca = pca.to(self.device) if pca is not None else None
                labels = label_patch.unsqueeze(0).to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    output = self.model(hsi, pca)
                    logits = output["logits"]

                    if logits.shape[2:] != labels.shape[1:]:
                        logits = nn.functional.interpolate(
                            logits,
                            size=labels.shape[1:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    loss = loss_fn(logits, labels)["total"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_used += 1

            if n_used == 0:
                print(
                    f"  WARNING: Round {round_idx}, epoch {epoch+1}: "
                    f"no labeled patches available."
                )
                break

            epoch_loss /= n_used

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {
                    n: p.detach().cpu().clone()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                }
            else:
                patience_counter += 1
                if patience_counter >= self.train_patience:
                    break

        # Restore best-loss weights (not just the last epoch's).
        if best_state is not None:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in best_state:
                        p.copy_(best_state[n].to(p.device))

        print(
            f"  Trained {epoch+1} epochs, best loss: {best_loss:.4f} "
            f"(last: {epoch_loss:.4f}), lr={self.lr:.1e}, "
            f"class_weights="
            f"{None if class_weights is None else [round(w, 3) for w in class_weights.tolist()]}"
        )

        # BN stats must match the weights we are keeping.
        self._recalibrate_bn()

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def _compute_uncertainty(self) -> Dict[str, torch.Tensor]:
        """MC-Dropout uncertainty, patch-by-patch, reassembled into the strip."""
        height = self.full_data.shape[1]
        total_width = self.full_data.shape[2]

        entropy_map = torch.zeros(height, total_width, dtype=torch.float32)
        bald_map = torch.zeros(height, total_width, dtype=torch.float32)

        for patch_idx in tqdm(
            range(self.num_patches), desc="MC-Dropout uncertainty"
        ):
            cs = patch_idx * self.patch_w
            ce = cs + self.patch_w

            hsi, pca = self._patch_inputs(patch_idx)

            uncertainty = mc_dropout_inference(
                self.model,
                hsi,
                pca,
                num_passes=self.mc_passes,
                device=self.device,
            )

            entropy_map[:, cs:ce] = uncertainty["entropy"].squeeze(0).cpu() \
                if uncertainty["entropy"].dim() == 3 else uncertainty["entropy"].cpu()
            bald_map[:, cs:ce] = uncertainty["bald"].squeeze(0).cpu() \
                if uncertainty["bald"].dim() == 3 else uncertainty["bald"].cpu()

        return {"entropy": entropy_map, "bald": bald_map}

    # ------------------------------------------------------------------
    # Query selection (logic unchanged)
    # ------------------------------------------------------------------

    def _select_queries(
        self,
        uncertainty: Dict[str, torch.Tensor],
        labeled_mask: torch.Tensor,
        round_idx: int,
    ) -> torch.Tensor:
        train_unlabeled_mask = (
            self.train_pool_mask
            & (self.full_labels >= 0)
            & (~labeled_mask)
        )

        veg = train_unlabeled_mask.clone()

        if self.vegetation_mask is not None:
            veg = train_unlabeled_mask & self.vegetation_mask.bool()
            n_veg_unlabeled = int(veg.sum().item())

            if n_veg_unlabeled < self.query_budget:
                print(
                    f"  [AL] WARNING: vegetation mask leaves only "
                    f"{n_veg_unlabeled} training-pool candidates "
                    f"(budget={self.query_budget}). "
                    f"Falling back to all eligible training-pool pixels."
                )
                veg = train_unlabeled_mask

        if self.strategy == "random":
            query_coords = random_query(
                num_pixels=self.query_budget,
                total_pixels=labeled_mask.numel(),
                labeled_mask=labeled_mask,
                vegetation_mask=veg,
                seed=42 + round_idx,
            )
        elif self.strategy == "entropy":
            query_coords = uncertainty_query(
                uncertainty_scores=uncertainty["entropy"],
                num_pixels=self.query_budget,
                labeled_mask=labeled_mask,
                vegetation_mask=veg,
            )
        elif self.strategy in ("bald", "badge_inspired"):
            scores = uncertainty["bald"]
            if self.strategy == "badge_inspired" and self.use_diversity:
                features = self._compute_feature_strip()
                query_coords = badge_inspired_query(
                    uncertainty_scores=scores,
                    features=features,
                    num_pixels=self.query_budget,
                    num_clusters=self.diversity_clusters,
                    labeled_mask=labeled_mask,
                    vegetation_mask=veg,
                )
            else:
                query_coords = uncertainty_query(
                    uncertainty_scores=scores,
                    num_pixels=self.query_budget,
                    labeled_mask=labeled_mask,
                    vegetation_mask=veg,
                )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # --- Minimum budget guarantee ---
        min_budget = max(10, self.query_budget // 5)
        gt = self.oracle.ground_truth

        if len(query_coords) > 0:
            r = query_coords[:, 0]
            c = query_coords[:, 1]
            valid_new = (
                self.train_pool_mask[r, c]
                & (gt[r, c] >= 0)
                & (~labeled_mask[r, c])
            )
            n_valid = int(valid_new.sum().item())
        else:
            n_valid = 0

        if n_valid < min_budget:
            shortfall = min_budget - n_valid

            valid_pool_mask = (
                self.train_pool_mask & (gt >= 0) & (~labeled_mask)
            )

            if len(query_coords) > 0:
                valid_pool_mask[query_coords[:, 0], query_coords[:, 1]] = False

            pool_coords = torch.nonzero(valid_pool_mask)

            if len(pool_coords) > 0:
                n_sample = min(shortfall, len(pool_coords))
                g = torch.Generator().manual_seed(42 + round_idx * 1000)
                perm = torch.randperm(len(pool_coords), generator=g)[:n_sample]
                extra_coords = pool_coords[perm]

                query_coords = (
                    torch.cat([query_coords, extra_coords], dim=0)
                    if len(query_coords) > 0 else extra_coords
                )

                print(
                    f"  [AL] Top-up guarantee: added "
                    f"{n_sample} random training-pool pixels "
                    f"(projected new: {n_valid + n_sample}, "
                    f"min_budget={min_budget})"
                )

        return query_coords

    @torch.no_grad()
    def _compute_feature_strip(self) -> torch.Tensor:
        """
        Patch-by-patch feature extraction for badge_inspired (the old code
        pushed the whole 128x6400 strip through SAM2 at once).
        Returns (D, H, W_total) on CPU.
        """
        self.model.eval()
        feats = []
        for patch_idx in range(self.num_patches):
            hsi, pca = self._patch_inputs(patch_idx)
            hsi = hsi.to(self.device)
            pca = pca.to(self.device) if pca is not None else None
            out = self.model(hsi, pca)
            feats.append(out["features"].squeeze(0).cpu())
        return torch.cat(feats, dim=2)

    # ------------------------------------------------------------------
    # Evaluation (logic unchanged)
    # ------------------------------------------------------------------

    def _evaluate_on_test(self) -> Dict:
        self.model.eval()

        height = self.full_data.shape[1]
        total_width = self.full_data.shape[2]

        pred = torch.zeros(height, total_width, dtype=torch.long)

        with torch.no_grad():
            for patch_idx in range(self.num_patches):
                cs = patch_idx * self.patch_w
                ce = cs + self.patch_w

                test_patch = self.test_labels[:, cs:ce]
                if not torch.any(test_patch >= 0):
                    continue

                hsi, pca = self._patch_inputs(patch_idx)
                hsi = hsi.to(self.device)
                pca = pca.to(self.device) if pca is not None else None

                logits = self.model(hsi, pca)["logits"]

                if logits.shape[2:] != test_patch.shape:
                    logits = nn.functional.interpolate(
                        logits,
                        size=test_patch.shape,
                        mode="bilinear",
                        align_corners=False,
                    )

                pred[:, cs:ce] = logits.squeeze(0).argmax(dim=0).cpu()

        valid = self.test_labels >= 0

        if valid.sum() == 0:
            return {"miou": 0.0, "per_class_iou": {}}

        pred_valid = pred[valid]
        gt_valid = self.test_labels[valid]

        miou = compute_miou(pred_valid, gt_valid, self.model.num_classes)
        per_class = compute_per_class_iou(
            pred_valid, gt_valid, self.model.num_classes
        )

        return {"miou": miou, "per_class_iou": per_class}

    def _save_results(self):
        output_path = os.path.join(
            self.output_dir, f"al_results_{self.strategy}.json"
        )
        serializable = json.loads(json.dumps(self.results, default=str))
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[AL Loop] Results saved to {output_path}")