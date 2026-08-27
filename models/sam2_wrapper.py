"""
models/sam2_wrapper.py — Full model assembly: Spectral Adapter + LoRA-SAM2 + Multi-Class Head.

This module orchestrates the complete segmentation pipeline:
  1. Spectral Cross-Attention Adapter: HSI (B bands) → feature map (256 channels)
  2. Frozen SAM2 image encoder (Hiera) with LoRA adapters on Q/V projections
  3. SAM2 mask decoder (fully trainable)
  4. Multi-class segmentation head (1×1 conv on decoder features)

The "Residual Approach A" from the plan:
  - PCA the HSI to 3 bands → pass through SAM2's normal stem → spatial features
  - Also pass HSI through the spectral adapter → spectral features
  - ADD them → feed combined features into the rest of Hiera

This preserves SAM2's learned spatial processing while adding spectral awareness.

NOTE: SAM2's mask decoder is class-agnostic (binary foreground/background).
We add a small multi-class head on top to produce C-class segmentation maps.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from sklearn.decomposition import PCA
import numpy as np

from models.spectral_adapter import SpectralCrossAttentionAdapter
from models.lora import inject_lora, get_lora_params, count_trainable_params


class MultiClassHead(nn.Module):
    """
    Simple multi-class segmentation head.

    Takes SAM2's 256-dim feature map and produces per-pixel class logits.
    Intentionally simple (1×1 conv) — the heavy lifting is done by SAM2's
    encoder and our spectral adapter. Adding depth here risks overfitting
    on small HSI datasets.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 9,
        head_type: str = "conv1x1",
        hidden_dim: int = 128,
    ):
        """
        Args:
            in_channels: Input feature dimension (256 from SAM2).
            num_classes: Number of output classes (dataset-specific).
            head_type: "conv1x1" for a single 1×1 conv, or "mlp_2layer"
                for a 2-layer MLP with hidden dim. Start with conv1x1.
            hidden_dim: Hidden dimension for mlp_2layer.
        """
        super().__init__()
        self.head_type = head_type
        self.num_classes = num_classes

        if head_type == "conv1x1":
            self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        elif head_type == "mlp_2layer":
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, in_channels, H, W) from SAM2 features
        Returns:
            logits: (batch, num_classes, H, W)
        """
        return self.head(features)


class AdaptedSAM2(nn.Module):
    """
    Full segmentation model: Spectral Adapter + Frozen SAM2 + LoRA + Multi-Class Head.

    This is a "SAM2-as-feature-extractor" architecture. We don't use SAM2's
    prompt-based interface (points, boxes, masks) — instead, we directly
    access the image encoder's feature maps and attach our own segmentation head.

    This approach is chosen because:
    1. We don't have point/box prompts for HSI segmentation (it's dense prediction).
    2. SAM2's mask decoder is designed for interactive segmentation, not dense
       semantic segmentation. We keep its features but add our own classification.
    3. The prompt encoder is frozen and unused — no memory waste.
    """

    def __init__(
        self,
        num_bands: int,
        num_classes: int,
        sam2_checkpoint: Optional[str] = None,
        sam2_model_cfg: str = "sam2.1_hiera_b+.yaml",
        adapter_cfg: Optional[dict] = None,
        lora_cfg: Optional[dict] = None,
        head_cfg: Optional[dict] = None,
        use_adapter: bool = True,
        use_pca_residual: bool = True,
        pca_model: Optional[PCA] = None,
    ):
        """
        Args:
            num_bands: Number of input HSI bands.
            num_classes: Number of segmentation classes (excluding background,
                which is handled via ignore_index in the loss).
            sam2_checkpoint: Path to SAM2 .pt checkpoint file.
            sam2_model_cfg: SAM2 config name for model building.
            adapter_cfg: Dict of SpectralCrossAttentionAdapter kwargs.
            lora_cfg: Dict with keys: rank, alpha, dropout, target_modules.
            head_cfg: Dict with keys: type, hidden_dim.
            use_adapter: If False, skip the spectral adapter (baseline mode).
            use_pca_residual: If True, add PCA→stem features as residual to
                adapter output (Approach A from the plan).
            pca_model: Pre-fitted PCA model for the residual pathway.
        """
        super().__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes
        self.use_adapter = use_adapter
        self.use_pca_residual = use_pca_residual
        self.embed_dim = 256  # SAM2's image encoder output dimension

        # --- Default configs ---
        adapter_cfg = adapter_cfg or {
            "num_queries": 12, "d_model": 256, "num_heads": 8,
            "ffn_dim": 512, "dropout": 0.1,
        }
        lora_cfg = lora_cfg or {
            "rank": 8, "alpha": 16.0, "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
        }
        head_cfg = head_cfg or {"type": "conv1x1", "hidden_dim": 128}

        # === Component 1: Spectral Cross-Attention Adapter ===
        if use_adapter:
            # Ensure adapter d_model matches SAM2's embed_dim
            adapter_cfg["d_model"] = self.embed_dim
            self.spectral_adapter = SpectralCrossAttentionAdapter(
                num_bands=num_bands,
                **adapter_cfg,
            )
        else:
            self.spectral_adapter = None

        # === Component 2: Feature Encoder (SAM2 Hiera or Lightweight Fallback) ===
        self.sam2_encoder = None
        self.use_sam2_backbone = False

        if sam2_checkpoint is not None and os.path.exists(sam2_checkpoint):
            try:
                from sam2.build_sam import build_sam2
                cfg_path = sam2_model_cfg
                if not cfg_path.startswith("configs/"):
                    if "sam2.1" in cfg_path:
                        cfg_path = f"configs/sam2.1/{cfg_path.split('/')[-1]}"
                    else:
                        cfg_path = f"configs/{cfg_path}"

                sam2_model = build_sam2(cfg_path, sam2_checkpoint, device="cpu")
                self.sam2_encoder = sam2_model.image_encoder

                # Freeze SAM2 backbone parameters
                for param in self.sam2_encoder.parameters():
                    param.requires_grad = False

                # Inject LoRA into SAM2 encoder linear projections
                if lora_cfg and lora_cfg.get("rank", 0) > 0:
                    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj", "proj", "qkv"])
                    inject_lora(
                        self.sam2_encoder,
                        target_module_names=target_modules,
                        rank=lora_cfg.get("rank", 8),
                        alpha=lora_cfg.get("alpha", 16.0),
                        dropout=lora_cfg.get("dropout", 0.1),
                    )

                self.use_sam2_backbone = True
                print(f"[AdaptedSAM2] Successfully loaded frozen SAM2 backbone from {sam2_checkpoint} with LoRA")
            except Exception as e:
                print(f"[AdaptedSAM2] Warning: Failed to load SAM2 ({e}). Falling back to lightweight encoder.")
                self.feature_encoder = self._build_lightweight_encoder()
        else:
            self.feature_encoder = self._build_lightweight_encoder()

        # === Component 3: Multi-class segmentation head ===
        self.seg_head = MultiClassHead(
            in_channels=self.embed_dim,
            num_classes=num_classes,
            head_type=head_cfg["type"],
            hidden_dim=head_cfg.get("hidden_dim", 128),
        )

        # === PCA residual pathway ===
        if use_pca_residual:
            self.pca_stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, self.embed_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU(),
            )
        else:
            self.pca_stem = None

        # Store PCA model for the residual pathway
        self.pca_model = pca_model

        # === Log parameter counts ===
        param_info = count_trainable_params(self)
        print(f"[AdaptedSAM2] Parameters: {param_info}")

    def _build_lightweight_encoder(self) -> nn.Module:
        """
        Build a lightweight CNN encoder for initial development or CPU testing.
        """
        class ResBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.block = nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )
            def forward(self, x):
                return x + self.block(x)

        return nn.Sequential(
            ResBlock(self.embed_dim),
            ResBlock(self.embed_dim),
            ResBlock(self.embed_dim),
            ResBlock(self.embed_dim),
        )

    def forward(
        self,
        hsi: torch.Tensor,
        pca_rgb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: HSI -> per-pixel class logits.

        Args:
            hsi: Hyperspectral input, shape (batch, B, H, W).
            pca_rgb: PCA-reduced 3-channel input for spatial pathway,
                shape (batch, 3, H, W).

        Returns:
            Dict with:
              - "logits": (batch, num_classes, H, W) — class predictions
              - "features": (batch, embed_dim, H, W) — intermediate features
        """
        batch, B, H, W = hsi.shape

        # --- Step 1: Spectral adapter ---
        if self.use_adapter and self.spectral_adapter is not None:
            adapter_features = self.spectral_adapter(hsi)  # (batch, 256, H, W)
        else:
            adapter_features = torch.zeros(
                batch, self.embed_dim, H, W, device=hsi.device
            )

        # --- Step 2: Spatial Feature Extraction (SAM2 Hiera or PCA stem) ---
        if self.use_sam2_backbone and self.sam2_encoder is not None:
            if pca_rgb is None:
                # Fallback: create 3-channel input on the fly or take first 3 bands
                pca_rgb = hsi[:, :3, :, :] if B >= 3 else hsi.repeat(1, 3, 1, 1)[:, :3, :, :]

            # SAM2's Hiera backbone uses windowed positional embeddings that
            # require spatial dimensions to be divisible by a specific factor.
            # Resize to the nearest compatible size, run the encoder, then
            # resize features back to the original (H, W).
            _ALIGN = 64  # Hiera patch_stride(4) × window(8) × hierarchy(2)
            sam2_H = ((H + _ALIGN - 1) // _ALIGN) * _ALIGN
            sam2_W = ((W + _ALIGN - 1) // _ALIGN) * _ALIGN
            if (sam2_H, sam2_W) != (H, W):
                pca_rgb_resized = F.interpolate(
                    pca_rgb, size=(sam2_H, sam2_W), mode="bilinear", align_corners=False
                )
            else:
                pca_rgb_resized = pca_rgb

            # SAM 2 image encoder forward
            sam2_out = self.sam2_encoder(pca_rgb_resized)
            # Use high-res backbone FPN feature or vision_features
            if "backbone_fpn" in sam2_out and len(sam2_out["backbone_fpn"]) > 0:
                spatial_feats = sam2_out["backbone_fpn"][0]
            else:
                spatial_feats = sam2_out["vision_features"]

            # Resize spatial features to match adapter spatial dimensions
            if spatial_feats.shape[2:] != (H, W):
                spatial_feats = F.interpolate(
                    spatial_feats, size=(H, W), mode="bilinear", align_corners=False
                )

            # Residual fusion: spectral adapter features + SAM2 spatial features
            features = adapter_features + spatial_feats
        else:
            # Fallback lightweight CNN pathway
            if self.use_pca_residual and self.pca_stem is not None:
                if pca_rgb is not None:
                    pca_features = self.pca_stem(pca_rgb)
                else:
                    pca_features = self._compute_pca_features(hsi)
                combined = adapter_features + pca_features
            else:
                combined = adapter_features

            features = self.feature_encoder(combined)

        # --- Step 3: Multi-class segmentation head ---
        logits = self.seg_head(features)  # (batch, num_classes, H, W)

        return {
            "logits": logits,
            "features": features,
        }

    def _compute_pca_features(self, hsi: torch.Tensor) -> torch.Tensor:
        """
        On-the-fly PCA → stem features (fallback when precomputed PCA isn't available).

        This is slow (CPU-bound PCA per batch) — prefer precomputing PCA and
        passing pca_rgb to forward() directly.
        """
        batch, B, H, W = hsi.shape

        if self.pca_model is None:
            # No PCA model fitted — return zeros (effectively disabling residual)
            return torch.zeros(batch, self.embed_dim, H, W, device=hsi.device)

        # Move to CPU for sklearn PCA
        hsi_np = hsi.detach().cpu().numpy()
        pca_outputs = []
        for i in range(batch):
            flat = hsi_np[i].reshape(B, -1).T  # (H*W, B)
            reduced = self.pca_model.transform(flat)  # (H*W, 3)
            reduced = reduced.reshape(H, W, 3).transpose(2, 0, 1)  # (3, H, W)
            pca_outputs.append(reduced)

        pca_rgb = torch.from_numpy(np.stack(pca_outputs)).float().to(hsi.device)
        return self.pca_stem(pca_rgb)

    def enable_mc_dropout(self):
        """
        Enable dropout during inference for MC-Dropout uncertainty estimation.

        Normally, dropout is disabled during model.eval(). For MC-Dropout,
        we need dropout active even during inference to get stochastic
        predictions. This method selectively enables dropout layers while
        keeping batch norm in eval mode (so running stats are used, not batch stats).
        """
        self.eval()  # First, set everything to eval mode

        # Then re-enable dropout layers specifically
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def get_adapter_attention_maps(self, hsi: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral attention maps for interpretability/visualization.

        Delegates to the spectral adapter's get_attention_maps method.
        """
        if self.spectral_adapter is not None:
            return self.spectral_adapter.get_attention_maps(hsi)
        return None


def build_model(cfg: dict, num_bands: int, num_classes: int, pca_model=None) -> AdaptedSAM2:
    """
    Factory function to build the model from a config dict.

    Args:
        cfg: Parsed YAML config (the full config, not just model section).
        num_bands: Number of spectral bands in the dataset.
        num_classes: Number of segmentation classes.
        pca_model: Pre-fitted PCA model for residual pathway.

    Returns:
        AdaptedSAM2 model instance.
    """
    model = AdaptedSAM2(
        num_bands=num_bands,
        num_classes=num_classes,
        sam2_checkpoint=cfg.get("sam2", {}).get("checkpoint"),
        sam2_model_cfg=cfg.get("sam2", {}).get("model_cfg", "sam2.1_hiera_b+.yaml"),
        adapter_cfg={
            "num_queries": cfg.get("adapter", {}).get("num_queries", 12),
            "d_model": cfg.get("adapter", {}).get("d_model", 256),
            "num_heads": cfg.get("adapter", {}).get("num_heads", 8),
            "ffn_dim": cfg.get("adapter", {}).get("ffn_dim", 512),
            "dropout": cfg.get("adapter", {}).get("dropout", 0.1),
        },
        lora_cfg={
            "rank": cfg.get("lora", {}).get("rank", 8),
            "alpha": cfg.get("lora", {}).get("alpha", 16.0),
            "dropout": cfg.get("lora", {}).get("dropout", 0.1),
            "target_modules": cfg.get("lora", {}).get("target_modules", ["q_proj", "v_proj"]),
        },
        head_cfg={
            "type": cfg.get("seg_head", {}).get("type", "conv1x1"),
            "hidden_dim": cfg.get("seg_head", {}).get("hidden_dim", 128),
        },
        use_adapter=cfg.get("adapter", {}).get("enabled", True),
        use_pca_residual=cfg.get("adapter", {}).get("use_residual", True),
        pca_model=pca_model,
    )
    return model
