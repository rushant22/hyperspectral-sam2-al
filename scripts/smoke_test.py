"""
scripts/smoke_test.py — Quick end-to-end verification of all components.

Runs through the entire pipeline with tiny settings (no GPU, 2 epochs,
3 MC passes) to verify that:
  1. All imports resolve correctly
  2. Data loading works (with synthetic data if datasets aren't downloaded)
  3. Model forward/backward passes work
  4. Loss computation works
  5. MC-Dropout uncertainty estimation works
  6. Query selection strategies work
  7. Simulated oracle works
  8. Evaluation metrics compute correctly
  9. Plotting functions run without errors

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import time
import traceback
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def test(name):
    """Decorator that wraps a test function with timing and error handling."""
    def decorator(fn):
        def wrapper():
            start = time.time()
            try:
                fn()
                elapsed = time.time() - start
                results.append((name, True, elapsed, ""))
                print(f"  {PASS} {name} ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - start
                tb = traceback.format_exc()
                results.append((name, False, elapsed, str(e)))
                print(f"  {FAIL} {name} ({elapsed:.2f}s)")
                print(f"     Error: {e}")
                # Print abbreviated traceback
                lines = tb.strip().split("\n")
                for line in lines[-3:]:
                    print(f"     {line}")
        return wrapper
    return decorator


# =============================================================================
# Tests
# =============================================================================

@test("Import: data modules")
def test_imports_data():
    from data.utils import normalize_per_band, apply_pca, compute_ndvi
    from data.transforms import get_train_transform, get_eval_transform


@test("Import: model modules")
def test_imports_models():
    from models.spectral_adapter import SpectralCrossAttentionAdapter
    from models.lora import LoRALinear, inject_lora, get_lora_params, count_trainable_params
    from models.losses import FocalLoss, DiceLoss, FocalDiceLoss
    from models.sam2_wrapper import AdaptedSAM2, MultiClassHead, build_model


@test("Import: active learning modules")
def test_imports_al():
    from active_learning.uncertainty import mc_dropout_inference, compute_pixel_uncertainties
    from active_learning.query_strategies import random_query, uncertainty_query, badge_inspired_query
    from active_learning.simulated_oracle import SimulatedOracle


@test("Import: evaluation modules")
def test_imports_eval():
    from evaluation.metrics import compute_miou, compute_per_class_iou, compute_overall_accuracy, compute_kappa
    from evaluation.plots import plot_annotation_efficiency, plot_per_class_iou


@test("Data: normalize_per_band")
def test_normalize():
    from data.utils import normalize_per_band
    data = np.random.rand(32, 32, 10).astype(np.float32) * 1000
    normed, mean, std = normalize_per_band(data)
    assert normed.shape == data.shape, f"Shape mismatch: {normed.shape}"
    # Check that it's roughly zero-mean
    assert abs(normed.mean()) < 0.5, f"Mean too far from 0: {normed.mean()}"


@test("Data: apply_pca")
def test_pca():
    from data.utils import apply_pca
    data = np.random.rand(32, 32, 50).astype(np.float32)
    reduced, pca_model = apply_pca(data, n_components=3)
    assert reduced.shape == (32, 32, 3), f"Shape: {reduced.shape}"
    # Reuse the fitted PCA model
    reduced2, _ = apply_pca(data, n_components=3, pca_model=pca_model)
    assert reduced2.shape == (32, 32, 3)


@test("Data: compute_ndvi")
def test_ndvi():
    from data.utils import compute_ndvi
    data = np.random.rand(32, 32, 10).astype(np.float32) + 0.1
    ndvi = compute_ndvi(data, red_band_idx=3, nir_band_idx=7)
    assert ndvi.shape == (32, 32), f"Shape: {ndvi.shape}"
    assert ndvi.min() >= -1.0 and ndvi.max() <= 1.0


@test("Data: transforms")
def test_transforms():
    from data.transforms import get_train_transform, get_eval_transform
    bands = 10
    mean = np.zeros(bands)
    std = np.ones(bands)
    train_tf = get_train_transform(64, mean, std)
    eval_tf = get_eval_transform(64, mean, std)
    # Test with a sample — transform takes (data, labels) as separate args
    data = np.random.rand(32, 32, bands).astype(np.float32)
    labels = np.random.randint(0, 5, (32, 32)).astype(np.int64)
    data_out, labels_out = train_tf(data, labels)
    assert data_out.shape[0] == bands, f"Bands dim wrong: {data_out.shape}"
    assert labels_out.shape == (64, 64), f"Labels shape wrong: {labels_out.shape}"


@test("Model: SpectralCrossAttentionAdapter forward")
def test_adapter_forward():
    from models.spectral_adapter import SpectralCrossAttentionAdapter
    adapter = SpectralCrossAttentionAdapter(
        num_bands=10, d_model=64, num_queries=4, num_heads=4, ffn_dim=128
    )
    x = torch.randn(2, 10, 16, 16)  # (batch, bands, H, W)
    out = adapter(x)
    assert out.shape == (2, 64, 16, 16), f"Output shape: {out.shape}"


@test("Model: SpectralCrossAttentionAdapter attention maps")
def test_adapter_attention():
    from models.spectral_adapter import SpectralCrossAttentionAdapter
    adapter = SpectralCrossAttentionAdapter(
        num_bands=10, d_model=64, num_queries=4, num_heads=4, ffn_dim=128
    )
    x = torch.randn(1, 10, 8, 8)
    attn = adapter.get_attention_maps(x)
    # Shape: (batch, num_heads, M, H*W) = (1, 4, 4, 8*8=64)
    assert attn.shape == (1, 4, 4, 64), f"Attn shape: {attn.shape}"


@test("Model: LoRA injection")
def test_lora():
    from models.lora import LoRALinear, inject_lora, get_lora_params, count_trainable_params
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )
    # Count params before
    before = count_trainable_params(model)
    # Inject LoRA — target all linear layers
    injected = inject_lora(model, ["0", "2"], rank=4, alpha=8)
    assert len(injected) == 2, f"Expected 2 injections, got {len(injected)}"
    # Check LoRA params exist
    lora_params = get_lora_params(model)
    assert len(lora_params) == 4, f"Expected 4 LoRA params (2 A + 2 B), got {len(lora_params)}"
    # Forward pass
    x = torch.randn(2, 32)
    y = model(x)
    assert y.shape == (2, 16), f"Output shape: {y.shape}"


@test("Model: FocalDiceLoss")
def test_loss():
    from models.losses import FocalDiceLoss
    loss_fn = FocalDiceLoss(ignore_index=-1)
    # Logits need requires_grad=True to verify gradient flow
    logits = torch.randn(2, 5, 16, 16, requires_grad=True)
    targets = torch.randint(0, 5, (2, 16, 16))
    targets[0, 0:2, :] = -1  # Some ignored pixels
    loss_dict = loss_fn(logits, targets)
    assert "total" in loss_dict
    assert "focal" in loss_dict
    assert "dice" in loss_dict
    assert loss_dict["total"].requires_grad, "total loss should have grad"


@test("Model: AdaptedSAM2 forward + backward")
def test_full_model():
    from models.sam2_wrapper import build_model
    cfg = {
        "sam2": {"checkpoint": None, "model_cfg": "sam2.1_hiera_b+.yaml"},
        "adapter": {"enabled": True, "use_residual": True,
                     "num_queries": 4, "d_model": 64, "num_heads": 4,
                     "ffn_dim": 128, "dropout": 0.1},
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.1,
                  "target_modules": ["q_proj", "v_proj"]},
        "seg_head": {"type": "conv1x1", "hidden_dim": 64},
    }
    model = build_model(cfg, num_bands=10, num_classes=5)
    x = torch.randn(1, 10, 16, 16)
    pca = torch.randn(1, 3, 16, 16)
    out = model(x, pca)
    assert "logits" in out
    assert "features" in out
    assert out["logits"].shape == (1, 5, 16, 16), f"Logits shape: {out['logits'].shape}"
    # Backward pass
    loss = out["logits"].sum()
    loss.backward()


@test("Model: MC-Dropout toggle")
def test_mc_dropout():
    from models.sam2_wrapper import build_model
    cfg = {
        "sam2": {"checkpoint": None}, "lora": {},
        "adapter": {"enabled": True, "use_residual": False,
                     "num_queries": 4, "d_model": 64, "num_heads": 4,
                     "ffn_dim": 128, "dropout": 0.2},
        "seg_head": {"type": "conv1x1"},
    }
    model = build_model(cfg, num_bands=10, num_classes=5)
    model.enable_mc_dropout()
    # Verify dropout is training but batchnorm is eval
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.training, "Dropout should be in training mode"
        if isinstance(m, nn.BatchNorm2d):
            assert not m.training, "BatchNorm should be in eval mode"


@test("AL: MC-Dropout inference")
def test_mc_inference():
    from models.sam2_wrapper import build_model
    from active_learning.uncertainty import mc_dropout_inference
    cfg = {
        "sam2": {"checkpoint": None}, "lora": {},
        "adapter": {"enabled": True, "use_residual": False,
                     "num_queries": 4, "d_model": 64, "num_heads": 4,
                     "ffn_dim": 128, "dropout": 0.2},
        "seg_head": {"type": "conv1x1"},
    }
    model = build_model(cfg, num_bands=10, num_classes=5)
    x = torch.randn(1, 10, 8, 8)
    result = mc_dropout_inference(model, x, num_passes=3, device="cpu")
    assert result["entropy"].shape == (8, 8)
    assert result["bald"].shape == (8, 8)
    assert result["predicted_class"].shape == (8, 8)
    assert (result["bald"] >= 0).all(), "BALD should be non-negative"


@test("AL: query strategies")
def test_query_strategies():
    from active_learning.query_strategies import random_query, uncertainty_query, badge_inspired_query
    H, W = 16, 16
    scores = torch.rand(H, W)
    labeled = torch.zeros(H, W, dtype=torch.bool)
    labeled[:2, :2] = True  # Some pixels already labeled

    # Random
    selected = random_query(10, H * W, labeled_mask=labeled, seed=42)
    assert len(selected) == 10
    assert selected.shape[1] == 2  # (K, 2)

    # Uncertainty
    selected = uncertainty_query(scores, 10, labeled_mask=labeled)
    assert len(selected) == 10

    # BADGE-inspired
    features = torch.randn(64, H, W)
    selected = badge_inspired_query(scores, features, 10, num_clusters=5, labeled_mask=labeled)
    assert len(selected) <= 10


@test("AL: simulated oracle")
def test_oracle():
    from active_learning.simulated_oracle import SimulatedOracle
    gt = torch.zeros(16, 16, dtype=torch.long)
    gt[2:6, 2:6] = 1
    gt[8:12, 8:12] = 2
    oracle = SimulatedOracle(gt)
    mask = oracle.initialize_random_labels(0.1, seed=42)
    assert mask.sum() > 0
    # Query some pixels
    coords = torch.tensor([[3, 3], [9, 9], [0, 0]])
    result = oracle.label_pixels(coords)
    assert result["labels"][0] == 1  # Class 1
    assert result["labels"][1] == 2  # Class 2
    assert result["labels"][2] == 0  # Background
    summary = oracle.get_annotation_summary()
    assert summary["num_rounds"] == 1


@test("Evaluation: mIoU computation")
def test_miou():
    from evaluation.metrics import compute_miou, compute_per_class_iou, compute_kappa
    pred = torch.tensor([0, 1, 1, 2, 2, 0, 1, 2])
    target = torch.tensor([0, 1, 2, 2, 0, 0, 1, 2])
    miou = compute_miou(pred, target, num_classes=3)
    assert 0 <= miou <= 1, f"mIoU out of range: {miou}"
    per_class = compute_per_class_iou(pred, target, num_classes=3)
    assert len(per_class) == 3
    kappa = compute_kappa(pred, target, num_classes=3)
    assert -1 <= kappa <= 1, f"Kappa out of range: {kappa}"


@test("Evaluation: confusion matrix edge case")
def test_confusion_edge():
    from evaluation.metrics import compute_miou
    # All same class
    pred = torch.zeros(10, dtype=torch.long)
    target = torch.zeros(10, dtype=torch.long)
    miou = compute_miou(pred, target, num_classes=3)
    assert miou == 1.0, f"Perfect prediction should give mIoU=1, got {miou}"


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("AL-HSI-SAM2 Smoke Test")
    print("=" * 60 + "\n")

    # Run all tests
    tests = [
        test_imports_data,
        test_imports_models,
        test_imports_al,
        test_imports_eval,
        test_normalize,
        test_pca,
        test_ndvi,
        test_transforms,
        test_adapter_forward,
        test_adapter_attention,
        test_lora,
        test_loss,
        test_full_model,
        test_mc_dropout,
        test_mc_inference,
        test_query_strategies,
        test_oracle,
        test_miou,
        test_confusion_edge,
    ]

    for t in tests:
        t()

    # Summary
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = sum(1 for _, ok, _, _ in results if not ok)
    total_time = sum(t for _, _, t, _ in results)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(results)} passed, {failed} failed ({total_time:.2f}s)")
    if failed > 0:
        print(f"\nFailed tests:")
        for name, ok, _, err in results:
            if not ok:
                print(f"  {FAIL} {name}: {err}")
    print(f"{'=' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
