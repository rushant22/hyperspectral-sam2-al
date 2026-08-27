# Active-Learning Hyperspectral Segmentation via Adapted SAM2

A framework for segmenting vegetation (and, by extension, invasive species) in
hyperspectral imagery by adapting a frozen SAM2 backbone with a custom Spectral
Cross-Attention Adapter, LoRA fine-tuning, and uncertainty-driven active
learning.

> **Scope note:** This codebase validates the methodology on standard HSI
> vegetation-classification benchmarks (Indian Pines, Pavia University).
> Invasive-species-specific validation requires labeled HSI datasets that are
> not yet publicly available — the framework is designed to be species-agnostic
> and transferable when such data becomes accessible.

## Quick Start

### 1. Environment Setup

```bash
# Create a virtual environment (Python 3.10+)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install SAM2 from source
pip install git+https://github.com/facebookresearch/sam2.git
```

### 2. Download Data

```bash
# Downloads Indian Pines and Pavia University datasets (~50 MB total)
python data/download.py --dataset all --output-dir ./datasets
```

### 3. Download SAM2 Checkpoint

```bash
# Download SAM2.1 Hiera Base+ checkpoint (~309 MB)
python data/download.py --sam2-checkpoint --output-dir ./checkpoints
```

### 4. Train Baseline (no adapter — PCA → SAM2)

```bash
python scripts/train_baseline.py --config configs/default.yaml
```

### 5. Train with Spectral Adapter + LoRA

```bash
python scripts/train_adapter.py --config configs/default.yaml
```

### 6. Run Active Learning Experiment

```bash
python scripts/run_al_loop.py --config configs/default.yaml
```

### 7. Run All Ablations

```bash
python scripts/run_ablations.py --config configs/default.yaml
```

## Project Structure

```
CP/
├── configs/          # YAML hyperparameter configs
├── data/             # Dataset download, loaders, transforms
├── models/           # Spectral adapter, LoRA, SAM2 wrapper, losses
├── active_learning/  # Uncertainty estimation, query strategies, AL loop
├── evaluation/       # Metrics (mIoU) and plotting
├── visualization/    # Web dashboard (simulated multi-panel display)
├── scripts/          # Training and experiment entry points
└── paper/            # IEEE-format paper and generated figures
```

## Datasets

| Dataset | Bands | Size | Classes | Source |
|---------|-------|------|---------|--------|
| Indian Pines | 200 | 145×145 | 16 | [EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Pavia University | 103 | 610×340 | 9 | [EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |

## Key Design Decisions

- **SAM2.1 Hiera Base+** (~100M params): best compute/performance trade-off for ≤24GB VRAM
- **LoRA rank r=8, α=16**: applied to Q/V projections in all Hiera attention layers
- **Spectral Adapter**: M=12 learnable biochemical-response queries, cross-attention over spectral tokens, residual addition to SAM2 stem output
- **Active Learning**: BALD (via MC-Dropout, T=10 passes) + BADGE-inspired spatial diversity selection
- **Loss**: Focal (γ=2) + Dice, equally weighted

## Reproducibility

Every number in the paper comes from running the scripts above. Set
`seed: 42` in the config (default) for deterministic results. GPU
non-determinism may cause minor (<0.5% mIoU) variation across runs.

## Citation

If you use this code, please cite:

```bibtex
@article{al_hsi_sam2_2026,
  title={Active-Learning Hyperspectral Vegetation Segmentation via Adapted SAM2},
  year={2026}
}
```

## License

MIT License
