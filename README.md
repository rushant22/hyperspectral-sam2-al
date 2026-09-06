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

---

## Key Results

| Metric | Value |
|--------|-------|
| **Full Supervised Accuracy** | 90.95% OA / 81.51% mIoU (Pavia University) |
| **Active Learning (Entropy, 10 rounds)** | 74.74% mIoU with only 5.5% labels |
| **Annotation Savings** | 94.5% fewer labels vs. full supervision |
| **Active vs. Random Advantage** | +8.87% mIoU at round 10 |
| **Adapter Contribution** | +79.19% mIoU over no-adapter baseline |
| **Best Spectral Query Count** | M=8 queries → 82.35% mIoU / 92.20% OA |

See [RESULTS_SUMMARY.md](documentation/RESULTS_SUMMARY.md) for complete tables and analysis.

---

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

### 8. Generate Publication Figures

```bash
python evaluation/plots.py
```

---

## Visualization Dashboard

An interactive multi-panel web dashboard is included at `visualization/dashboard/`:

```bash
# Open directly in your browser:
# Simply double-click visualization/dashboard/index.html
# Or serve locally:
python -m http.server 8000 --directory visualization/dashboard
# Then open http://localhost:8000
```

The dashboard displays:
- **False-color composite** (PCA → RGB)
- **Segmentation map** (ground truth / predicted, toggleable)
- **Uncertainty heatmap** (BALD/entropy, inferno colormap)
- **AL query locations** + annotation-efficiency chart

> **Note:** The dashboard runs with synthetic demo data out-of-the-box. To load
> real experiment results, run `python visualization/export_results.py` after
> training to generate the JSON data files.

---

## Project Structure

```
CP/
├── configs/              # YAML hyperparameter configs
│   └── default.yaml      # Main experiment configuration
├── data/                 # Dataset download, loaders, transforms
│   ├── download.py       # Auto-download Indian Pines & Pavia University
│   ├── hsi_dataset.py    # PyTorch Dataset for HSI patches
│   └── transforms.py     # PCA, normalization, augmentation
├── models/               # Core model components
│   ├── spectral_adapter.py  # Spectral Cross-Attention Adapter (M queries)
│   ├── sam2_wrapper.py      # AdaptedSAM2 wrapper (backbone + adapter + head)
│   ├── lora.py              # LoRA injection for Hiera attention layers
│   └── losses.py            # Focal + Dice combined loss
├── active_learning/      # Uncertainty-driven query strategies
│   ├── loop.py           # Main AL training loop
│   ├── uncertainty.py    # BALD & Shannon entropy estimation (MC-Dropout)
│   └── strategies.py     # Query strategy implementations
├── evaluation/           # Metrics and plotting
│   ├── metrics.py        # mIoU, per-class IoU, OA computation
│   └── plots.py          # Publication figure generation
├── visualization/        # Web dashboard (simulated multi-panel display)
│   └── dashboard/        # HTML + CSS + JS interactive dashboard
├── scripts/              # Training and experiment entry points
│   ├── train_baseline.py
│   ├── train_adapter.py
│   ├── run_al_loop.py
│   └── run_ablations.py
├── documentation/        # Project documentation
│   ├── AL_HSI_SAM2_Project_Documentation.docx
│   └── RESULTS_SUMMARY.md
├── paper/                # IEEE-format paper and generated figures
│   └── figures/          # 4 publication-ready PNG figures
├── notebooks/            # Colab execution notebook
│   └── colab_runner.ipynb
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Datasets

| Dataset | Bands | Spatial Size | Classes | Source |
|---------|:-----:|:------------:|:-------:|--------|
| Indian Pines | 200 | 145×145 | 16 | [EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Pavia University | 103 | 610×340 | 9 | [EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |

---

## Key Design Decisions

- **SAM2.1 Hiera Base+** (~100M params): best compute/performance trade-off for ≤24GB VRAM
- **LoRA rank r=8, α=16**: applied to Q/V projections in all Hiera attention layers
- **Spectral Adapter**: M=12 learnable biochemical-response queries, cross-attention over spectral tokens, residual addition to SAM2 stem output
- **Active Learning**: BALD (via MC-Dropout, T=10 passes) + BADGE-inspired spatial diversity selection
- **Loss**: Focal (γ=2) + Dice, equally weighted

---

## Reproducibility

Every number in the paper comes from running the scripts above. Set
`seed: 42` in the config (default) for deterministic results. GPU
non-determinism may cause minor (<0.5% mIoU) variation across runs.

---

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
