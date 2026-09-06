"""
add_export_cells.py — Patch colab_runner.ipynb to add dashboard export cells.
Run this once: python scripts/add_export_cells.py
"""
import json, os, sys

NB_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'colab_runner.ipynb')
NB_PATH = os.path.normpath(NB_PATH)

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

NEW_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 9B: Generate Real Dashboard Data (GPU Inference)\n",
            "\n",
            "Runs the trained model on the full Pavia University dataset and exports the 5 JSON files the interactive web dashboard needs:\n",
            "- `false_color.json` — PCA → RGB false-color image\n",
            "- `uncertainty_map.json` — Real BALD uncertainty heatmap\n",
            "- `segmentation_map.json` — Ground truth + model predictions\n",
            "- `query_history.json` — AL query coordinates per round\n",
            "- `metrics_summary.json` — Per-round mIoU for all 3 strategies\n",
            "\n",
            "**Run after training completes. Takes ~2–5 min on T4 GPU.**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys, os\n",
            "\n",
            "# Ensure scikit-image is available for image downsampling\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-image'], check=True)\n",
            "\n",
            "# Run the dashboard data export script\n",
            "!python scripts/generate_dashboard_data.py\n",
            "\n",
            "# Verify output\n",
            "data_dir = 'visualization/dashboard/data'\n",
            "if os.path.exists(data_dir):\n",
            "    files_generated = os.listdir(data_dir)\n",
            "    total_kb = sum(os.path.getsize(os.path.join(data_dir, f)) for f in files_generated) / 1024\n",
            "    print(f'\\n✅ Dashboard data ready: {files_generated}')\n",
            "    print(f'   Total size: {total_kb:.1f} KB')\n",
            "else:\n",
            "    print('❌ Export failed — check output above')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 9C: Download Dashboard Data\n",
            "\n",
            "Download the 5 JSON files and extract them into `visualization/dashboard/data/` in your local project folder.\n",
            "The dashboard will then **auto-load your real experiment results** when you open `index.html`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from google.colab import files\n",
            "import os\n",
            "\n",
            "# Zip just the dashboard JSON files\n",
            "!zip -j dashboard_data.zip visualization/dashboard/data/*.json\n",
            "!unzip -l dashboard_data.zip\n",
            "\n",
            "# Download\n",
            "files.download('dashboard_data.zip')\n",
            "print('\\n📥 Downloaded dashboard_data.zip')\n",
            "print('👉 On your local machine:')\n",
            "print('   1. Extract dashboard_data.zip')\n",
            "print('   2. Copy all .json files into: visualization/dashboard/data/')\n",
            "print('   3. Open visualization/dashboard/index.html in your browser')\n",
            "print('   The dashboard will auto-load your real data!')"
        ]
    },
]

# Find the last cell index and append new cells before closing
nb['cells'].extend(NEW_CELLS)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Added %d cells to %s" % (len(NEW_CELLS), NB_PATH))
print("     Steps 9B and 9C added: GPU inference export + dashboard data download")
