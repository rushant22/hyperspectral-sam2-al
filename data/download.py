"""
data/download.py — Download public hyperspectral datasets and SAM2 checkpoints.

Supported datasets:
  - Indian Pines (AVIRIS, 200 bands, 145×145, 16 classes)
  - Pavia University (ROSIS, 103 bands, 610×340, 9 classes)

Supported checkpoints:
  - SAM2.1 Hiera Base+ (~309 MB)

All downloads are idempotent — re-running skips files that already exist.
"""

import os
import sys
import argparse
import hashlib
import requests
from tqdm import tqdm


# =============================================================================
# Dataset URLs
# =============================================================================
# These are hosted on widely-used academic mirrors. If a URL goes dead, the
# fallback sources listed in comments can be used instead.

DATASETS = {
    "indian_pines": {
        "data": {
            "url": "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_corrected.mat",
            "filename": "Indian_pines_corrected.mat",
            "description": "Indian Pines corrected HSI cube (200 bands, 145x145)",
        },
        "labels": {
            "url": "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_gt.mat",
            "filename": "Indian_pines_gt.mat",
            "description": "Indian Pines ground truth labels (16 classes)",
        },
    },
    "pavia": {
        "data": {
            "url": "https://huggingface.co/datasets/danaroth/pavia/resolve/main/PaviaU.mat",
            "filename": "PaviaU.mat",
            "description": "Pavia University HSI cube (103 bands, 610x340)",
        },
        "labels": {
            "url": "https://huggingface.co/datasets/danaroth/pavia/resolve/main/PaviaU_gt.mat",
            "filename": "PaviaU_gt.mat",
            "description": "Pavia University ground truth labels (9 classes)",
        },
    },
}

SAM2_CHECKPOINT = {
    "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "filename": "sam2.1_hiera_base_plus.pt",
    "description": "SAM2.1 Hiera Base+ checkpoint (~309 MB)",
}


def download_file(url: str, dest_path: str, description: str = "") -> None:
    """
    Download a file with a progress bar. Skips if file already exists.

    Args:
        url: Direct download URL.
        dest_path: Full local path to save the file.
        description: Human-readable label for the progress bar.
    """
    if os.path.exists(dest_path):
        print(f"  [SKIP] Already exists: {dest_path}")
        return

    # Create parent directories if needed
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"  [DOWNLOAD] {description or url}")
    print(f"    -> {dest_path}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, stream=True, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Try manually downloading from: {url}")
        return

    # Get total file size for progress bar (if server provides it)
    total_size = int(response.headers.get("content-length", 0))

    # Write to a temporary file first, then rename — prevents partial downloads
    # from being mistaken for complete files on re-run
    temp_path = dest_path + ".partial"
    with open(temp_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="    ") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Rename temp file to final name (atomic on most filesystems)
    os.rename(temp_path, dest_path)
    print(f"  [DONE] Saved: {dest_path}")


def download_dataset(name: str, output_dir: str) -> None:
    """Download a specific dataset (data + labels)."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    dataset = DATASETS[name]
    dataset_dir = os.path.join(output_dir, name)
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"{'='*60}")

    for key in ["data", "labels"]:
        info = dataset[key]
        dest = os.path.join(dataset_dir, info["filename"])
        download_file(info["url"], dest, info["description"])


def download_sam2_checkpoint(output_dir: str) -> None:
    """Download the SAM2.1 Hiera Base+ checkpoint."""
    print(f"\n{'='*60}")
    print("Downloading: SAM2.1 Hiera Base+ checkpoint")
    print(f"{'='*60}")
    dest = os.path.join(output_dir, SAM2_CHECKPOINT["filename"])
    download_file(SAM2_CHECKPOINT["url"], dest, SAM2_CHECKPOINT["description"])


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and checkpoints for AL-HSI-SAM2."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()) + ["all"],
        help="Which dataset to download. Use 'all' for all datasets.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        action="store_true",
        help="Download the SAM2.1 Hiera Base+ checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Root directory for downloaded files.",
    )
    args = parser.parse_args()

    # If nothing specified, download everything
    if args.dataset is None and not args.sam2_checkpoint:
        print("No --dataset or --sam2-checkpoint specified. Downloading everything.")
        args.dataset = "all"
        args.sam2_checkpoint = True

    # Download datasets
    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(name, args.output_dir)
    elif args.dataset is not None:
        download_dataset(args.dataset, args.output_dir)

    # Download SAM2 checkpoint
    if args.sam2_checkpoint:
        checkpoint_dir = os.path.join(
            os.path.dirname(args.output_dir), "checkpoints"
        )
        download_sam2_checkpoint(checkpoint_dir)

    print(f"\n{'='*60}")
    print("All downloads complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
