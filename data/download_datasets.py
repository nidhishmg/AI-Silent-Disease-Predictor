"""
download_datasets.py — Download real biomedical datasets for training.

Downloads two publicly available clinical datasets:
    1. UCI Heart Disease  (303 records, 14 clinical attributes)
    2. PIMA Diabetes       (768 records, 8 metabolic attributes)

Usage::

    python data/download_datasets.py

Files are saved to ``data/datasets/``.
"""

from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Dataset URLs (reliable public mirrors)
# ---------------------------------------------------------------------------
DATASETS = {
    "heart.csv": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/heart.csv",
        "description": "UCI Heart Disease — 303 clinical records",
        "rows": 303,
        "source": "UCI Machine Learning Repository",
    },
    "diabetes.csv": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "description": "PIMA Indians Diabetes — 768 metabolic records",
        "rows": 768,
        "source": "National Institute of Diabetes and Digestive and Kidney Diseases",
    },
}

# ---------------------------------------------------------------------------
# Target directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")


def download_all(force: bool = False) -> dict[str, bool]:
    """Download all datasets.  Returns {filename: success}."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    results: dict[str, bool] = {}

    for filename, meta in DATASETS.items():
        dest = os.path.join(DATASET_DIR, filename)

        if os.path.isfile(dest) and not force:
            size = os.path.getsize(dest)
            print(f"  ✓ {filename} already exists ({size:,} bytes) — skipping")
            results[filename] = True
            continue

        print(f"  ↓ Downloading {meta['description']} ...")
        print(f"    Source: {meta['url']}")

        try:
            urllib.request.urlretrieve(meta["url"], dest)
            size = os.path.getsize(dest)
            print(f"  ✓ Saved {filename} ({size:,} bytes)")
            results[filename] = True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"  ✗ Failed to download {filename}: {exc}")
            results[filename] = False

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AI Silent Disease Predictor — Dataset Downloader")
    print("=" * 60)
    print(f"Target directory: {DATASET_DIR}\n")

    force = "--force" in sys.argv
    if force:
        print("(--force mode: re-downloading all files)\n")

    results = download_all(force=force)

    print("\n" + "-" * 40)
    total = len(results)
    ok = sum(results.values())
    print(f"Downloaded: {ok}/{total} datasets")

    if ok < total:
        print("⚠  Some downloads failed. Training will fall back to synthetic data.")
        sys.exit(1)
    else:
        print("✓ All datasets ready. Run  python train_model.py  to train.")
