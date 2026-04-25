#!/usr/bin/env python3
"""Generate notebooks/07_ndvi_baseline_check.ipynb programmatically."""

import json
import os


def make_cell(cell_type, source, metadata=None):
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(text):
    return make_cell("markdown", [line + "\n" for line in text.strip().split("\n")])


def code(text):
    lines = text.strip().split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("code", source)


cells = []

cells.append(md("""# NDVI Baseline Check

**Task**: Week 5, Task 2  
**Composite**: `NDVI`, `NDWI`, `SAVI`  
**Goal**: Convert multispectral Sentinel-2 inputs into a 3-channel spectral-index composite, then run the frozen CLIP image encoder for retrieval.

## Spectral Indices

- `NDVI = (NIR - Red) / (NIR + Red)`
- `NDWI = (Green - NIR) / (Green + NIR)`
- `SAVI = 1.5 * (NIR - Red) / (NIR + Red + 0.5)`

All three channels are clipped to `[-1, 1]` and mapped to `[0, 1]` before CLIP preprocessing.
"""))

cells.append(code("""import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.baselines.ndvi_baseline import (
    build_spectral_index_composite,
    load_openai_clip_model,
    run_bigearth_ndvi_baseline,
    run_eurosat_ndvi_baseline,
    save_csv_rows,
)
from src.datasets.eurosat import EuroSATMSDataset
from src.models.per_band_encoder import get_device

plt.rcParams["figure.dpi"] = 150
torch.set_grad_enabled(False)

print(f"Project root: {PROJECT_ROOT}")
print(f"PyTorch: {torch.__version__}")"""))

cells.append(code("""EUROSAT_ROOT = PROJECT_ROOT / "data" / "EuroSAT_MS"
BIGEARTH_ROOT = PROJECT_ROOT / "data" / "BigEarthNetS2"
CLIP_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ViT-B-16.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

EUROSAT_BATCH_SIZE = 128
BIGEARTH_BATCH_SIZE = 128
NUM_WORKERS = 0
SEED = 42

BIGEARTH_MAX_SAMPLES = 100_000
BIGEARTH_QUERY_SIZE = 1_000
BIGEARTH_GALLERY_SIZE = 10_000

DEVICE = get_device()
clip_model, _ = load_openai_clip_model(CLIP_CHECKPOINT, DEVICE)

print(f"Device: {DEVICE}")
print(f"EuroSAT root exists: {EUROSAT_ROOT.exists()}")
print(f"BigEarthNet root exists: {BIGEARTH_ROOT.exists()}")
print(f"CLIP checkpoint exists: {CLIP_CHECKPOINT.exists()}")"""))

cells.append(md("""## 1. Visual Check on One EuroSAT Sample"""))

cells.append(code("""eurosat_ds = EuroSATMSDataset(
    EUROSAT_ROOT,
    normalize=True,
    reflectance_scale=10_000.0,
    clamp_range=(0.0, 1.0),
)

sample = eurosat_ds[0]
image = sample["image"]

nir_idx = eurosat_ds.band_to_idx["B08"]
red_idx = eurosat_ds.band_to_idx["B04"]
green_idx = eurosat_ds.band_to_idx["B03"]
blue_idx = eurosat_ds.band_to_idx["B02"]

composite = build_spectral_index_composite(
    image,
    nir_idx=nir_idx,
    red_idx=red_idx,
    green_idx=green_idx,
)

rgb = image[[red_idx, green_idx, blue_idx]].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
ndvi = composite[0].numpy()
ndwi = composite[1].numpy()
savi = composite[2].numpy()

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
axes[0].imshow(rgb)
axes[0].set_title(f"RGB\\n{sample['label_name']}")
axes[1].imshow(ndvi, cmap="RdYlGn")
axes[1].set_title("NDVI")
axes[2].imshow(ndwi, cmap="Blues")
axes[2].set_title("NDWI")
axes[3].imshow(savi, cmap="YlGn")
axes[3].set_title("SAVI")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()"""))

cells.append(md("""## 2. Run EuroSAT NDVI Baseline"""))

cells.append(code("""eurosat_result = run_eurosat_ndvi_baseline(
    root=EUROSAT_ROOT,
    clip_model=clip_model,
    device=DEVICE,
    batch_size=EUROSAT_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=SEED,
    show_progress=True,
)

pd.DataFrame([eurosat_result["summary"]])"""))

cells.append(md("""## 3. Run BigEarthNet NDVI Baseline

The default configuration below runs the full `100k / 1k / 10k` protocol.  
If you only want a quick smoke test first, reduce `BIGEARTH_MAX_SAMPLES`, `BIGEARTH_QUERY_SIZE`, and `BIGEARTH_GALLERY_SIZE` in the config cell above.
"""))

cells.append(code("""bigearth_result = run_bigearth_ndvi_baseline(
    root=BIGEARTH_ROOT,
    clip_model=clip_model,
    device=DEVICE,
    batch_size=BIGEARTH_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    max_samples=BIGEARTH_MAX_SAMPLES,
    query_size=BIGEARTH_QUERY_SIZE,
    gallery_size=BIGEARTH_GALLERY_SIZE,
    seed=SEED,
    show_progress=True,
)

pd.DataFrame([bigearth_result["summary"]])"""))

cells.append(md("""## 4. Save Notebook Results"""))

cells.append(code("""summary_rows = [
    eurosat_result["summary"],
    bigearth_result["summary"],
]

save_csv_rows(summary_rows, RESULTS_DIR / "ndvi_baseline_results_notebook.csv")
save_csv_rows(
    eurosat_result["per_query_results"],
    RESULTS_DIR / "ndvi_baseline_eurosat_per_query_notebook.csv",
)
save_csv_rows(
    bigearth_result["per_query_results"],
    RESULTS_DIR / "ndvi_baseline_bigearth_per_query_notebook.csv",
)

pd.DataFrame(summary_rows)"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "notebooks",
        "07_ndvi_baseline_check.ipynb",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
