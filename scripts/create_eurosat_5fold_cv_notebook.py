#!/usr/bin/env python3
"""Generate notebooks/08_eurosat_5fold_cv.ipynb programmatically."""

from __future__ import annotations

import json
import os
from pathlib import Path


def make_cell(cell_type: str, source, metadata=None):
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(text: str):
    return make_cell("markdown", [line + "\n" for line in text.strip().split("\n")])


def code(text: str):
    lines = text.strip().split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("code", source)


cells = []

cells.append(md("""# EuroSAT 5-Fold CV

**Week 6 / Task 1**

This notebook runs a stratified **5-fold cross-validation** benchmark on **EuroSAT-MS** and exports:

- `fold metrics CSV`
- `mean±std comparison table`
- `paired t-test p-values`
- `per-class summary table`

The notebook evaluates the local methods already supported in this repo:

- `RGB-CLIP`
- `PCA`
- `NDVI`
- `Tip-Adapter`
- `RS-TransCLIP`
- `Ours`

`DOFA` is exposed through **external CSV templates** because the repo currently does not contain a DOFA model implementation.
"""))

cells.append(code("""import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.eurosat_5fold_cv import (
    DEFAULT_ALL_METHODS,
    run_eurosat_5fold_cv,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
# NOTE: Do NOT call torch.set_grad_enabled(False) globally here.
# The "Ours" pipeline runs test-time optimization with Adam and needs grads.
# Encoding paths already use inference/no_grad internally where appropriate.

print(f"Project root: {PROJECT_ROOT}")
print(f"PyTorch: {torch.__version__}")"""))

cells.append(md("""## Config

Use `METHODS` and `FOLD_IDS` to control runtime:

- Fast smoke test: `METHODS = ["RGB-CLIP", "PCA"]`, `FOLD_IDS = [0]`
- Even faster smoke test: also set `MAX_TRAIN_PER_CLASS = 24`, `MAX_GALLERY_PER_CLASS = 8`
- `MAX_GALLERY_PER_CLASS` caps both held-out splits: the query fold and the gallery fold
- Full local run: keep the default method list and set `FOLD_IDS = None`
- Add `DOFA`: fill the template CSVs printed below, then set `EXTERNAL_METHOD_INPUTS["DOFA"]`
"""))

cells.append(code("""EUROSAT_ROOT = PROJECT_ROOT / "data" / "EuroSAT_MS"
CLIP_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ViT-B-16.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "eurosat_5fold_cv"

METHODS = [
    "RGB-CLIP",
    "PCA",
    "NDVI",
    "Tip-Adapter",
    "RS-TransCLIP",
    "Ours",
    # "DOFA",
]

FOLD_IDS = None
NUM_FOLDS = 5
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0
SHOW_PROGRESS = True
# MAX_GALLERY_PER_CLASS caps both held-out splits: query + gallery.
MAX_TRAIN_PER_CLASS = None
MAX_GALLERY_PER_CLASS = None

EXTERNAL_METHOD_INPUTS = {
    # "DOFA": {
    #     "summary_csv": RESULTS_DIR / "external_templates" / "dofa_fold_metrics_template.csv",
    #     "per_query_csv": RESULTS_DIR / "external_templates" / "dofa_per_query_template.csv",
    # }
}

TIP_ADAPTER_ALPHA = 20.0
TIP_ADAPTER_BETA = 5.5
RS_TRANSCLIP_ALPHA = 0.3

OURS_SIGMA = 0.5
OURS_NUM_STEPS = 5
OURS_LR = 0.01
OURS_LAMBDA_M = 0.1
OURS_K = 5

print(f"EuroSAT root exists: {EUROSAT_ROOT.exists()}")
print(f"CLIP checkpoint exists: {CLIP_CHECKPOINT.exists()}")
print(f"Results dir: {RESULTS_DIR}")
print(f"Methods: {METHODS}")
print(f"Folds: {FOLD_IDS if FOLD_IDS is not None else 'all'}")
print(f"Debug caps: train={MAX_TRAIN_PER_CLASS}, heldout={MAX_GALLERY_PER_CLASS}")"""))

cells.append(md("""## Run 5-Fold CV

This cell writes the main artifacts under `results/eurosat_5fold_cv/`.
"""))

cells.append(code("""result = run_eurosat_5fold_cv(
    root=EUROSAT_ROOT,
    clip_checkpoint=CLIP_CHECKPOINT,
    results_dir=RESULTS_DIR,
    methods=METHODS,
    external_method_inputs=EXTERNAL_METHOD_INPUTS,
    num_folds=NUM_FOLDS,
    fold_ids=FOLD_IDS,
    seed=SEED,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    show_progress=SHOW_PROGRESS,
    max_train_per_class=MAX_TRAIN_PER_CLASS,
    max_gallery_per_class=MAX_GALLERY_PER_CLASS,
    tip_adapter_alpha=TIP_ADAPTER_ALPHA,
    tip_adapter_beta=TIP_ADAPTER_BETA,
    rs_transclip_alpha=RS_TRANSCLIP_ALPHA,
    ours_sigma=OURS_SIGMA,
    ours_num_steps=OURS_NUM_STEPS,
    ours_lr=OURS_LR,
    ours_lambda_m=OURS_LAMBDA_M,
    ours_k=OURS_K,
)

print(f"Manifest: {result['manifest_path']}")
print("External templates:")
for method, paths in result["external_templates"].items():
    print(f"  {method}:")
    for name, path in paths.items():
        print(f"    {name}: {path}")"""))

cells.append(md("""## Split Manifest

Each run uses:

- 1 fold as **query**
- the next fold as **gallery**
- the remaining 3 folds as **train/reference**

This keeps query/gallery disjoint and makes the benchmark a real **image-to-image retrieval** protocol.
"""))

cells.append(code("""result["split_manifest_df"]"""))

cells.append(md("""## Fold Metrics

Raw metrics per fold and per method.
"""))

cells.append(code("""result["fold_metrics_df"]"""))

cells.append(md("""## Mean ± Std Comparison

Primary table for the Week 6 summary.
"""))

cells.append(code("""result["comparison_df"]"""))

cells.append(md("""## Paired t-test

By default this compares every method against `Ours` on the matched fold-level scores.
"""))

cells.append(code("""result["paired_ttest_df"]"""))

cells.append(md("""## Per-Class Table

This table aggregates metrics over the **real query images** of each class across the selected folds.
"""))

cells.append(code("""result["per_class_df"]"""))

cells.append(md("""## Exported Files

Main CSV outputs:

- `eurosat_5fold_fold_metrics.csv`
- `eurosat_5fold_comparison.csv`
- `eurosat_5fold_paired_ttest.csv`
- `eurosat_5fold_per_class.csv`
- `eurosat_5fold_per_query.csv`
- `eurosat_5fold_split_manifest.csv`
"""))

cells.append(code("""sorted(str(path) for path in RESULTS_DIR.glob("*.csv"))"""))


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


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "notebooks" / "08_eurosat_5fold_cv.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2))
    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    main()
