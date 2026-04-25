#!/usr/bin/env python3
"""Generate notebooks/10_eurosat_1fold.ipynb programmatically."""

from __future__ import annotations

import json
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

cells.append(md("""# EuroSAT 1-Fold Runner

**Week 6 / Task 1**

This notebook is the **single-fold** version of `08_eurosat_5fold_cv.ipynb`.

Use it when you want to:

- run only one fold at a time
- inspect outputs before launching all 5 folds
- save time while debugging metrics or per-class behavior

The notebook still uses the same underlying experiment code and exports the same artifact types:

- `fold metrics CSV`
- `comparison table`
- `paired t-test table`
- `per-class summary table`

Notes:

- `comparison_df` still works for one fold, but `std` will usually be `0`
- `paired_ttest_df` is not statistically meaningful for one fold and will usually contain `NaN`
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

Set `FOLD_ID` manually to choose which fold to run.

Recommended usage:

1. Start with `METHODS = ["Ours"]` and one `FOLD_ID`
2. Inspect `fold_metrics_df`, `per_class_df`, and exported CSVs
3. If the outputs look correct, switch to another `FOLD_ID`
4. After all folds are verified, use notebook 08 for the full 5-fold run if needed

Notes:

- `MAX_GALLERY_PER_CLASS` caps both held-out splits: the query fold and the gallery fold
- each `FOLD_ID` writes to its own result directory to avoid overwriting previous runs
"""))

cells.append(code("""EUROSAT_ROOT = PROJECT_ROOT / "data" / "EuroSAT_MS"
CLIP_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "ViT-B-16.pt"

METHODS = [
    "RGB-CLIP",
    "PCA",
    "NDVI",
    "Tip-Adapter",
    "RS-TransCLIP",
    "Ours",
    # "DOFA",
]

FOLD_ID = 0
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
    #     "summary_csv": PROJECT_ROOT / "results" / "eurosat_1fold" / f"fold_{FOLD_ID}" / "external_templates" / "dofa_fold_metrics_template.csv",
    #     "per_query_csv": PROJECT_ROOT / "results" / "eurosat_1fold" / f"fold_{FOLD_ID}" / "external_templates" / "dofa_per_query_template.csv",
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

if not (0 <= FOLD_ID < NUM_FOLDS):
    raise ValueError(f"FOLD_ID must be in [0, {NUM_FOLDS - 1}], got {FOLD_ID}")

fold_tag = f"fold_{FOLD_ID}"
RESULTS_DIR = PROJECT_ROOT / "results" / "eurosat_1fold" / fold_tag

print(f"EuroSAT root exists: {EUROSAT_ROOT.exists()}")
print(f"CLIP checkpoint exists: {CLIP_CHECKPOINT.exists()}")
print(f"Results dir: {RESULTS_DIR}")
print(f"Methods: {METHODS}")
print(f"Fold: {FOLD_ID}")
print(f"Debug caps: train={MAX_TRAIN_PER_CLASS}, heldout={MAX_GALLERY_PER_CLASS}")"""))

cells.append(md("""## Run 1 Fold

This cell runs only `FOLD_ID` and writes the main artifacts under `results/eurosat_1fold/fold_<id>/`.
"""))

cells.append(code("""result = run_eurosat_5fold_cv(
    root=EUROSAT_ROOT,
    clip_checkpoint=CLIP_CHECKPOINT,
    results_dir=RESULTS_DIR,
    methods=METHODS,
    external_method_inputs=EXTERNAL_METHOD_INPUTS,
    num_folds=NUM_FOLDS,
    fold_ids=[FOLD_ID],
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

For the selected `FOLD_ID`, the protocol uses:

- 1 fold as **query**
- the next fold as **gallery**
- the remaining 3 folds as **train/reference**
"""))

cells.append(code("""result["split_manifest_df"]"""))

cells.append(md("""## Fold Metrics

Raw metrics for the selected fold and methods.
"""))

cells.append(code("""result["fold_metrics_df"]"""))

cells.append(md("""## Comparison Table

For one fold, this is mainly a convenient side-by-side method comparison.
"""))

cells.append(code("""result["comparison_df"]"""))

cells.append(md("""## Paired t-test

Shown for consistency with notebook 08, but not statistically meaningful with only one fold.
"""))

cells.append(code("""result["paired_ttest_df"]"""))

cells.append(md("""## Per-Class Table

This table helps inspect which classes are strong or weak on the selected fold.
"""))

cells.append(code("""result["per_class_df"]"""))

cells.append(md("""## Per-Query Preview

Quick sanity check on the first few query-level rows written to CSV.
"""))

cells.append(code("""result["per_query_df"].head(20)"""))

cells.append(md("""## Exported Files

Main CSV outputs inside this fold-specific directory:

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
    output_path = project_root / "notebooks" / "10_eurosat_1fold.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2))
    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    main()
