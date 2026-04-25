#!/usr/bin/env python3
"""Generate a Colab-ready notebook for running one EuroSAT 5-fold split."""

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

cells.append(md("""# EuroSAT 1-Fold Colab Runner

This notebook is designed for **Google Colab** and runs **one fold end-to-end** from the EuroSAT 5-fold protocol.

Default target:

- method group: `["Ours"]`
- fold selection: one fold only, e.g. `FOLD_ID = 0`
- output style: one dedicated result directory per fold

Recommended usage:

1. Open this notebook in Colab
2. Set runtime to **GPU**
3. Mount Google Drive
4. Point the config to your repo, dataset, and CLIP checkpoint in Drive
5. Run all cells
6. Repeat with `FOLD_ID = 1,2,3,4`

This notebook is intentionally conservative:

- it stages the repo code to Colab local disk
- it copies the EuroSAT data and CLIP checkpoint to local disk before running
- it writes results locally first, then syncs them back to Drive
"""))

cells.append(md("""## 0. Install Dependencies

Colab already ships with PyTorch, but we still need a few Python packages plus the OpenAI CLIP package.
"""))

cells.append(code("""%%capture
!pip -q install ftfy regex tqdm rasterio scipy pandas h5py
!pip -q install git+https://github.com/openai/CLIP.git"""))

cells.append(md("""## 1. Mount Google Drive"""))

cells.append(code("""from google.colab import drive
drive.mount("/content/drive")"""))

cells.append(md("""## 2. Runtime Check

Make sure you selected **GPU** from `Runtime -> Change runtime type`.
"""))

cells.append(code("""!nvidia-smi || true"""))

cells.append(code("""import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
# NOTE: Do NOT call torch.set_grad_enabled(False) globally here.
# The "Ours" pipeline uses test-time optimization and needs gradients enabled.

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))"""))

cells.append(md("""## 3. Config

Choose **one** code source:

- `SETUP_MODE = "drive_repo"` if your repo already exists in Drive
- `SETUP_MODE = "git_clone"` if you want Colab to clone from GitHub

Recommended defaults:

- `METHODS = ["Ours"]`
- `FOLD_ID = 0`
- `COPY_DATA_TO_LOCAL = True`
- `BATCH_SIZE = 64`
- `NUM_WORKERS = 2`

Protocol for one selected `FOLD_ID`:

- query split: fold `FOLD_ID`
- gallery split: next fold in cyclic order
- train/reference split: remaining 3 folds
"""))

cells.append(code("""# -----------------------------
# Source code setup
# -----------------------------
SETUP_MODE = "drive_repo"   # "drive_repo" or "git_clone"

DRIVE_REPO_ROOT = Path("/content/drive/MyDrive/ACIVS_ThayBach")
GIT_REPO_URL = ""           # fill only if SETUP_MODE == "git_clone"
GIT_BRANCH = ""             # optional

# -----------------------------
# Data / checkpoint in Drive
# -----------------------------
DRIVE_EUROSAT_ROOT = DRIVE_REPO_ROOT / "data" / "EuroSAT_MS"
DRIVE_CLIP_CHECKPOINT = DRIVE_REPO_ROOT / "checkpoints" / "ViT-B-16.pt"

# -----------------------------
# Fold run configuration
# -----------------------------
METHODS = ["Ours"]          # e.g. ["Ours"] or ["RGB-CLIP", "Tip-Adapter", "RS-TransCLIP"]
FOLD_ID = 0
NUM_FOLDS = 5
SEED = 42

# Safe defaults for Colab
BATCH_SIZE = 64
NUM_WORKERS = 2
IMAGE_SIZE = 224
MICRO_BATCH_SIZE = 64
SHOW_PROGRESS = True

# Optional debug caps: keep None for full fold.
# MAX_GALLERY_PER_CLASS caps both held-out splits: query + gallery.
MAX_TRAIN_PER_CLASS = None
MAX_GALLERY_PER_CLASS = None

# -----------------------------
# Method hyperparameters
# -----------------------------
TIP_ADAPTER_ALPHA = 20.0
TIP_ADAPTER_BETA = 5.5
TIP_ADAPTER_CHUNK_SIZE = 256

RS_TRANSCLIP_ALPHA = 0.3
RS_TRANSCLIP_PATCH_POOL_SIZE = 4
RS_TRANSCLIP_AFFINITY_TOPK = 20
RS_TRANSCLIP_AFFINITY_CHUNK_SIZE = 256

OURS_SIGMA = 0.5
OURS_NUM_STEPS = 5
OURS_LR = 0.01
OURS_LAMBDA_M = 0.1
OURS_K = 5
OURS_GRAD_CLIP = 1.0
OURS_EARLY_STOP_TOL = 1e-6

EXTERNAL_METHOD_INPUTS = {}

# -----------------------------
# Local runtime layout
# -----------------------------
COPY_DATA_TO_LOCAL = True
RESET_LOCAL_RUN_ROOT = False

LOCAL_RUN_ROOT = Path("/content/acivs_fold_runtime")
LOCAL_PROJECT_ROOT = LOCAL_RUN_ROOT / "repo"
LOCAL_EUROSAT_ROOT = LOCAL_RUN_ROOT / "data" / "EuroSAT_MS"
LOCAL_CLIP_CHECKPOINT = LOCAL_RUN_ROOT / "checkpoints" / "ViT-B-16.pt"

method_tag = "_".join(method.lower().replace("-", "_").replace(" ", "_") for method in METHODS)
run_tag = f"fold_{FOLD_ID}_{method_tag}"
LOCAL_RESULTS_DIR = LOCAL_RUN_ROOT / "results" / run_tag
DRIVE_RESULTS_DIR = DRIVE_REPO_ROOT / "results" / "colab_single_fold" / run_tag

print("SETUP_MODE:", SETUP_MODE)
print("Methods:", METHODS)
print("Fold:", FOLD_ID)
print("Drive repo root:", DRIVE_REPO_ROOT)
print("Drive dataset root exists:", DRIVE_EUROSAT_ROOT.exists())
print("Drive checkpoint exists:", DRIVE_CLIP_CHECKPOINT.exists())
print("Debug caps: train=", MAX_TRAIN_PER_CLASS, "heldout=", MAX_GALLERY_PER_CLASS)
print("Drive results dir:", DRIVE_RESULTS_DIR)"""))

cells.append(md("""## 4. Stage Repo Code to Colab Local Disk

This keeps the Python package import path and output writes off Drive during execution.
"""))

cells.append(code("""def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


if RESET_LOCAL_RUN_ROOT and LOCAL_RUN_ROOT.exists():
    shutil.rmtree(LOCAL_RUN_ROOT)

ensure_clean_dir(LOCAL_RUN_ROOT)

if LOCAL_PROJECT_ROOT.exists():
    shutil.rmtree(LOCAL_PROJECT_ROOT)

if SETUP_MODE == "drive_repo":
    if not DRIVE_REPO_ROOT.exists():
        raise FileNotFoundError(f"Drive repo root not found: {DRIVE_REPO_ROOT}")

    shutil.copytree(
        DRIVE_REPO_ROOT,
        LOCAL_PROJECT_ROOT,
        ignore=shutil.ignore_patterns(
            ".git",
            "data",
            "checkpoints",
            "results",
            "acivs",
            "__pycache__",
            ".pytest_cache",
        ),
    )

elif SETUP_MODE == "git_clone":
    if not GIT_REPO_URL:
        raise ValueError("Set GIT_REPO_URL when SETUP_MODE == 'git_clone'.")

    clone_cmd = ["git", "clone", GIT_REPO_URL, str(LOCAL_PROJECT_ROOT)]
    if GIT_BRANCH:
        clone_cmd = ["git", "clone", "--branch", GIT_BRANCH, GIT_REPO_URL, str(LOCAL_PROJECT_ROOT)]
    subprocess.run(clone_cmd, check=True)

else:
    raise ValueError(f"Unsupported SETUP_MODE: {SETUP_MODE}")

if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))

print("Local project root:", LOCAL_PROJECT_ROOT)
print("Top-level files:", sorted(p.name for p in LOCAL_PROJECT_ROOT.iterdir())[:20])"""))

cells.append(md("""## 5. Stage Dataset and CLIP Checkpoint to Local Disk

For Colab, this is safer and usually faster than reading thousands of small files directly from Drive.
"""))

cells.append(code("""def copy_tree_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"Skip existing directory: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def copy_file_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"Skip existing file: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


if COPY_DATA_TO_LOCAL:
    if not DRIVE_EUROSAT_ROOT.exists():
        raise FileNotFoundError(f"EuroSAT root not found in Drive: {DRIVE_EUROSAT_ROOT}")
    if not DRIVE_CLIP_CHECKPOINT.exists():
        raise FileNotFoundError(f"CLIP checkpoint not found in Drive: {DRIVE_CLIP_CHECKPOINT}")

    copy_tree_if_needed(DRIVE_EUROSAT_ROOT, LOCAL_EUROSAT_ROOT)
    copy_file_if_needed(DRIVE_CLIP_CHECKPOINT, LOCAL_CLIP_CHECKPOINT)

    EUROSAT_ROOT = LOCAL_EUROSAT_ROOT
    CLIP_CHECKPOINT = LOCAL_CLIP_CHECKPOINT
else:
    EUROSAT_ROOT = DRIVE_EUROSAT_ROOT
    CLIP_CHECKPOINT = DRIVE_CLIP_CHECKPOINT

LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("EUROSAT_ROOT:", EUROSAT_ROOT)
print("CLIP_CHECKPOINT:", CLIP_CHECKPOINT)
print("LOCAL_RESULTS_DIR:", LOCAL_RESULTS_DIR)"""))

cells.append(md("""## 6. Import Project Code and Select Device"""))

cells.append(code("""from src.experiments.eurosat_5fold_cv import run_eurosat_5fold_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", DEVICE)"""))

cells.append(md("""## 7. Run One Fold

This executes exactly one fold: `fold_ids=[FOLD_ID]`.
"""))

cells.append(code("""result = run_eurosat_5fold_cv(
    root=EUROSAT_ROOT,
    clip_checkpoint=CLIP_CHECKPOINT,
    results_dir=LOCAL_RESULTS_DIR,
    methods=METHODS,
    external_method_inputs=EXTERNAL_METHOD_INPUTS,
    num_folds=NUM_FOLDS,
    fold_ids=[FOLD_ID],
    seed=SEED,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    image_size=IMAGE_SIZE,
    micro_batch_size=MICRO_BATCH_SIZE,
    show_progress=SHOW_PROGRESS,
    max_train_per_class=MAX_TRAIN_PER_CLASS,
    max_gallery_per_class=MAX_GALLERY_PER_CLASS,
    tip_adapter_alpha=TIP_ADAPTER_ALPHA,
    tip_adapter_beta=TIP_ADAPTER_BETA,
    tip_adapter_chunk_size=TIP_ADAPTER_CHUNK_SIZE,
    rs_transclip_alpha=RS_TRANSCLIP_ALPHA,
    rs_transclip_patch_pool_size=RS_TRANSCLIP_PATCH_POOL_SIZE,
    rs_transclip_affinity_topk=RS_TRANSCLIP_AFFINITY_TOPK,
    rs_transclip_affinity_chunk_size=RS_TRANSCLIP_AFFINITY_CHUNK_SIZE,
    ours_sigma=OURS_SIGMA,
    ours_num_steps=OURS_NUM_STEPS,
    ours_lr=OURS_LR,
    ours_lambda_m=OURS_LAMBDA_M,
    ours_k=OURS_K,
    ours_grad_clip=OURS_GRAD_CLIP,
    ours_early_stop_tol=OURS_EARLY_STOP_TOL,
    device=DEVICE,
)

print("Run completed.")
print("Manifest:", result["manifest_path"])"""))

cells.append(md("""## 8. Inspect Outputs"""))

cells.append(code("""result["fold_metrics_df"]"""))

cells.append(code("""result["comparison_df"]"""))

cells.append(code("""result["per_class_df"]"""))

cells.append(md("""## 9. Sync Results Back to Drive

Each fold gets its own dedicated result directory in Drive.
"""))

cells.append(code("""DRIVE_RESULTS_DIR.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR, dirs_exist_ok=True)

print("Synced results to:", DRIVE_RESULTS_DIR)
print("Files:")
for path in sorted(DRIVE_RESULTS_DIR.glob("*")):
    print(" -", path.name)"""))

cells.append(md("""## 10. Optional Cleanup

Run this only after you confirm the Drive sync is complete.
"""))

cells.append(code("""# shutil.rmtree(LOCAL_RUN_ROOT)
# print("Removed local runtime folder:", LOCAL_RUN_ROOT)"""))


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
        "colab": {
            "name": "09_eurosat_1fold_colab.ipynb",
            "provenance": [],
            "gpuType": "T4",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "notebooks" / "09_eurosat_1fold_colab.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2))
    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    main()
