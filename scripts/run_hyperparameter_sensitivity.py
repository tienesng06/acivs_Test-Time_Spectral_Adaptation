#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis — Task 12, Week 7
======================================================

Sweeps 6 hyperparameters of the 13-band "Ours" pipeline on EuroSAT fold 0:
  σ  (query alignment temperature)  : [0.1, 0.3, 0.5, 0.7, 1.0]
  τ  (affinity softmax temperature)  : [0.05, 0.1, 0.2, 0.5]
  λ_m (manifold loss weight)         : [0.01, 0.05, 0.1, 0.2]
  k  (k-NN neighbors)                : [3, 5, 7, 10]
  num_steps (optimization steps)     : [1, 3, 5, 7, 10]
  lr (learning rate)                 : [0.005, 0.01, 0.02, 0.05]

Strategy: encode per-band embeddings ONCE, then sweep fusion params only.
This reduces wall-clock time from ~30h to ~4-5h.

Outputs (results/hyperparameter_sensitivity/):
  sensitivity_raw.csv       — 26 rows, one per (param, value)
  sensitivity_summary.csv   — 6 rows, recommended value per param
  sensitivity_<param>.png   — 6 individual curves (300 DPI)
  sensitivity_all_6.png     — Figure 5 paper: 2×3 grid
  sensitivity_manifest.json — experiment metadata

Usage:
  python scripts/run_hyperparameter_sensitivity.py
  python scripts/run_hyperparameter_sensitivity.py --param k --smoke-test
  python scripts/run_hyperparameter_sensitivity.py --fold-id 0 --output-dir results/sens
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# Make sure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.eurosat import EuroSATMSDataset
from src.experiments.eurosat_5fold_cv import (
    make_stratified_kfold_buckets,
    build_retrieval_fold_splits,
    encode_class_text_features,
)
from src.models.per_band_encoder import encode_multispectral_batch
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline
from src.utils.metrics import evaluate_text_to_image_retrieval
from src.utils.shared import get_device, load_openai_clip_model, save_csv_rows

# ============================================================
# Hyperparameter grid (matches ACIVS_2026_Implementation_Plan Day 46-47)
# ============================================================

SENSITIVITY_GRID: Dict[str, List[Any]] = {
    "sigma":     [0.1, 0.3, 0.5, 0.7, 1.0],
    "tau":       [0.05, 0.1, 0.2, 0.5],
    "lambda_m":  [0.01, 0.05, 0.1, 0.2],
    "k":         [3, 5, 7, 10],
    "num_steps": [1, 3, 5, 7, 10],
    "lr":        [0.005, 0.01, 0.02, 0.05],
}

# Smoke test: only 2 values per param
SMOKE_GRID: Dict[str, List[Any]] = {
    "sigma":     [0.1, 0.5],
    "tau":       [0.05, 0.2],
    "lambda_m":  [0.01, 0.1],
    "k":         [3, 7],
    "num_steps": [1, 5],
    "lr":        [0.005, 0.02],
}

# Fixed optimal values (baseline for "fix all others, vary one")
OPTIMAL_FIXED: Dict[str, Any] = {
    "sigma":     0.5,
    "tau":       None,   # None = symmetric normalize (default behavior)
    "lambda_m":  0.1,
    "k":         5,
    "num_steps": 5,
    "lr":        0.01,
}

# Human-readable labels for plots
PARAM_LABELS: Dict[str, str] = {
    "sigma":     r"$\sigma$ (query temperature)",
    "tau":       r"$\tau$ (affinity temperature)",
    "lambda_m":  r"$\lambda_m$ (manifold weight)",
    "k":         r"$k$ (k-NN neighbors)",
    "num_steps": "Optimization steps",
    "lr":        "Learning rate",
}

OPTIMAL_VALUES: Dict[str, Any] = {
    "sigma":     0.5,
    "tau":       0.1,
    "lambda_m":  0.1,
    "k":         5,
    "num_steps": 5,
    "lr":        0.01,
}

# Plot style
CURVE_COLOR   = "#2196F3"
OPTIMAL_COLOR = "#FF5722"
SHADE_COLOR   = "#E0E0E0"


# ============================================================
# Data helpers
# ============================================================

def _build_loader(
    dataset: EuroSATMSDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


@torch.inference_mode()
def encode_band_embeddings(
    loader: DataLoader,
    clip_model,
    device: torch.device,
    *,
    micro_batch_size: int = 64,
    desc: str = "Encoding",
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Encode all samples in loader → (N, B, D) band embeddings + labels."""
    clip_model.eval()
    band_chunks: List[torch.Tensor] = []
    label_list:  List[torch.Tensor] = []

    iterator: Iterable = tqdm(loader, desc=desc) if show_progress else loader
    for batch in iterator:
        band_emb = encode_multispectral_batch(
            batch["image"], clip_model, device=device,
            micro_batch_size=micro_batch_size,
        )
        band_chunks.append(band_emb.cpu())
        label_list.append(batch["label"].cpu())

    return {
        "band_embeddings": torch.cat(band_chunks, dim=0),   # (N, B, D)
        "labels":          torch.cat(label_list, dim=0),    # (N,)
    }


# ============================================================
# Fusion helpers
# ============================================================

def fuse_split(
    band_embeddings: torch.Tensor,  # (N, B, D)
    labels: torch.Tensor,           # (N,)
    class_text_features: torch.Tensor,  # (C, D)
    pipeline: MultispectralRetrievalPipeline,
    *,
    desc: str = "Fusing",
    show_progress: bool = True,
) -> torch.Tensor:
    """Fuse (N, B, D) band embeddings → (N, D) fused embeddings."""
    N = band_embeddings.shape[0]
    D = band_embeddings.shape[2]
    fused = torch.zeros(N, D)

    query_tf = F.normalize(class_text_features.float().cpu(), dim=-1)
    iterator: Iterable = range(N)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, leave=False)

    for i in iterator:
        band_emb  = band_embeddings[i].float()
        label_idx = int(labels[i].item())
        query_emb = query_tf[label_idx]
        with torch.enable_grad():
            result = pipeline.retrieve(band_emb, query_emb)
        fused[i] = result.fused_embedding.detach().cpu()

    return F.normalize(fused, dim=-1)


# ============================================================
# Single parameter sweep
# ============================================================

def run_single_sweep(
    param_name: str,
    values: List[Any],
    *,
    band_query:  Dict[str, torch.Tensor],
    band_gallery: Dict[str, torch.Tensor],
    class_text_features: torch.Tensor,
    fixed: Dict[str, Any],
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fix all params except param_name, sweep over values.
    Returns list of result dicts, one per value.
    """
    rows: List[Dict[str, Any]] = []

    for val in values:
        params = dict(fixed)
        params[param_name] = val

        # Build pipeline
        pipeline = MultispectralRetrievalPipeline(
            sigma=params["sigma"],
            tau=params["tau"],
            num_steps=params["num_steps"],
            lr=params["lr"],
            lambda_m=params["lambda_m"],
            k=params["k"],
        )

        t0 = time.perf_counter()
        fused_q = fuse_split(
            band_query["band_embeddings"], band_query["labels"],
            class_text_features, pipeline,
            desc=f"  {param_name}={val} query", show_progress=show_progress,
        )
        fused_g = fuse_split(
            band_gallery["band_embeddings"], band_gallery["labels"],
            class_text_features, pipeline,
            desc=f"  {param_name}={val} gallery", show_progress=show_progress,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics, *_ = evaluate_text_to_image_retrieval(
            query_features=fused_q,
            query_labels=band_query["labels"],
            gallery_features=fused_g,
            gallery_labels=band_gallery["labels"],
            ks=(1, 5, 10),
        )

        row = {
            "param_name":    param_name,
            "param_value":   val,
            "R@1":           float(metrics["R@1"]),
            "R@5":           float(metrics["R@5"]),
            "R@10":          float(metrics["R@10"]),
            "mAP":           float(metrics["mAP"]),
            "elapsed_ms":    round(elapsed_ms, 1),
        }
        rows.append(row)
        print(f"    {param_name}={val:<8}  R@1={row['R@1']:.2f}%  R@10={row['R@10']:.2f}%  ({elapsed_ms:.0f} ms)")

    return rows


# ============================================================
# Plotting helpers
# ============================================================

def _x_label_str(val: Any) -> str:
    """Format x-axis tick label: floats get minimal repr."""
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def plot_single_sensitivity(
    rows: List[Dict[str, Any]],
    param_name: str,
    output_path: Path,
    *,
    second_axis: bool = False,
    dpi: int = 300,
) -> None:
    """
    Plot R@1 vs param_value. Optionally add inference time on right axis
    (used for num_steps trade-off plot).
    """
    values = [r["param_value"] for r in rows]
    r1     = [r["R@1"]        for r in rows]
    ms     = [r["elapsed_ms"] for r in rows]

    x = list(range(len(values)))
    labels = [_x_label_str(v) for v in values]

    fig, ax1 = plt.subplots(figsize=(5.5, 4))

    # Primary axis: R@1
    ax1.plot(x, r1, "o-", color=CURVE_COLOR, linewidth=2, markersize=7, label="R@1 (%)")
    ax1.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=11)
    ax1.set_ylabel("R@1 (%)", color=CURVE_COLOR, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=CURVE_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--")

    # Mark optimal expected value
    opt = OPTIMAL_VALUES.get(param_name)
    if opt is not None:
        try:
            opt_x = values.index(opt)
            ax1.axvline(opt_x, color=OPTIMAL_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
            ax1.annotate(
                f"Optimal\n{_x_label_str(opt)}",
                xy=(opt_x, r1[opt_x]),
                xytext=(opt_x + 0.3, max(r1) - 0.3),
                fontsize=8, color=OPTIMAL_COLOR,
                arrowprops=dict(arrowstyle="->", color=OPTIMAL_COLOR),
            )
        except ValueError:
            pass

    # Shade "good range" for num_steps
    if param_name == "num_steps" and len(x) >= 3:
        ax1.axvspan(1, len(x) - 2, alpha=0.08, color=SHADE_COLOR, label="Good range")

    # Secondary axis: inference time (for num_steps trade-off)
    if second_axis:
        ax2 = ax1.twinx()
        per_image_ms = [m / max(band_n, 1) for m, band_n in
                        zip(ms, [1] * len(ms))]  # already per-image from fuse_split
        ax2.plot(x, ms, "s--", color="#9C27B0", linewidth=1.5, markersize=6,
                 label="Total time (ms)", alpha=0.75)
        ax2.set_ylabel("Fusion time (ms, query+gallery)", color="#9C27B0", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#9C27B0")

    plt.title(f"Sensitivity: {PARAM_LABELS.get(param_name, param_name)}", fontsize=11, pad=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {output_path}")


def plot_composite_6(
    all_rows: List[Dict[str, Any]],
    output_path: Path,
    *,
    param_order: Optional[List[str]] = None,
    dpi: int = 300,
) -> None:
    """
    2×3 grid of all 6 sensitivity plots — Figure 5 for paper.
    """
    if param_order is None:
        param_order = ["k", "lambda_m", "num_steps", "sigma", "tau", "lr"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for ax_idx, param_name in enumerate(param_order):
        ax = axes_flat[ax_idx]
        param_rows = [r for r in all_rows if r["param_name"] == param_name]
        if not param_rows:
            ax.set_visible(False)
            continue

        values = [r["param_value"] for r in param_rows]
        r1     = [r["R@1"]        for r in param_rows]
        ms_val = [r["elapsed_ms"] for r in param_rows]

        x = list(range(len(values)))
        labels = [_x_label_str(v) for v in values]

        ax.plot(x, r1, "o-", color=CURVE_COLOR, linewidth=2, markersize=6)
        ax.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=10)
        ax.set_ylabel("R@1 (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

        # Mark optimal
        opt = OPTIMAL_VALUES.get(param_name)
        if opt is not None:
            try:
                opt_x = values.index(opt)
                ax.axvline(opt_x, color=OPTIMAL_COLOR, linestyle="--",
                           linewidth=1.2, alpha=0.8)
            except ValueError:
                pass

        # Dual-axis for num_steps
        if param_name == "num_steps":
            ax2 = ax.twinx()
            ax2.plot(x, ms_val, "s--", color="#9C27B0", linewidth=1.2,
                     markersize=5, alpha=0.7)
            ax2.set_ylabel("Time (ms)", color="#9C27B0", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="#9C27B0")

        title = PARAM_LABELS.get(param_name, param_name).replace("$", "").replace("\\", "")
        ax.set_title(f"Sensitivity: {param_name}", fontsize=10, pad=5)

    fig.suptitle("Hyperparameter Sensitivity Analysis (EuroSAT, Fold 0)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite Figure 5 → {output_path}")


# ============================================================
# Summary CSV helper
# ============================================================

def build_summary_rows(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each param, find the value with highest R@1 and report variance."""
    summary: List[Dict[str, Any]] = []
    params = list(dict.fromkeys(r["param_name"] for r in all_rows))
    for param in params:
        pr = [r for r in all_rows if r["param_name"] == param]
        r1_vals = [r["R@1"] for r in pr]
        best = pr[int(np.argmax(r1_vals))]
        variance = float(np.max(r1_vals) - np.min(r1_vals))
        summary.append({
            "param": param,
            "tested_values": str([r["param_value"] for r in pr]),
            "best_value": best["param_value"],
            "best_R@1": round(best["R@1"], 2),
            "best_R@10": round(best["R@10"], 2),
            "R@1_range": round(variance, 2),
            "robust": "Yes" if variance < 1.5 else "No",
        })
    return summary


# ============================================================
# Main
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for the 13-band retrieval pipeline."
    )
    parser.add_argument("--root",           type=Path, default=Path("data/EuroSAT_MS"))
    parser.add_argument("--checkpoint",     type=Path, default=Path("checkpoints/ViT-B-16.pt"))
    parser.add_argument("--output-dir",     type=Path, default=Path("results/hyperparameter_sensitivity"))
    parser.add_argument("--fold-id",        type=int,  default=0,
                        help="Which fold to use as the sensitivity validation fold (default: 0)")
    parser.add_argument("--num-folds",      type=int,  default=5)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--batch-size",     type=int,  default=128)
    parser.add_argument("--num-workers",    type=int,  default=0)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--param",          type=str,  default=None,
                        choices=list(SENSITIVITY_GRID.keys()),
                        help="Run only this hyperparameter (default: all)")
    parser.add_argument("--smoke-test",     action="store_true",
                        help="Use reduced value ranges for fast validation (~15 min)")
    parser.add_argument("--no-progress",    action="store_true")
    parser.add_argument("--dpi",            type=int, default=300)
    args = parser.parse_args(argv)

    show_progress = not args.no_progress
    grid = SMOKE_GRID if args.smoke_test else SENSITIVITY_GRID

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv      = args.output_dir / "sensitivity_raw.csv"
    summary_csv  = args.output_dir / "sensitivity_summary.csv"
    manifest_json = args.output_dir / "sensitivity_manifest.json"
    composite_png = args.output_dir / "sensitivity_all_6.png"

    # --- Setup -------------------------------------------------------
    device = get_device()
    print(f"Device: {device}")
    clip_model, clip_tokenize = load_openai_clip_model(args.checkpoint, device)
    clip_model.eval()

    dataset = EuroSATMSDataset(
        args.root, normalize=True, reflectance_scale=10_000.0, clamp_range=(0.0, 1.0),
    )

    # Build fold splits and select the target fold
    fold_buckets = make_stratified_kfold_buckets(
        dataset.get_labels(), num_folds=args.num_folds, seed=args.seed,
    )
    fold_splits = build_retrieval_fold_splits(fold_buckets)
    fold = next(f for f in fold_splits if int(f["fold_id"]) == args.fold_id)

    print(f"Fold {args.fold_id}: query={len(fold['query'])}, gallery={len(fold['gallery'])}")

    query_loader   = _build_loader(dataset, fold["query"],   batch_size=args.batch_size, num_workers=args.num_workers)
    gallery_loader = _build_loader(dataset, fold["gallery"], batch_size=args.batch_size, num_workers=args.num_workers)

    # --- Encode text features ----------------------------------------
    print("Encoding class text features...")
    class_text_features = encode_class_text_features(
        clip_model, clip_tokenize,
        class_prompts=dataset.get_text_prompts(),
        device=device,
    )

    # --- Encode band embeddings ONCE ---------------------------------
    print("\n[Phase 1] Encoding per-band embeddings (done once, reused for all sweeps)...")
    band_query = encode_band_embeddings(
        query_loader, clip_model, device,
        micro_batch_size=args.micro_batch_size,
        desc="Query band embeddings", show_progress=show_progress,
    )
    band_gallery = encode_band_embeddings(
        gallery_loader, clip_model, device,
        micro_batch_size=args.micro_batch_size,
        desc="Gallery band embeddings", show_progress=show_progress,
    )
    print(f"  Query:   {band_query['band_embeddings'].shape}")
    print(f"  Gallery: {band_gallery['band_embeddings'].shape}")

    # --- Determine which params to sweep -----------------------------
    params_to_sweep = [args.param] if args.param else list(grid.keys())
    smoke_tag = " [SMOKE TEST]" if args.smoke_test else ""
    print(f"\n[Phase 2] Sweeping {len(params_to_sweep)} params{smoke_tag}: {params_to_sweep}")

    # Load existing rows if resuming
    all_rows: List[Dict[str, Any]] = []
    if raw_csv.exists() and args.param is not None:
        # Load existing rows, skip already-swept params
        with raw_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        all_rows = [
            {k: (float(v) if k in ("R@1","R@5","R@10","mAP","elapsed_ms") else v)
             for k, v in r.items()}
            for r in all_rows
        ]
        print(f"  Loaded {len(all_rows)} existing rows from {raw_csv}")

    # --- Main sweep loop ---------------------------------------------
    for param_name in params_to_sweep:
        values = grid[param_name]
        print(f"\n  ── {param_name} sweep: {values}")
        new_rows = run_single_sweep(
            param_name, values,
            band_query=band_query,
            band_gallery=band_gallery,
            class_text_features=class_text_features,
            fixed=OPTIMAL_FIXED,
            show_progress=show_progress,
        )
        # Remove stale rows for this param if re-sweeping
        all_rows = [r for r in all_rows if r["param_name"] != param_name]
        all_rows.extend(new_rows)

        # Save incrementally after each param
        save_csv_rows(all_rows, raw_csv)

        # Plot individual sensitivity curve
        plot_path = args.output_dir / f"sensitivity_{param_name}.png"
        plot_single_sensitivity(
            new_rows, param_name, plot_path,
            second_axis=(param_name == "num_steps"),
            dpi=args.dpi,
        )

    # --- Summary CSV -------------------------------------------------
    summary_rows = build_summary_rows(all_rows)
    save_csv_rows(summary_rows, summary_csv)
    print(f"\nSaved summary → {summary_csv}")

    # --- Composite Figure 5 ------------------------------------------
    if len(params_to_sweep) > 1 or set(params_to_sweep) == set(grid.keys()):
        plot_composite_6(all_rows, composite_png, dpi=args.dpi)

    # --- Manifest ----------------------------------------------------
    manifest = {
        "script":        "scripts/run_hyperparameter_sensitivity.py",
        "dataset":       str(args.root),
        "checkpoint":    str(args.checkpoint),
        "fold_id":       args.fold_id,
        "num_folds":     args.num_folds,
        "seed":          args.seed,
        "smoke_test":    args.smoke_test,
        "params_swept":  params_to_sweep,
        "optimal_fixed": {k: str(v) for k, v in OPTIMAL_FIXED.items()},
        "grid":          {k: v for k, v in grid.items() if k in params_to_sweep},
        "num_experiments": sum(len(grid[p]) for p in params_to_sweep),
        "outputs": {
            "raw_csv":       str(raw_csv),
            "summary_csv":   str(summary_csv),
            "composite_png": str(composite_png),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest → {manifest_json}")

    # --- Print summary table -----------------------------------------
    print("\n╔══ Hyperparameter Sensitivity Summary ══╗")
    print(f"  {'Param':<12} {'Best value':<14} {'Best R@1':>8}  {'R@1 range':>10}  {'Robust':>7}")
    print("  " + "─" * 58)
    for row in summary_rows:
        print(f"  {row['param']:<12} {str(row['best_value']):<14} {row['best_R@1']:>8.2f}%  "
              f"{row['R@1_range']:>9.2f}%  {row['robust']:>7}")
    print("╚" + "═" * 60 + "╝")


if __name__ == "__main__":
    main()
