"""
Component Ablation Study — 5-Fold Cross-Validation  (Task 1, Week 7)
=====================================================================

Evaluates 5 incremental pipeline configurations on EuroSAT-MS using a
stratified 5-fold CV to produce paper-aligned mean ± std results.

Ablation stages
---------------
  Stage 1  Ablation-RGB      RGB-CLIP (bands B04/B03/B02 → 3-ch CLIP)
  Stage 2  Ablation-PBMean   Naïve mean of 13 per-band CLIP embeddings
  Stage 3  Ablation-Affinity  Query-conditioned affinity weights (Eq.1),
                              no Fiedler, no test-time optimisation
  Stage 4  Ablation-Fiedler   Affinity (Eq.1) + Fiedler init (Eq.2),
                              num_steps=0 (no test-time optimisation)
  Stage 5  Ablation-Full      Full pipeline: Affinity + Fiedler + Manifold
                              optimisation (num_steps=5)  ← should match "Ours"

Design constraints
------------------
* Does NOT modify src/experiments/eurosat_5fold_cv.py (Week 6 code).
* Only *imports* make_stratified_kfold_buckets / build_retrieval_fold_splits
  as read-only utilities.
* Per-band embeddings are encoded ONCE per fold and reused across stages 2-5.
* Query embedding for Affinity (Eq.1) is always the class *text* embedding
  (Design Rule 1 / G7 fix).

Usage
-----
    python scripts/run_component_ablation_5fold.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# --- Project imports (read-only) -----------------------------------------
from src.datasets.eurosat import EuroSATMSDataset
from src.experiments.eurosat_5fold_cv import (
    make_stratified_kfold_buckets,
    build_retrieval_fold_splits,
)
from src.models.clip_utils import encode_test_gallery_rgb, preprocess_rgb_for_clip
from src.models.per_band_encoder import encode_multispectral_batch
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline
from src.utils.metrics import evaluate_text_to_image_retrieval
from src.utils.shared import get_device, load_openai_clip_model, save_csv_rows
from src.utils.visualization import extract_rgb_bands

# =========================================================================
# Constants
# =========================================================================

STAGE_LABELS: List[str] = [
    "Ablation-RGB",
    "Ablation-PBMean",
    "Ablation-Affinity",
    "Ablation-Fiedler",
    "Ablation-Full",
]

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)

# =========================================================================
# Data helpers
# =========================================================================


def _build_loader(
    dataset: EuroSATMSDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


# =========================================================================
# Text feature encoding (with cache)
# =========================================================================


@torch.no_grad()
def _encode_class_text_features(
    clip_model,
    clip_tokenize,
    *,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Encode class-level text prompts with CLIP text encoder.

    Returns
    -------
    text_features : (C, D) L2-normalised float tensor on CPU
    """
    prompts = [f"A satellite image of {name.lower()}" for name in class_names]
    tokens = clip_tokenize(prompts).to(device)
    text_features = clip_model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).cpu()


# =========================================================================
# Stage 1 — RGB-CLIP
# =========================================================================


@torch.no_grad()
def _encode_rgb(
    loader: DataLoader,
    clip_model,
    device: torch.device,
    *,
    image_size: int = 224,
    rgb_indices: tuple[int, ...] = (3, 2, 1),
    desc: str,
    show_progress: bool,
) -> Dict[str, Any]:
    """Encode multispectral images using only the RGB bands (B04/B03/B02)."""
    clip_model.eval()
    features_chunks: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    iterator: Iterable = tqdm(loader, desc=desc) if show_progress else loader
    for batch in iterator:
        image_13 = batch["image"]
        rgb = extract_rgb_bands(image_13, rgb_indices=rgb_indices)
        rgb = preprocess_rgb_for_clip(rgb, image_size=image_size).to(device)
        feats = clip_model.encode_image(rgb).float()
        features_chunks.append(F.normalize(feats, dim=-1).cpu())
        labels_list.append(batch["label"].cpu())

    return {
        "features": torch.cat(features_chunks, dim=0),
        "labels": torch.cat(labels_list, dim=0),
    }


# =========================================================================
# Stage 2-5 — Per-band embedding encoding (encode ONCE, reuse)
# =========================================================================


@torch.no_grad()
def _encode_band_embeddings(
    loader: DataLoader,
    clip_model,
    device: torch.device,
    *,
    micro_batch_size: int = 64,
    desc: str,
    show_progress: bool,
) -> Dict[str, Any]:
    """
    Encode all 13 spectral bands with CLIP image encoder.

    Returns
    -------
    band_embeddings : (N, B, D) L2-normalised per-band CLIP embeddings on CPU
    labels          : (N,) integer class indices
    """
    clip_model.eval()
    band_chunks: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    iterator: Iterable = tqdm(loader, desc=desc) if show_progress else loader
    for batch in iterator:
        band_emb = encode_multispectral_batch(
            batch["image"],
            clip_model,
            device=device,
            micro_batch_size=micro_batch_size,
        )
        band_chunks.append(band_emb.cpu())
        labels_list.append(batch["label"].cpu())

    return {
        "band_embeddings": torch.cat(band_chunks, dim=0),   # (N, B, D)
        "labels": torch.cat(labels_list, dim=0),             # (N,)
    }


# =========================================================================
# Fusion helpers for each ablation stage
# =========================================================================


def _fuse_pb_mean(band_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Stage 2 — Naïve mean fusion.

    Args:
        band_embeddings: (B, D) per-band CLIP embeddings for one sample
    Returns:
        (D,) L2-normalised fused embedding
    """
    return F.normalize(band_embeddings.mean(dim=0), dim=0)


def _fuse_affinity_only(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Stage 3 — Query-conditioned affinity weights (Eq. 1), no Fiedler / optimisation.

    Args:
        band_embeddings: (B, D) L2-normalised per-band embeddings
        query_embedding: (D,)  L2-normalised class text embedding
        sigma:           temperature for softmax
    Returns:
        (D,) L2-normalised fused embedding
    """
    scores = (band_embeddings @ query_embedding) / sigma   # (B,)
    weights = torch.softmax(scores, dim=0)                 # (B,)
    fused = (weights.unsqueeze(1) * band_embeddings).sum(0)
    return F.normalize(fused, dim=0)


def _fuse_fiedler_only(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Stage 4 — Affinity (Eq.1) + Fiedler init weights (Eq.2), no test-time optimisation.

    Uses MultispectralRetrievalPipeline with num_steps=0.
    """
    pipeline = MultispectralRetrievalPipeline(sigma=sigma, tau=None, num_steps=0)
    with torch.enable_grad():
        result = pipeline.retrieve(band_embeddings.float(), query_embedding.float())
    return result.fused_embedding.detach()


def _fuse_full_pipeline(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    sigma: float = 0.5,
    num_steps: int = 5,
    lr: float = 0.01,
    lambda_m: float = 0.1,
    k: int = 5,
    grad_clip: float = 1.0,
    early_stop_tol: float = 1e-6,
) -> torch.Tensor:
    """
    Stage 5 — Full pipeline: Affinity + Fiedler + Manifold optimisation.
    Should match "Ours" results from eurosat_5fold_cv.py.
    """
    pipeline = MultispectralRetrievalPipeline(
        sigma=sigma,
        tau=None,
        num_steps=num_steps,
        lr=lr,
        lambda_m=lambda_m,
        k=k,
        grad_clip=grad_clip,
        early_stop_tol=early_stop_tol,
    )
    with torch.enable_grad():
        result = pipeline.retrieve(band_embeddings.float(), query_embedding.float())
    return result.fused_embedding.detach()


# =========================================================================
# Run one stage across all samples of query/gallery
# =========================================================================


def _run_stage(
    stage: str,
    *,
    # Stage 1 inputs (RGB)
    query_rgb_features: Optional[torch.Tensor] = None,
    gallery_rgb_features: Optional[torch.Tensor] = None,
    query_rgb_labels: Optional[torch.Tensor] = None,
    gallery_rgb_labels: Optional[torch.Tensor] = None,
    # Stage 2-5 inputs
    query_band_embeddings: Optional[torch.Tensor] = None,
    gallery_band_embeddings: Optional[torch.Tensor] = None,
    query_band_labels: Optional[torch.Tensor] = None,
    gallery_band_labels: Optional[torch.Tensor] = None,
    class_text_features: Optional[torch.Tensor] = None,
    # Hyperparameters
    sigma: float = 0.5,
    num_steps: int = 5,
    lr: float = 0.01,
    lambda_m: float = 0.1,
    k: int = 5,
    grad_clip: float = 1.0,
    early_stop_tol: float = 1e-6,
    # Evaluation
    ks: Sequence[int] = DEFAULT_KS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run one ablation stage and return metrics dict."""

    if stage == "Ablation-RGB":
        assert query_rgb_features is not None
        metrics, *_ = evaluate_text_to_image_retrieval(
            query_rgb_features, query_rgb_labels,
            gallery_rgb_features, gallery_rgb_labels,
            ks=ks,
        )
        return metrics

    # Stages 2-5: fuse per-band embeddings then evaluate
    assert query_band_embeddings is not None
    assert gallery_band_embeddings is not None
    assert class_text_features is not None

    n_query = query_band_embeddings.shape[0]
    n_gallery = gallery_band_embeddings.shape[0]

    query_fused = torch.zeros(n_query, query_band_embeddings.shape[2])
    gallery_fused = torch.zeros(n_gallery, gallery_band_embeddings.shape[2])

    q_iter: Iterable = range(n_query)
    g_iter: Iterable = range(n_gallery)
    if show_progress:
        q_iter = tqdm(q_iter, desc=f"  Fusing query [{stage}]", leave=False)
        g_iter = tqdm(g_iter, desc=f"  Fusing gallery [{stage}]", leave=False)

    for i in q_iter:
        band_emb = query_band_embeddings[i].float()
        label_idx = int(query_band_labels[i].item())
        query_emb = class_text_features[label_idx].float()

        if stage == "Ablation-PBMean":
            query_fused[i] = _fuse_pb_mean(band_emb)
        elif stage == "Ablation-Affinity":
            query_fused[i] = _fuse_affinity_only(band_emb, query_emb, sigma)
        elif stage == "Ablation-Fiedler":
            query_fused[i] = _fuse_fiedler_only(band_emb, query_emb, sigma)
        elif stage == "Ablation-Full":
            query_fused[i] = _fuse_full_pipeline(
                band_emb, query_emb, sigma=sigma, num_steps=num_steps,
                lr=lr, lambda_m=lambda_m, k=k,
                grad_clip=grad_clip, early_stop_tol=early_stop_tol,
            )

    for i in g_iter:
        band_emb = gallery_band_embeddings[i].float()
        label_idx = int(gallery_band_labels[i].item())
        query_emb = class_text_features[label_idx].float()

        if stage == "Ablation-PBMean":
            gallery_fused[i] = _fuse_pb_mean(band_emb)
        elif stage == "Ablation-Affinity":
            gallery_fused[i] = _fuse_affinity_only(band_emb, query_emb, sigma)
        elif stage == "Ablation-Fiedler":
            gallery_fused[i] = _fuse_fiedler_only(band_emb, query_emb, sigma)
        elif stage == "Ablation-Full":
            gallery_fused[i] = _fuse_full_pipeline(
                band_emb, query_emb, sigma=sigma, num_steps=num_steps,
                lr=lr, lambda_m=lambda_m, k=k,
                grad_clip=grad_clip, early_stop_tol=early_stop_tol,
            )

    query_fused = F.normalize(query_fused, dim=-1)
    gallery_fused = F.normalize(gallery_fused, dim=-1)

    metrics, *_ = evaluate_text_to_image_retrieval(
        query_fused, query_band_labels,
        gallery_fused, gallery_band_labels,
        ks=ks,
    )
    return metrics


# =========================================================================
# Main experiment loop
# =========================================================================


def run_component_ablation_5fold(
    *,
    root: Path | str,
    clip_checkpoint: Path | str,
    results_dir: Path | str = Path("results/ablation/component_ablation_5fold"),
    num_folds: int = 5,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    image_size: int = 224,
    micro_batch_size: int = 64,
    show_progress: bool = True,
    # Pipeline hyperparams (same defaults as "Ours" in Week 6)
    sigma: float = 0.5,
    num_steps: int = 5,
    lr: float = 0.01,
    lambda_m: float = 0.1,
    k: int = 5,
    grad_clip: float = 1.0,
    early_stop_tol: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run the 5-stage component ablation study over num_folds folds.

    For each fold:
      1. Encode RGB features for query and gallery  (Stage 1)
      2. Encode per-band embeddings for query and gallery  (Stages 2-5)
      3. Encode class-level text features  (shared across stages)
      4. Run all 5 stages on the cached embeddings
      5. Append per-fold metrics to fold_rows
    
    Saves:
      results_dir/fold_metrics.csv   — per-fold metrics for all stages
      results_dir/manifest.json      — experiment parameters
    """
    root = Path(root)
    clip_checkpoint = Path(clip_checkpoint)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()

    print(f"[Ablation 5-fold] Device: {device}")
    print(f"[Ablation 5-fold] Root:   {root}")
    print(f"[Ablation 5-fold] Output: {results_dir}")

    # ------------------------------------------------------------------
    # Load CLIP model
    # ------------------------------------------------------------------
    clip_model, clip_tokenize = load_openai_clip_model(clip_checkpoint, device)
    clip_model.eval()

    # ------------------------------------------------------------------
    # Load dataset and build fold splits (same seed=42 as Week 6)
    # ------------------------------------------------------------------
    dataset = EuroSATMSDataset(
        root,
        normalize=True,
        reflectance_scale=10_000.0,
        clamp_range=(0.0, 1.0),
    )

    fold_buckets = make_stratified_kfold_buckets(
        dataset.get_labels(),
        num_folds=num_folds,
        seed=seed,
    )
    fold_splits = build_retrieval_fold_splits(fold_buckets)

    # ------------------------------------------------------------------
    # Encode class text features (shared across all folds & stages)
    # ------------------------------------------------------------------
    print("[Ablation 5-fold] Encoding class text features...")
    class_text_features = _encode_class_text_features(
        clip_model, clip_tokenize,
        class_names=dataset.class_names,
        device=device,
    )  # (C, D) on CPU

    # ------------------------------------------------------------------
    # Main fold loop
    # ------------------------------------------------------------------
    fold_rows: List[Dict[str, Any]] = []

    for fold in fold_splits:
        fold_id = int(fold["fold_id"])
        query_indices = fold["query"]
        gallery_indices = fold["gallery"]

        print(f"\n[Ablation 5-fold] ── Fold {fold_id + 1}/{num_folds} "
              f"(query={len(query_indices)}, gallery={len(gallery_indices)}) ──")

        query_loader = _build_loader(
            dataset, query_indices,
            batch_size=batch_size, num_workers=num_workers, shuffle=False,
        )
        gallery_loader = _build_loader(
            dataset, gallery_indices,
            batch_size=batch_size, num_workers=num_workers, shuffle=False,
        )

        # --------------------------------------------------------------
        # A. Stage 1 — RGB features
        # --------------------------------------------------------------
        print(f"  [Stage 1] Encoding RGB features...")
        t0 = time.perf_counter()
        query_rgb = _encode_rgb(
            query_loader, clip_model, device,
            image_size=image_size, desc="  RGB query", show_progress=show_progress,
        )
        gallery_rgb = _encode_rgb(
            gallery_loader, clip_model, device,
            image_size=image_size, desc="  RGB gallery", show_progress=show_progress,
        )
        rgb_encode_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [Stage 1] RGB encoding done in {rgb_encode_ms:.0f} ms")

        # --------------------------------------------------------------
        # B. Stages 2-5 — Per-band embeddings (encode ONCE, reuse)
        # --------------------------------------------------------------
        print(f"  [Stages 2-5] Encoding per-band embeddings...")
        t0 = time.perf_counter()
        query_band = _encode_band_embeddings(
            query_loader, clip_model, device,
            micro_batch_size=micro_batch_size,
            desc="  Band query", show_progress=show_progress,
        )
        gallery_band = _encode_band_embeddings(
            gallery_loader, clip_model, device,
            micro_batch_size=micro_batch_size,
            desc="  Band gallery", show_progress=show_progress,
        )
        band_encode_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [Stages 2-5] Band encoding done in {band_encode_ms:.0f} ms")

        # --------------------------------------------------------------
        # C. Run all 5 stages
        # --------------------------------------------------------------
        for stage in STAGE_LABELS:
            print(f"  [{stage}] Evaluating...", end=" ", flush=True)
            t0 = time.perf_counter()

            metrics = _run_stage(
                stage,
                query_rgb_features=query_rgb["features"],
                gallery_rgb_features=gallery_rgb["features"],
                query_rgb_labels=query_rgb["labels"],
                gallery_rgb_labels=gallery_rgb["labels"],
                query_band_embeddings=query_band["band_embeddings"],
                gallery_band_embeddings=gallery_band["band_embeddings"],
                query_band_labels=query_band["labels"],
                gallery_band_labels=gallery_band["labels"],
                class_text_features=class_text_features,
                sigma=sigma, num_steps=num_steps, lr=lr,
                lambda_m=lambda_m, k=k,
                grad_clip=grad_clip, early_stop_tol=early_stop_tol,
                ks=DEFAULT_KS,
                show_progress=show_progress,
            )

            stage_ms = (time.perf_counter() - t0) * 1000.0
            row: Dict[str, Any] = {
                "fold_id": fold_id,
                "stage": stage,
                "num_query": len(query_indices),
                "num_gallery": len(gallery_indices),
                **metrics,
                "elapsed_ms": round(stage_ms, 1),
            }
            fold_rows.append(row)
            print(f"R@1={metrics['R@1']:.2f}%  R@10={metrics['R@10']:.2f}%  "
                  f"({stage_ms:.0f} ms)")

    # ------------------------------------------------------------------
    # Save per-fold metrics
    # ------------------------------------------------------------------
    fold_metrics_path = results_dir / "fold_metrics.csv"
    save_csv_rows(fold_rows, fold_metrics_path)
    print(f"\n[Ablation 5-fold] Saved fold metrics → {fold_metrics_path}")

    # ------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------
    manifest = {
        "script": "scripts/run_component_ablation_5fold.py",
        "dataset_root": str(root),
        "clip_checkpoint": str(clip_checkpoint),
        "num_folds": num_folds,
        "seed": seed,
        "stages": STAGE_LABELS,
        "hyperparams": {
            "sigma": sigma,
            "num_steps": num_steps,
            "lr": lr,
            "lambda_m": lambda_m,
            "k": k,
            "grad_clip": grad_clip,
            "early_stop_tol": early_stop_tol,
        },
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[Ablation 5-fold] Saved manifest     → {manifest_path}")

    return {"fold_rows": fold_rows, "manifest": manifest}


# =========================================================================
# CLI entry point
# =========================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5-fold component ablation study for multispectral retrieval."
    )
    p.add_argument(
        "--root", type=Path,
        default=Path("data/EuroSAT_MS"),
        help="Path to EuroSAT-MS dataset root",
    )
    p.add_argument(
        "--clip-checkpoint", type=Path,
        default=Path("checkpoints/ViT-B-16.pt"),
        help="Path to local CLIP ViT-B/16 checkpoint",
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("results/ablation/component_ablation_5fold"),
        help="Output directory",
    )
    p.add_argument("--num-folds",      type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--num-workers",    type=int,   default=0)
    p.add_argument("--image-size",     type=int,   default=224)
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--sigma",          type=float, default=0.5)
    p.add_argument("--num-steps",      type=int,   default=5)
    p.add_argument("--lr",             type=float, default=0.01)
    p.add_argument("--lambda-m",       type=float, default=0.1)
    p.add_argument("--k",              type=int,   default=5)
    p.add_argument("--no-progress",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_component_ablation_5fold(
        root=args.root,
        clip_checkpoint=args.clip_checkpoint,
        results_dir=args.results_dir,
        num_folds=args.num_folds,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        micro_batch_size=args.micro_batch_size,
        sigma=args.sigma,
        num_steps=args.num_steps,
        lr=args.lr,
        lambda_m=args.lambda_m,
        k=args.k,
        show_progress=not args.no_progress,
    )
