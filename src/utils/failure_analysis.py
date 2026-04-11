"""
Failure case detection and analysis for multispectral retrieval.

Identifies samples where the retrieval pipeline fails (R@1 < threshold),
classifies failure causes into 4 categories, and provides visualization
utilities for failure gallery inspection.

Failure categories:
    1. Cloud cover       — high B10 (Cirrus) or high B09 (Water Vapor) reflectance
    2. Mixed scene       — high spectral entropy across bands (heterogeneous content)
    3. Rare class        — class with few training samples / low gallery representation
    4. Seasonal variation — high within-class embedding variance (phenological changes)

Core functions:
    - identify_failure_cases()          : find samples with R@1 < threshold
    - classify_failure_causes()         : assign cause category to each failure
    - compute_failure_statistics()      : aggregate failure stats per class/cause
    - plot_failure_gallery()            : visualize RGB + attribution + prediction
    - plot_failure_cause_distribution() : pie/bar chart of failure breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Constants
# ============================================================

FAILURE_CAUSES = [
    "cloud_cover",
    "mixed_scene",
    "rare_class",
    "seasonal_variation",
]

FAILURE_CAUSE_LABELS = {
    "cloud_cover":        "Cloud Cover",
    "mixed_scene":        "Mixed Scene",
    "rare_class":         "Rare Class",
    "seasonal_variation": "Seasonal Variation",
}

FAILURE_CAUSE_COLORS = {
    "cloud_cover":        "#708090",   # Slate gray
    "mixed_scene":        "#FF8C00",   # Dark orange
    "rare_class":         "#DC143C",   # Crimson
    "seasonal_variation": "#4169E1",   # Royal blue
}

# Sentinel-2 band indices for heuristic checks
_B09_WV_IDX = 9     # Water Vapor band
_B10_CIRRUS_IDX = 10 # Cirrus band
_RGB_INDICES = [3, 2, 1]  # B04, B03, B02


# ============================================================
# 1) Data containers
# ============================================================

@dataclass
class FailureCase:
    """A single sample identified as a retrieval failure."""

    sample_index: int
    true_label: str
    predicted_label: str
    true_label_idx: int
    predicted_label_idx: int
    similarity_score: float         # cosine similarity to top-1 match
    first_hit_rank: Optional[int]   # rank of first correct retrieval (None = never)
    average_precision: float        # AP for this query
    failure_cause: str              # one of FAILURE_CAUSES
    cause_confidence: float         # how confident the cause classification is [0,1]
    band_attribution: Optional[np.ndarray] = None   # (13,) attribution scores
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAnalysisResult:
    """Complete failure analysis output."""

    total_samples: int
    total_failures: int
    failure_rate: float                     # fraction of failures
    threshold: float                        # R@1 threshold used
    failures: List[FailureCase]             # all failure cases
    per_class_stats: Dict[str, Dict[str, Any]]
    per_cause_stats: Dict[str, Dict[str, Any]]
    class_names: List[str]


# ============================================================
# 2) Core: identify failure cases
# ============================================================

def identify_failure_cases(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    label_names: Optional[List[str]] = None,
    idx_to_class: Optional[Dict[int, str]] = None,
    threshold_rank: int = 1,
    image_paths: Optional[List[str]] = None,
) -> Tuple[List[FailureCase], Dict[str, Any]]:
    """
    Identify samples where retrieval fails (no correct match in top-K).

    A failure is defined as: the true class does NOT appear in the
    top-`threshold_rank` retrieval results (i.e., R@1 miss when
    threshold_rank=1).

    Args:
        query_features:  (Q, D) L2-normalized query embeddings
        query_labels:    (Q,) integer class labels
        gallery_features: (N, D) L2-normalized gallery embeddings
        gallery_labels:  (N,) integer class labels
        label_names:     optional list mapping query index → class name string
        idx_to_class:    optional dict mapping label int → class name string
        threshold_rank:  rank cutoff for failure detection (default 1 = R@1)
        image_paths:     optional list of image file paths for each query

    Returns:
        failures: list of FailureCase objects
        summary:  dict with overall statistics
    """
    query_features = query_features.cpu().float()
    query_labels = query_labels.cpu()
    gallery_features = gallery_features.cpu().float()
    gallery_labels = gallery_labels.cpu()

    Q = query_features.shape[0]

    # Cosine similarity matrix
    similarity_matrix = query_features @ gallery_features.T  # (Q, N)
    ranked_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

    # Get ranked labels
    expanded_gallery = gallery_labels.unsqueeze(0).expand(Q, -1)
    ranked_labels = torch.gather(expanded_gallery, 1, ranked_indices)
    ranked_relevance = ranked_labels.eq(query_labels.unsqueeze(1))

    # Get ranked similarities
    ranked_sims = torch.gather(similarity_matrix, 1, ranked_indices)

    failures = []
    n_hits = 0

    for q in range(Q):
        # Check if correct class is in top-K
        is_hit = ranked_relevance[q, :threshold_rank].any().item()

        if is_hit:
            n_hits += 1
            continue

        # This is a failure — collect details
        true_label_idx = int(query_labels[q].item())
        pred_label_idx = int(ranked_labels[q, 0].item())
        top1_sim = float(ranked_sims[q, 0].item())

        # Find first correct hit rank
        hit_positions = torch.where(ranked_relevance[q])[0]
        first_hit = int(hit_positions[0].item()) + 1 if len(hit_positions) > 0 else None

        # Average Precision
        rel_row = ranked_relevance[q].float()
        n_rel = int(rel_row.sum().item())
        if n_rel > 0:
            prec_at_k = torch.cumsum(rel_row, dim=0) / torch.arange(
                1, rel_row.numel() + 1, dtype=torch.float32
            )
            ap = float((prec_at_k * rel_row).sum().item() / n_rel)
        else:
            ap = 0.0

        # Class name resolution
        if label_names is not None:
            true_name = label_names[q]
        elif idx_to_class is not None:
            true_name = idx_to_class.get(true_label_idx, str(true_label_idx))
        else:
            true_name = str(true_label_idx)

        if idx_to_class is not None:
            pred_name = idx_to_class.get(pred_label_idx, str(pred_label_idx))
        else:
            pred_name = str(pred_label_idx)

        fc = FailureCase(
            sample_index=q,
            true_label=true_name,
            predicted_label=pred_name,
            true_label_idx=true_label_idx,
            predicted_label_idx=pred_label_idx,
            similarity_score=top1_sim,
            first_hit_rank=first_hit,
            average_precision=ap,
            failure_cause="unknown",
            cause_confidence=0.0,
            image_path=image_paths[q] if image_paths else None,
        )
        failures.append(fc)

    summary = {
        "total_queries": Q,
        "total_failures": len(failures),
        "total_hits": n_hits,
        "failure_rate": len(failures) / max(Q, 1),
        "hit_rate": n_hits / max(Q, 1),
        "threshold_rank": threshold_rank,
    }

    return failures, summary


# ============================================================
# 3) Classify failure causes
# ============================================================

def _detect_cloud_cover(
    image: Optional[torch.Tensor],
    band_embeddings: Optional[torch.Tensor] = None,
    *,
    cirrus_threshold: float = 0.15,
    wv_threshold: float = 0.25,
) -> Tuple[bool, float]:
    """
    Heuristic: high Cirrus (B10) or Water Vapor (B09) reflectance indicates clouds.

    Args:
        image: (13, H, W) multispectral image, values in [0, 1]
        band_embeddings: (13, D) — unused here, reserved for future use
        cirrus_threshold: mean B10 value above this → cloud flag
        wv_threshold: mean B09 value above this → cloud flag

    Returns:
        (is_cloudy, confidence)
    """
    if image is None:
        return False, 0.0

    if image.ndim != 3 or image.shape[0] < 13:
        return False, 0.0

    b10_mean = image[_B10_CIRRUS_IDX].mean().item()
    b09_mean = image[_B09_WV_IDX].mean().item()

    # Combined score: higher = more likely cloudy
    cloud_score = max(b10_mean / cirrus_threshold, b09_mean / wv_threshold)
    is_cloudy = cloud_score > 1.0
    confidence = min(cloud_score, 2.0) / 2.0  # normalize to [0, 1]

    return is_cloudy, float(confidence)


def _detect_mixed_scene(
    image: Optional[torch.Tensor],
    band_embeddings: Optional[torch.Tensor] = None,
    *,
    entropy_threshold: float = 0.7,
) -> Tuple[bool, float]:
    """
    Heuristic: high spectral entropy / inter-band variance indicates
    heterogeneous (mixed) scene content.

    Uses the coefficient of variation across band means as a proxy
    for scene heterogeneity.

    Args:
        image: (13, H, W) multispectral image
        entropy_threshold: normalized entropy above this → mixed scene

    Returns:
        (is_mixed, confidence)
    """
    if image is None and band_embeddings is None:
        return False, 0.0

    if image is not None and image.ndim == 3 and image.shape[0] >= 13:
        # Compute spatial variance across all bands
        band_means = image.mean(dim=(1, 2))          # (13,)
        band_spatial_std = image.std(dim=(1, 2))      # (13,)

        # High spatial std within bands → mixed content
        mean_spatial_cv = (band_spatial_std / (band_means + 1e-8)).mean().item()

        # Normalize to [0, 1] range (typical CV for homogeneous ~0.2, mixed ~0.8+)
        normalized_score = min(mean_spatial_cv / 1.0, 1.0)
        is_mixed = normalized_score > entropy_threshold
        return is_mixed, float(normalized_score)

    if band_embeddings is not None:
        # Use pairwise cosine distance between band embeddings
        if band_embeddings.ndim == 2:
            sim = F.cosine_similarity(
                band_embeddings.unsqueeze(0),
                band_embeddings.unsqueeze(1),
                dim=-1,
            )  # (13, 13)
            # Low average similarity → bands see very different content → mixed
            avg_sim = sim.mean().item()
            diversity = 1.0 - avg_sim
            is_mixed = diversity > (1.0 - entropy_threshold)
            return is_mixed, float(min(diversity / 0.5, 1.0))

    return False, 0.0


def _detect_rare_class(
    true_label: str,
    class_distribution: Dict[str, int],
    *,
    percentile_threshold: float = 20.0,
) -> Tuple[bool, float]:
    """
    Heuristic: classes with very few samples in the gallery are harder to retrieve.

    Args:
        true_label: the ground-truth class name
        class_distribution: {class_name: count} in gallery
        percentile_threshold: below this percentile of class counts → rare

    Returns:
        (is_rare, confidence)
    """
    if not class_distribution or true_label not in class_distribution:
        return False, 0.0

    counts = np.array(list(class_distribution.values()))
    threshold_count = np.percentile(counts, percentile_threshold)
    sample_count = class_distribution[true_label]

    is_rare = sample_count <= threshold_count
    # Confidence: how far below median this class is
    median_count = np.median(counts)
    if median_count > 0:
        confidence = max(0.0, 1.0 - sample_count / median_count)
    else:
        confidence = 0.0

    return is_rare, float(min(confidence, 1.0))


def _detect_seasonal_variation(
    band_embeddings: Optional[torch.Tensor],
    class_mean_embedding: Optional[torch.Tensor] = None,
    class_embedding_std: Optional[float] = None,
    *,
    deviation_threshold: float = 1.5,
) -> Tuple[bool, float]:
    """
    Heuristic: if this sample's embedding is far from the class mean,
    it may represent seasonal/temporal variation.

    Uses L2 distance from the sample's mean embedding to the class centroid.

    Args:
        band_embeddings: (13, D) this sample's band embeddings
        class_mean_embedding: (D,) centroid of this class
        class_embedding_std: typical intra-class std (L2 scale)
        deviation_threshold: # of std deviations to flag

    Returns:
        (is_seasonal, confidence)
    """
    if band_embeddings is None or class_mean_embedding is None:
        return False, 0.0

    # Mean of all bands for this sample
    sample_mean = band_embeddings.mean(dim=0)  # (D,)
    distance = torch.norm(sample_mean - class_mean_embedding).item()

    if class_embedding_std is None or class_embedding_std < 1e-8:
        return False, 0.0

    z_score = distance / class_embedding_std
    is_seasonal = z_score > deviation_threshold
    confidence = min(z_score / (deviation_threshold * 2), 1.0)

    return is_seasonal, float(confidence)


def classify_failure_causes(
    failures: List[FailureCase],
    *,
    images: Optional[List[torch.Tensor]] = None,
    band_embeddings: Optional[List[torch.Tensor]] = None,
    class_distribution: Optional[Dict[str, int]] = None,
    class_centroids: Optional[Dict[str, torch.Tensor]] = None,
    class_stds: Optional[Dict[str, float]] = None,
    cirrus_threshold: float = 0.15,
    wv_threshold: float = 0.25,
    entropy_threshold: float = 0.7,
    rare_percentile: float = 20.0,
    deviation_threshold: float = 1.5,
) -> List[FailureCase]:
    """
    Classify failure cause for each FailureCase.

    Priority order (first match wins):
        1. Cloud cover (spectral heuristic)
        2. Mixed scene (spatial heterogeneity)
        3. Rare class (gallery statistics)
        4. Seasonal variation (embedding deviation)

    If no detector triggers, defaults to "mixed_scene" with low confidence
    (as the most common catch-all for ambiguous failures).

    Args:
        failures: list of FailureCase (from identify_failure_cases)
        images: optional list of (13, H, W) images aligned with failures
        band_embeddings: optional list of (13, D) embeddings aligned with failures
        class_distribution: {class_name: count} in gallery
        class_centroids: {class_name: (D,) mean embedding}
        class_stds: {class_name: float L2 std}
        *_threshold: tuning parameters for each detector

    Returns:
        Updated list of FailureCase with cause and confidence filled in
    """
    for i, fc in enumerate(failures):
        causes_scores = []

        img = images[i] if images and i < len(images) else None
        emb = band_embeddings[i] if band_embeddings and i < len(band_embeddings) else None

        # 1) Cloud cover
        is_cloud, cloud_conf = _detect_cloud_cover(
            img, cirrus_threshold=cirrus_threshold, wv_threshold=wv_threshold
        )
        if is_cloud:
            causes_scores.append(("cloud_cover", cloud_conf))

        # 2) Mixed scene
        is_mixed, mix_conf = _detect_mixed_scene(
            img, emb, entropy_threshold=entropy_threshold
        )
        if is_mixed:
            causes_scores.append(("mixed_scene", mix_conf))

        # 3) Rare class
        if class_distribution:
            is_rare, rare_conf = _detect_rare_class(
                fc.true_label, class_distribution, percentile_threshold=rare_percentile
            )
            if is_rare:
                causes_scores.append(("rare_class", rare_conf))

        # 4) Seasonal variation
        if class_centroids and fc.true_label in class_centroids:
            centroid = class_centroids[fc.true_label]
            cls_std = class_stds.get(fc.true_label, None) if class_stds else None
            is_seasonal, seasonal_conf = _detect_seasonal_variation(
                emb, centroid, cls_std, deviation_threshold=deviation_threshold
            )
            if is_seasonal:
                causes_scores.append(("seasonal_variation", seasonal_conf))

        # Pick highest-confidence cause
        if causes_scores:
            causes_scores.sort(key=lambda x: x[1], reverse=True)
            fc.failure_cause = causes_scores[0][0]
            fc.cause_confidence = causes_scores[0][1]
        else:
            # Default: mixed scene with low confidence
            fc.failure_cause = "mixed_scene"
            fc.cause_confidence = 0.2

    return failures


# ============================================================
# 4) Compute statistics
# ============================================================

def compute_failure_statistics(
    failures: List[FailureCase],
    class_names: Optional[List[str]] = None,
    total_samples: Optional[int] = None,
) -> FailureAnalysisResult:
    """
    Compute aggregate failure statistics by class and by cause.

    Args:
        failures: list of classified FailureCase objects
        class_names: all class names (for missing-class handling)
        total_samples: total number of queries (for rate computation)

    Returns:
        FailureAnalysisResult with per-class and per-cause breakdowns
    """
    if class_names is None:
        class_names = sorted(set(fc.true_label for fc in failures))

    total = total_samples if total_samples else len(failures)

    # Per-class statistics
    per_class_stats = {}
    for cls in class_names:
        cls_failures = [fc for fc in failures if fc.true_label == cls]
        n_cls = len(cls_failures)

        # Confusion analysis: what do failures get predicted as?
        confusion = {}
        for fc in cls_failures:
            pred = fc.predicted_label
            confusion[pred] = confusion.get(pred, 0) + 1

        # Average first-hit rank for this class's failures
        hit_ranks = [fc.first_hit_rank for fc in cls_failures if fc.first_hit_rank is not None]
        avg_rank = float(np.mean(hit_ranks)) if hit_ranks else float("inf")

        # Cause breakdown
        cause_counts = {c: 0 for c in FAILURE_CAUSES}
        for fc in cls_failures:
            if fc.failure_cause in cause_counts:
                cause_counts[fc.failure_cause] += 1

        per_class_stats[cls] = {
            "n_failures": n_cls,
            "confusion_top": sorted(confusion.items(), key=lambda x: -x[1]),
            "avg_first_hit_rank": avg_rank,
            "cause_breakdown": cause_counts,
            "mean_ap": float(np.mean([fc.average_precision for fc in cls_failures])) if cls_failures else 0.0,
        }

    # Per-cause statistics
    per_cause_stats = {}
    for cause in FAILURE_CAUSES:
        cause_failures = [fc for fc in failures if fc.failure_cause == cause]
        n_cause = len(cause_failures)
        affected_classes = sorted(set(fc.true_label for fc in cause_failures))

        per_cause_stats[cause] = {
            "count": n_cause,
            "percentage": n_cause / max(len(failures), 1) * 100,
            "affected_classes": affected_classes,
            "avg_confidence": float(np.mean([fc.cause_confidence for fc in cause_failures])) if cause_failures else 0.0,
        }

    threshold = 1  # default
    if failures:
        # Infer from metadata if present
        pass

    return FailureAnalysisResult(
        total_samples=total,
        total_failures=len(failures),
        failure_rate=len(failures) / max(total, 1),
        threshold=threshold,
        failures=failures,
        per_class_stats=per_class_stats,
        per_cause_stats=per_cause_stats,
        class_names=class_names,
    )


# ============================================================
# 5) Compute class centroids and stds from embeddings
# ============================================================

def compute_class_embedding_stats(
    embeddings: torch.Tensor,
    label_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Compute per-class mean embedding (centroid) and intra-class L2 std.

    Args:
        embeddings: (N, B, D) all band embeddings
        label_names: (N,) class name for each sample

    Returns:
        centroids: {class_name: (D,) mean embedding}
        stds: {class_name: float average L2 distance from centroid}
    """
    unique_classes = sorted(set(label_names))
    centroids = {}
    stds = {}

    for cls in unique_classes:
        mask = [i for i, ln in enumerate(label_names) if ln == cls]
        cls_embs = embeddings[mask]  # (N_cls, B, D)

        # Mean across samples and bands
        cls_mean = cls_embs.mean(dim=(0, 1))  # (D,)
        centroids[cls] = cls_mean

        # Std: average L2 distance from each sample's mean to centroid
        sample_means = cls_embs.mean(dim=1)  # (N_cls, D)
        distances = torch.norm(sample_means - cls_mean.unsqueeze(0), dim=1)  # (N_cls,)
        stds[cls] = float(distances.std().item())

    return centroids, stds


# ============================================================
# 6) Visualization
# ============================================================

def plot_failure_cause_distribution(
    result: FailureAnalysisResult,
    *,
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot failure cause distribution: pie chart + per-class bar chart.

    Args:
        result: FailureAnalysisResult
        figsize: figure size
        save_path: optional path to save

    Returns:
        (fig, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---- Left: Pie chart of causes ----
    cause_counts = []
    cause_labels = []
    cause_colors = []

    for cause in FAILURE_CAUSES:
        count = result.per_cause_stats[cause]["count"]
        if count > 0:
            cause_counts.append(count)
            cause_labels.append(
                f"{FAILURE_CAUSE_LABELS[cause]}\n({count}, "
                f"{result.per_cause_stats[cause]['percentage']:.1f}%)"
            )
            cause_colors.append(FAILURE_CAUSE_COLORS[cause])

    if cause_counts:
        wedges, texts, autotexts = axes[0].pie(
            cause_counts,
            labels=cause_labels,
            colors=cause_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")
    else:
        axes[0].text(0.5, 0.5, "No failures", ha="center", va="center",
                     fontsize=14, transform=axes[0].transAxes)

    axes[0].set_title(
        f"Failure Cause Distribution (n={result.total_failures})",
        fontsize=13, fontweight="bold", pad=15,
    )

    # ---- Right: Per-class failure counts stacked by cause ----
    classes = result.class_names
    n_classes = len(classes)
    x = np.arange(n_classes)
    width = 0.6

    bottom = np.zeros(n_classes)
    for cause in FAILURE_CAUSES:
        heights = []
        for cls in classes:
            stats = result.per_class_stats.get(cls, {})
            cb = stats.get("cause_breakdown", {})
            heights.append(cb.get(cause, 0))
        heights = np.array(heights, dtype=float)

        if heights.sum() > 0:
            axes[1].bar(
                x, heights,
                width=width,
                bottom=bottom,
                color=FAILURE_CAUSE_COLORS[cause],
                edgecolor="white",
                linewidth=0.5,
                label=FAILURE_CAUSE_LABELS[cause],
            )
            bottom += heights

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, fontsize=8, rotation=45, ha="right")
    axes[1].set_ylabel("Number of Failures", fontsize=11, fontweight="bold")
    axes[1].set_title(
        "Failures per Class (by Cause)",
        fontsize=13, fontweight="bold", pad=12,
    )
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_failure_gallery(
    failures: List[FailureCase],
    images: Optional[List[torch.Tensor]] = None,
    band_attributions: Optional[List[np.ndarray]] = None,
    *,
    n_show: int = 12,
    n_cols: int = 4,
    figsize_per_row: Tuple[float, float] = (20, 4),
    save_path: Optional[str] = None,
    suptitle: str = "Failure Case Gallery",
) -> Any:
    """
    Plot a gallery of failure cases with RGB image, attribution heatmap,
    and prediction info.

    Each row shows one failure: [RGB image] [Attribution bar] [Info text]

    Args:
        failures: list of FailureCase (should be classified)
        images: list of (13, H, W) multispectral tensors for each failure
        band_attributions: list of (13,) attribution arrays
        n_show: max number of failures to display
        n_cols: columns per row (3 panels per failure)
        figsize_per_row: figure size per row of failures
        save_path: optional path to save
        suptitle: figure title

    Returns:
        (fig, axes) matplotlib objects
    """
    import matplotlib.pyplot as plt
    from src.utils.visualization import stretch_for_display, extract_rgb_bands

    SENTINEL2_BAND_NAMES = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B8A", "B09", "B10", "B11", "B12",
    ]

    n_show = min(n_show, len(failures))
    if n_show == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No failure cases to display",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    # Each failure gets 3 columns: RGB | Attribution bars | Info
    panels_per_failure = 3
    actual_cols = panels_per_failure
    n_rows = n_show

    fig_height = figsize_per_row[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, actual_cols,
        figsize=(figsize_per_row[0], fig_height),
        gridspec_kw={"width_ratios": [1, 1.5, 1]},
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    band_colors = [
        "#7B68EE", "#4169E1", "#2E8B57", "#DC143C",
        "#FF6347", "#FF7F50", "#FF4500", "#8B0000",
        "#A52A2A", "#B0C4DE", "#D3D3D3", "#CD853F", "#8B7355",
    ]

    for row_idx in range(n_rows):
        fc = failures[row_idx]
        ax_rgb = axes[row_idx, 0]
        ax_attr = axes[row_idx, 1]
        ax_info = axes[row_idx, 2]

        # ---- Column 1: RGB Image ----
        if images and row_idx < len(images) and images[row_idx] is not None:
            img = images[row_idx]
            if img.ndim == 3 and img.shape[0] >= 3:
                rgb = extract_rgb_bands(img)  # (3, H, W)
                rgb_display = stretch_for_display(rgb)
                ax_rgb.imshow(rgb_display)
        else:
            ax_rgb.text(0.5, 0.5, "No image\navailable",
                       ha="center", va="center", fontsize=10,
                       transform=ax_rgb.transAxes, color="#888")

        cause_color = FAILURE_CAUSE_COLORS.get(fc.failure_cause, "#888")
        ax_rgb.set_title(
            f"#{fc.sample_index} — {FAILURE_CAUSE_LABELS.get(fc.failure_cause, fc.failure_cause)}",
            fontsize=9, fontweight="bold", color=cause_color,
        )
        ax_rgb.axis("off")

        # ---- Column 2: Attribution Bar Chart ----
        if band_attributions and row_idx < len(band_attributions) and band_attributions[row_idx] is not None:
            attr = band_attributions[row_idx]
            n_bands = len(attr)
            ax_attr.bar(
                range(n_bands), attr,
                color=band_colors[:n_bands],
                edgecolor="white", linewidth=0.5,
                alpha=0.85, width=0.7,
            )
            ax_attr.set_xticks(range(n_bands))
            ax_attr.set_xticklabels(SENTINEL2_BAND_NAMES[:n_bands], fontsize=6, rotation=45)
            ax_attr.set_ylabel("Attribution", fontsize=8)
            ax_attr.set_ylim(0, max(attr.max() * 1.3, 0.1))
        else:
            ax_attr.text(0.5, 0.5, "No attribution\ndata",
                        ha="center", va="center", fontsize=10,
                        transform=ax_attr.transAxes, color="#888")

        ax_attr.spines["top"].set_visible(False)
        ax_attr.spines["right"].set_visible(False)
        ax_attr.grid(axis="y", alpha=0.2, linestyle="--")
        ax_attr.set_title("Band Attribution", fontsize=9, fontweight="bold")

        # ---- Column 3: Info Text ----
        ax_info.axis("off")
        info_lines = [
            f"True:  {fc.true_label}",
            f"Pred:  {fc.predicted_label}",
            f"Sim:   {fc.similarity_score:.3f}",
            f"Rank:  {fc.first_hit_rank if fc.first_hit_rank else '∞'}",
            f"AP:    {fc.average_precision:.3f}",
            f"Cause: {FAILURE_CAUSE_LABELS.get(fc.failure_cause, '?')}",
            f"Conf:  {fc.cause_confidence:.2f}",
        ]
        info_text = "\n".join(info_lines)

        ax_info.text(
            0.1, 0.95, info_text,
            transform=ax_info.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                     edgecolor="#ccc", alpha=0.9),
        )
        ax_info.set_title("Details", fontsize=9, fontweight="bold")

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_confusion_heatmap(
    failures: List[FailureCase],
    class_names: List[str],
    *,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    title: str = "Failure Confusion Matrix (True → Predicted)",
) -> Any:
    """
    Plot confusion heatmap showing which classes get confused with which.

    Only shows failure cases (off-diagonal entries).

    Args:
        failures: classified FailureCase list
        class_names: ordered list of all class names
        figsize: figure size
        save_path: optional save path
        title: figure title

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    n = len(class_names)
    confusion = np.zeros((n, n), dtype=int)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    for fc in failures:
        true_idx = cls_to_idx.get(fc.true_label, None)
        pred_idx = cls_to_idx.get(fc.predicted_label, None)
        if true_idx is not None and pred_idx is not None:
            confusion[true_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(confusion, cmap="Reds", aspect="auto")

    # Annotate
    for i in range(n):
        for j in range(n):
            val = confusion[i, j]
            if val > 0:
                text_color = "white" if val > confusion.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                       fontsize=8, fontweight="bold", color=text_color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Failure Count", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


# ============================================================
# 7) Summary report
# ============================================================

def print_failure_summary(result: FailureAnalysisResult) -> str:
    """
    Pretty-print failure analysis summary.

    Returns the formatted string.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("  FAILURE CASE ANALYSIS SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Total queries:   {result.total_samples}")
    lines.append(f"  Total failures:  {result.total_failures}")
    lines.append(f"  Failure rate:    {result.failure_rate * 100:.1f}%")
    lines.append(f"  Threshold:       R@{int(result.threshold)} miss")

    lines.append("")
    lines.append("-" * 72)
    lines.append("  FAILURE CAUSE BREAKDOWN")
    lines.append("-" * 72)

    for cause in FAILURE_CAUSES:
        stats = result.per_cause_stats[cause]
        label = FAILURE_CAUSE_LABELS[cause]
        count = stats["count"]
        pct = stats["percentage"]
        classes = ", ".join(stats["affected_classes"][:5])
        if len(stats["affected_classes"]) > 5:
            classes += ", ..."
        lines.append(f"  {label:20s}  {count:4d}  ({pct:5.1f}%)  Classes: {classes}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("  PER-CLASS FAILURE DETAILS")
    lines.append("-" * 72)

    for cls in result.class_names:
        stats = result.per_class_stats.get(cls, {})
        n_fail = stats.get("n_failures", 0)
        if n_fail == 0:
            continue

        avg_rank = stats.get("avg_first_hit_rank", float("inf"))
        mean_ap = stats.get("mean_ap", 0.0)
        confused = stats.get("confusion_top", [])

        lines.append(f"")
        lines.append(f"  ▸ {cls}")
        lines.append(f"    Failures: {n_fail}, Avg rank: {avg_rank:.1f}, Mean AP: {mean_ap:.3f}")
        if confused:
            confused_str = ", ".join(f"{pred}({cnt})" for pred, cnt in confused[:3])
            lines.append(f"    Confused with: {confused_str}")

        cb = stats.get("cause_breakdown", {})
        cause_str = ", ".join(
            f"{FAILURE_CAUSE_LABELS.get(c, c)}:{n}"
            for c, n in cb.items() if n > 0
        )
        if cause_str:
            lines.append(f"    Causes: {cause_str}")

    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    print(text)
    return text
