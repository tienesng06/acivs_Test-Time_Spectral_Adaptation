"""
Band attribution analysis for multispectral retrieval.

Computes the contribution of each spectral band to the final fusion
by combining query-conditioned alignment scores with Fiedler-based
structural importance weights.

Attribution formula:
    attribution[i] = alignment_score[i] × fiedler_weight[i]
    normalized to [0, 1]

Core functions:
    - compute_band_attribution()          : single query attribution
    - compute_class_band_attribution()    : aggregate per-class over dataset
    - plot_band_attribution_bar()         : bar chart for one class
    - plot_band_attribution_heatmap()     : heatmap for all classes × bands
    - analyze_per_class_band_preference() : text analysis of per-class preferences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.models.affinity_graph import compute_query_weights
from src.models.fiedler import compute_fiedler_magnitude_weights
from src.models.affinity_graph import compute_affinity_graph


# ============================================================
# Band metadata (Sentinel-2 MSI)
# ============================================================

SENTINEL2_BAND_NAMES: List[str] = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12",
]

SENTINEL2_BAND_LABELS: List[str] = [
    "B01\nCoastal",
    "B02\nBlue",
    "B03\nGreen",
    "B04\nRed",
    "B05\nRE-1",
    "B06\nRE-2",
    "B07\nRE-3",
    "B08\nNIR",
    "B8A\nNIR-N",
    "B09\nWV",
    "B10\nCirrus",
    "B11\nSWIR-1",
    "B12\nSWIR-2",
]

# Spectral group assignments for interpretability
BAND_SPECTRAL_GROUPS: Dict[str, List[int]] = {
    "Coastal/Aerosol": [0],               # B01
    "Visible (RGB)":   [1, 2, 3],         # B02, B03, B04
    "Red-Edge":        [4, 5, 6],         # B05, B06, B07
    "NIR":             [7, 8],            # B08, B8A
    "Water Vapor":     [9],               # B09
    "Cirrus/SWIR":     [10, 11, 12],      # B10, B11, B12
}

# Physical interpretation for each class-band association
BAND_CLASS_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "Forest": {
        "dominant": "NIR (B08, B8A) + Red-Edge (B05-B07)",
        "reason": (
            "Healthy vegetation reflects strongly in NIR due to leaf "
            "cell structure (mesophyll scattering). Red-Edge bands capture "
            "the sharp reflectance transition between red absorption and "
            "NIR plateau — a signature unique to chlorophyll-rich canopies."
        ),
    },
    "HerbaceousVegetation": {
        "dominant": "NIR (B08) + Red-Edge (B05-B07) + Green (B03)",
        "reason": (
            "Similar to forest but with higher green reflectance due to "
            "less canopy layering. The NIR plateau is slightly lower than "
            "dense forest. Green peak (B03) is more pronounced in open "
            "herbaceous cover compared to dense forest."
        ),
    },
    "AnnualCrop": {
        "dominant": "Red-Edge (B05-B07) + NIR (B08) + SWIR (B11-B12)",
        "reason": (
            "Crop fields show seasonal variation captured by Red-Edge "
            "sensitivity to chlorophyll concentration changes. SWIR bands "
            "detect soil moisture and crop water stress, differentiating "
            "irrigated vs. rain-fed agriculture."
        ),
    },
    "PermanentCrop": {
        "dominant": "NIR (B08, B8A) + Red-Edge (B06-B07)",
        "reason": (
            "Perennial crops (orchards, vineyards) have stable canopy "
            "structure with strong NIR reflectance year-round. Red-Edge "
            "bands help distinguish crop health and species differences."
        ),
    },
    "Pasture": {
        "dominant": "NIR (B08) + Green (B03) + Red-Edge (B05)",
        "reason": (
            "Grassland pastures have moderate NIR reflectance (lower than "
            "forest), strong green peak, and lower Red-Edge gradients. "
            "The combination distinguishes managed grasslands from both "
            "natural vegetation and cropland."
        ),
    },
    "River": {
        "dominant": "Blue (B02) + Green (B03) + SWIR (B11-B12)",
        "reason": (
            "Water absorbs strongly in NIR/SWIR (near-zero reflectance), "
            "while reflecting in blue/green. The contrast between high "
            "visible and low SWIR is the basis of NDWI. Turbidity and "
            "depth variations are captured by the Blue-Green ratio."
        ),
    },
    "SeaLake": {
        "dominant": "Blue (B02) + Coastal (B01) + Green (B03)",
        "reason": (
            "Open water reflects most in short-wavelength bands. Coastal "
            "aerosol band (B01) penetrates water deeper, useful for "
            "bathymetry and chlorophyll-a detection. Clean water reflects "
            "almost nothing beyond 700 nm."
        ),
    },
    "Highway": {
        "dominant": "SWIR (B11-B12) + Red (B04) + NIR (B08)",
        "reason": (
            "Asphalt and concrete have moderate, relatively flat spectral "
            "reflectance. SWIR bands distinguish impervious surfaces from "
            "vegetation. The absence of a strong vegetation signal (low "
            "Red-Edge gradient) is key."
        ),
    },
    "Industrial": {
        "dominant": "SWIR (B11-B12) + Red (B04) + Coastal (B01)",
        "reason": (
            "Industrial areas mix metal roofs, concrete, and bare soil. "
            "SWIR is sensitive to these materials' mineral composition. "
            "The wide spectral variability within industrial scenes "
            "requires multiple bands for robust discrimination."
        ),
    },
    "Residential": {
        "dominant": "SWIR (B11-B12) + Red (B04) + Green (B03)",
        "reason": (
            "Residential areas mix impervious surfaces (roofs, roads) with "
            "urban vegetation. SWIR separates built-up from greenery, while "
            "visible bands capture the heterogeneous texture of neighborhoods."
        ),
    },
}


# ============================================================
# 1) Core attribution computation
# ============================================================

@dataclass
class BandAttribution:
    """Result of compute_band_attribution()."""

    alignment_scores: torch.Tensor      # (B,) softmax query weights
    fiedler_weights: torch.Tensor       # (B,) Fiedler magnitude weights
    raw_attribution: torch.Tensor       # (B,) alignment × fiedler
    normalized_attribution: torch.Tensor  # (B,) min-max normalized to [0,1]
    band_names: List[str] = field(default_factory=lambda: list(SENTINEL2_BAND_NAMES))


def compute_band_attribution(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    *,
    sigma: float = 0.5,
    normalize_inputs: bool = True,
    normalized_laplacian: bool = True,
) -> BandAttribution:
    """
    Compute per-band attribution = alignment_score × fiedler_weight.

    The alignment score measures how relevant each band is to the query,
    while the Fiedler weight captures the structural importance of each
    band in the spectral affinity graph.  Their product gives the
    attributon: bands that are both query-relevant AND structurally
    important receive high scores.

    Args:
        band_embeddings: (B, D) per-band CLIP embeddings
        query_embedding: (D,) or (1, D) query embedding
        sigma: temperature for computing alignment scores
        normalize_inputs: L2-normalize embeddings before dot products
        normalized_laplacian: use normalized Laplacian for Fiedler

    Returns:
        BandAttribution with all intermediate and final attribution scores
    """
    if band_embeddings.ndim != 2:
        raise ValueError(
            f"band_embeddings must be 2D (B, D), got {tuple(band_embeddings.shape)}"
        )

    B = band_embeddings.shape[0]

    # Step 1: Compute query alignment scores (softmax-normalized)
    alignment_scores = compute_query_weights(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        sigma=sigma,
        normalize_inputs=normalize_inputs,
    )  # (B,)

    # Step 2: Compute affinity graph → Fiedler weights
    A_norm = compute_affinity_graph(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        sigma=sigma,
        normalize_inputs=normalize_inputs,
        return_details=False,
    )

    fiedler_weights = compute_fiedler_magnitude_weights(
        A=A_norm,
        normalized=normalized_laplacian,
    )  # (B,)

    # Step 3: Attribution = alignment × fiedler
    raw_attribution = alignment_scores * fiedler_weights

    # Step 4: Min-max normalize to [0, 1]
    attr_min = raw_attribution.min()
    attr_max = raw_attribution.max()
    denom = attr_max - attr_min
    if denom < 1e-10:
        normalized_attribution = torch.ones_like(raw_attribution) / B
    else:
        normalized_attribution = (raw_attribution - attr_min) / denom

    # Determine band names
    band_names = list(SENTINEL2_BAND_NAMES[:B])

    return BandAttribution(
        alignment_scores=alignment_scores.detach(),
        fiedler_weights=fiedler_weights.detach(),
        raw_attribution=raw_attribution.detach(),
        normalized_attribution=normalized_attribution.detach(),
        band_names=band_names,
    )


# ============================================================
# 2) Class-level aggregation
# ============================================================

@dataclass
class ClassBandAttribution:
    """Aggregated attribution across multiple samples for each class."""

    # class_name → (B,) mean normalized attribution
    class_attributions: Dict[str, np.ndarray]
    # class_name → (B,) standard deviation
    class_std: Dict[str, np.ndarray]
    # class_name → number of samples
    class_counts: Dict[str, int]
    # Ordered list of class names
    class_names: List[str]
    # Band names
    band_names: List[str]


def compute_class_band_attribution(
    band_embeddings_list: List[torch.Tensor],
    query_embeddings_list: List[torch.Tensor],
    class_labels: List[str],
    *,
    sigma: float = 0.5,
    normalize_inputs: bool = True,
    normalized_laplacian: bool = True,
) -> ClassBandAttribution:
    """
    Aggregate band attribution over a dataset, grouped by class.

    Args:
        band_embeddings_list: list of (B, D) tensors, one per sample
        query_embeddings_list: list of (D,) tensors, one per sample
        class_labels: list of class name strings, one per sample
        sigma, normalize_inputs, normalized_laplacian: forwarded to
            compute_band_attribution()

    Returns:
        ClassBandAttribution with per-class mean ± std attributions
    """
    if len(band_embeddings_list) != len(class_labels):
        raise ValueError(
            f"Length mismatch: {len(band_embeddings_list)} embeddings vs "
            f"{len(class_labels)} labels"
        )
    if len(query_embeddings_list) != len(class_labels):
        raise ValueError(
            f"Length mismatch: {len(query_embeddings_list)} queries vs "
            f"{len(class_labels)} labels"
        )

    # Collect attributions per class
    class_attr_accum: Dict[str, List[np.ndarray]] = {}

    for band_emb, query_emb, label in zip(
        band_embeddings_list, query_embeddings_list, class_labels
    ):
        attr = compute_band_attribution(
            band_embeddings=band_emb,
            query_embedding=query_emb,
            sigma=sigma,
            normalize_inputs=normalize_inputs,
            normalized_laplacian=normalized_laplacian,
        )

        if label not in class_attr_accum:
            class_attr_accum[label] = []
        class_attr_accum[label].append(attr.normalized_attribution.numpy())

    # Compute mean ± std per class
    class_names = sorted(class_attr_accum.keys())
    class_attributions = {}
    class_std = {}
    class_counts = {}

    B = band_embeddings_list[0].shape[0]
    band_names = list(SENTINEL2_BAND_NAMES[:B])

    for cls in class_names:
        stacked = np.stack(class_attr_accum[cls], axis=0)  # (N_cls, B)
        class_attributions[cls] = stacked.mean(axis=0)
        class_std[cls] = stacked.std(axis=0)
        class_counts[cls] = stacked.shape[0]

    return ClassBandAttribution(
        class_attributions=class_attributions,
        class_std=class_std,
        class_counts=class_counts,
        class_names=class_names,
        band_names=band_names,
    )


def compute_class_band_attribution_from_pipeline(
    band_embeddings_all: torch.Tensor,
    query_embeddings_all: torch.Tensor,
    class_labels: List[str],
    *,
    sigma: float = 0.5,
    normalize_inputs: bool = True,
    normalized_laplacian: bool = True,
) -> ClassBandAttribution:
    """
    Convenience wrapper when embeddings are stacked as tensors.

    Args:
        band_embeddings_all: (N, B, D) all sample band embeddings
        query_embeddings_all: (N, D) all query embeddings
        class_labels: (N,) class names

    Returns:
        ClassBandAttribution
    """
    N = band_embeddings_all.shape[0]
    band_list = [band_embeddings_all[i] for i in range(N)]
    query_list = [query_embeddings_all[i] for i in range(N)]

    return compute_class_band_attribution(
        band_embeddings_list=band_list,
        query_embeddings_list=query_list,
        class_labels=class_labels,
        sigma=sigma,
        normalize_inputs=normalize_inputs,
        normalized_laplacian=normalized_laplacian,
    )


# ============================================================
# 3) Visualization
# ============================================================

def _get_class_color(class_name: str) -> str:
    """Map class name to a thematic color."""
    color_map = {
        "Forest":                "#228B22",
        "HerbaceousVegetation":  "#7CFC00",
        "AnnualCrop":            "#DAA520",
        "PermanentCrop":         "#8B4513",
        "Pasture":               "#9ACD32",
        "River":                 "#4169E1",
        "SeaLake":               "#00CED1",
        "Highway":               "#808080",
        "Industrial":            "#FF6347",
        "Residential":           "#FF8C00",
    }
    return color_map.get(class_name, "#888888")


def _get_band_colors(n_bands: int = 13) -> List[str]:
    """Color palette for 13 Sentinel-2 bands by spectral region."""
    colors = [
        "#7B68EE",  # B01 Coastal – violet
        "#4169E1",  # B02 Blue
        "#2E8B57",  # B03 Green
        "#DC143C",  # B04 Red
        "#FF6347",  # B05 Red-Edge 1
        "#FF7F50",  # B06 Red-Edge 2
        "#FF4500",  # B07 Red-Edge 3
        "#8B0000",  # B08 NIR
        "#A52A2A",  # B08A NIR Narrow
        "#B0C4DE",  # B09 Water Vapor
        "#D3D3D3",  # B10 Cirrus
        "#CD853F",  # B11 SWIR-1
        "#8B7355",  # B12 SWIR-2
    ]
    return colors[:n_bands]


def plot_band_attribution_bar(
    attribution: Union[BandAttribution, np.ndarray],
    *,
    class_name: str = "",
    ax=None,
    figsize: Tuple[float, float] = (12, 5),
    show_values: bool = True,
    save_path: Optional[str] = None,
    band_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot bar chart of band attribution for a single class or query.

    Args:
        attribution: BandAttribution object or (B,) numpy array
        class_name: class label for title
        ax: existing matplotlib axes (optional)
        figsize: figure size if creating new figure
        show_values: annotate bars with numeric values
        save_path: if provided, save figure to this path
        band_labels: custom x-axis labels
        title: custom title override
    """
    import matplotlib.pyplot as plt

    if isinstance(attribution, BandAttribution):
        values = attribution.normalized_attribution.numpy()
        default_labels = [SENTINEL2_BAND_LABELS[i] for i in range(len(values))]
    else:
        values = np.asarray(attribution)
        default_labels = [SENTINEL2_BAND_LABELS[i] for i in range(len(values))]

    labels = band_labels if band_labels is not None else default_labels
    colors = _get_band_colors(len(values))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white",
                  linewidth=0.8, alpha=0.9, width=0.7)

    if show_values:
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("Attribution Score", fontsize=11, fontweight="bold")
    ax.set_ylim(0, min(values.max() * 1.25, 1.05))

    if title is not None:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    elif class_name:
        ax.set_title(
            f"Band Attribution — {class_name}",
            fontsize=13, fontweight="bold", pad=12,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        return fig, ax

    return ax


def plot_all_classes_bar(
    class_attr: ClassBandAttribution,
    *,
    figsize: Tuple[float, float] = (18, 24),
    save_path: Optional[str] = None,
    suptitle: str = "Band Attribution per EuroSAT Class",
):
    """
    Plot a grid of bar charts, one per class.

    Args:
        class_attr: ClassBandAttribution from compute_class_band_attribution
        figsize: figure size
        save_path: optional save path
        suptitle: overall figure title
    """
    import matplotlib.pyplot as plt

    n_classes = len(class_attr.class_names)
    n_cols = 2
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, cls in enumerate(class_attr.class_names):
        ax = axes[i]
        values = class_attr.class_attributions[cls]
        stds = class_attr.class_std[cls]
        n_samples = class_attr.class_counts[cls]
        colors = _get_band_colors(len(values))
        labels = [SENTINEL2_BAND_LABELS[j] for j in range(len(values))]

        bars = ax.bar(
            range(len(values)), values,
            yerr=stds, capsize=3,
            color=colors, edgecolor="white",
            linewidth=0.5, alpha=0.85, width=0.7,
        )

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=6, ha="center")
        ax.set_ylabel("Attribution", fontsize=9)
        ax.set_title(
            f"{cls}  (n={n_samples})",
            fontsize=11, fontweight="bold",
            color=_get_class_color(cls),
        )
        ax.set_ylim(0, min(values.max() * 1.4, 1.05))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Hide unused axes
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_band_attribution_heatmap(
    class_attr: ClassBandAttribution,
    *,
    figsize: Tuple[float, float] = (14, 7),
    save_path: Optional[str] = None,
    cmap: str = "YlOrRd",
    annotate: bool = True,
    title: str = "Band Attribution Heatmap (Class × Band)",
):
    """
    Plot a heatmap of attribution scores: classes (rows) × bands (columns).

    Args:
        class_attr: ClassBandAttribution
        figsize: figure size
        save_path: optional path to save figure
        cmap: matplotlib colormap name
        annotate: show numeric values in cells
        title: figure title
    """
    import matplotlib.pyplot as plt

    n_classes = len(class_attr.class_names)
    n_bands = len(class_attr.band_names)

    # Build attribution matrix (classes × bands)
    matrix = np.zeros((n_classes, n_bands))
    for i, cls in enumerate(class_attr.class_names):
        matrix[i, :] = class_attr.class_attributions[cls]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    # Annotate cells
    if annotate:
        for i in range(n_classes):
            for j in range(n_bands):
                val = matrix[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color=text_color,
                )

    # Axis labels
    band_short_labels = [SENTINEL2_BAND_NAMES[k] for k in range(n_bands)]
    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(band_short_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_attr.class_names, fontsize=10, fontweight="bold")

    ax.set_xlabel("Spectral Band", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Land-Cover Class", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Attribution Score", fontsize=10, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_spectral_group_summary(
    class_attr: ClassBandAttribution,
    *,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
    title: str = "Spectral Group Importance by Class",
):
    """
    Aggregate attributions by spectral group and plot stacked bars.

    Groups: Coastal/Aerosol, Visible (RGB), Red-Edge, NIR,
            Water Vapor, Cirrus/SWIR.
    """
    import matplotlib.pyplot as plt

    classes = class_attr.class_names
    group_names = list(BAND_SPECTRAL_GROUPS.keys())
    n_classes = len(classes)
    n_groups = len(group_names)

    # Aggregate: for each class, sum attribution within each spectral group
    group_values = np.zeros((n_classes, n_groups))
    for i, cls in enumerate(classes):
        attr = class_attr.class_attributions[cls]
        for g, group_name in enumerate(group_names):
            band_indices = BAND_SPECTRAL_GROUPS[group_name]
            group_values[i, g] = attr[band_indices].sum()

    # Normalize per class so groups sum to 1
    row_sums = group_values.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-8, None)
    group_values_norm = group_values / row_sums

    # Stacked horizontal bar chart
    group_colors = [
        "#7B68EE",  # Coastal
        "#4169E1",  # Visible
        "#FF6347",  # Red-Edge
        "#8B0000",  # NIR
        "#B0C4DE",  # Water Vapor
        "#CD853F",  # Cirrus/SWIR
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n_classes)

    left = np.zeros(n_classes)
    for g in range(n_groups):
        ax.barh(
            y_pos, group_values_norm[:, g],
            left=left,
            color=group_colors[g],
            edgecolor="white",
            linewidth=0.5,
            label=group_names[g],
            height=0.65,
        )
        left += group_values_norm[:, g]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalized Group Contribution", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, fontsize=9, frameon=False,
    )
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


# ============================================================
# 4) Per-class textual analysis
# ============================================================

def analyze_per_class_band_preference(
    class_attr: ClassBandAttribution,
    top_k: int = 3,
) -> Dict[str, Dict[str, object]]:
    """
    For each class, identify top-K most important bands and provide
    physical interpretation.

    Args:
        class_attr: ClassBandAttribution from compute_class_band_attribution
        top_k: how many top bands to highlight

    Returns:
        Dict[class_name] -> {
            "top_bands": [(band_name, score), ...],
            "bottom_bands": [(band_name, score), ...],
            "dominant_region": str,
            "explanation": str,
        }
    """
    analysis = {}

    for cls in class_attr.class_names:
        attr = class_attr.class_attributions[cls]
        band_names = class_attr.band_names
        n_bands = len(band_names)

        # Sort bands by attribution descending
        sorted_indices = np.argsort(attr)[::-1]

        top_bands = [
            (band_names[idx], float(attr[idx]))
            for idx in sorted_indices[:top_k]
        ]
        bottom_bands = [
            (band_names[idx], float(attr[idx]))
            for idx in sorted_indices[-top_k:]
        ]

        # Identify dominant spectral region
        group_sums = {}
        for group_name, indices in BAND_SPECTRAL_GROUPS.items():
            valid_indices = [i for i in indices if i < n_bands]
            if valid_indices:
                group_sums[group_name] = attr[valid_indices].sum()
        dominant_region = max(group_sums, key=group_sums.get)

        # Get physical explanation
        cls_info = BAND_CLASS_EXPLANATIONS.get(cls, {})
        explanation = cls_info.get(
            "reason",
            f"No predefined explanation for {cls}. "
            f"Top bands: {', '.join(b for b, _ in top_bands)}."
        )
        dominant_bands_text = cls_info.get(
            "dominant",
            f"Dominant region: {dominant_region}"
        )

        analysis[cls] = {
            "top_bands": top_bands,
            "bottom_bands": bottom_bands,
            "dominant_region": dominant_region,
            "dominant_bands_text": dominant_bands_text,
            "explanation": explanation,
            "n_samples": class_attr.class_counts[cls],
        }

    return analysis


def print_per_class_analysis(
    analysis: Dict[str, Dict[str, object]],
    verbose: bool = True,
) -> str:
    """
    Pretty-print the per-class band preference analysis.

    Returns the formatted string.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("  PER-CLASS BAND ATTRIBUTION ANALYSIS")
    lines.append("=" * 72)

    for cls, info in sorted(analysis.items()):
        lines.append("")
        lines.append(f"▸ {cls}  (n={info['n_samples']} samples)")
        lines.append(f"  Dominant region: {info['dominant_region']}")
        lines.append(f"  Key bands: {info['dominant_bands_text']}")
        lines.append(f"  Top-3: {', '.join(f'{b}={s:.3f}' for b, s in info['top_bands'])}")
        lines.append(f"  Bottom-3: {', '.join(f'{b}={s:.3f}' for b, s in info['bottom_bands'])}")

        if verbose:
            # Wrap explanation text
            explanation = info["explanation"]
            lines.append(f"  Physical reason:")
            # Simple line wrapping at ~65 chars
            words = explanation.split()
            current_line = "    "
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    lines.append(current_line)
                    current_line = "    " + word
                else:
                    current_line += " " + word if current_line.strip() else "    " + word
            if current_line.strip():
                lines.append(current_line)

        lines.append("-" * 72)

    text = "\n".join(lines)
    print(text)
    return text


# ============================================================
# 5) Helper: compute from HDF5 cache
# ============================================================

def compute_attribution_from_h5(
    h5_path: Union[str, Path],
    class_text_map: Optional[Dict[str, str]] = None,
    text_template: str = "A satellite image of {class_text}.",
    clip_model=None,
    clip_tokenize_fn=None,
    device: Optional[str] = None,
    sigma: float = 0.5,
    max_samples_per_class: Optional[int] = None,
) -> ClassBandAttribution:
    """
    Compute class-level band attributions from cached HDF5 embeddings.

    This is the most efficient way to run attribution analysis on a
    full dataset — it reads pre-computed per-band embeddings and
    generates text query embeddings via CLIP.

    Args:
        h5_path: path to band_embeddings.h5
        class_text_map: mapping class_name → text description
        text_template: template for text queries
        clip_model: CLIP model for text encoding
        clip_tokenize_fn: CLIP tokenizer function
        device: torch device string
        sigma: temperature for alignment scores
        max_samples_per_class: limit samples per class (for speed)

    Returns:
        ClassBandAttribution
    """
    import h5py

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 cache not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        embeddings = torch.from_numpy(f["embeddings"][:])   # (N, 13, 512)
        labels = f["labels"][:]                              # (N,)
        label_names = [s.decode("utf-8") if isinstance(s, bytes) else s
                       for s in f["label_names"][:]]         # (N,)

    # Build text query embeddings per class
    unique_classes = sorted(set(label_names))

    if class_text_map is None:
        # Default EuroSAT text map
        from src.datasets.eurosat import CLASS_TEXT_MAP
        class_text_map = CLASS_TEXT_MAP

    # Get query embeddings
    if clip_model is not None and clip_tokenize_fn is not None:
        import torch

        if device is None:
            device = "cpu"

        class_query_embeddings = {}
        for cls in unique_classes:
            text = class_text_map.get(cls, cls)
            full_text = text_template.format(class_text=text)
            with torch.no_grad():
                tokens = clip_tokenize_fn([full_text]).to(device)
                q_emb = clip_model.encode_text(tokens).float()
                q_emb = F.normalize(q_emb, dim=-1)
                class_query_embeddings[cls] = q_emb.squeeze(0).cpu()
    else:
        # Use mean band embedding as proxy query
        class_query_embeddings = {}
        for cls in unique_classes:
            mask = [i for i, ln in enumerate(label_names) if ln == cls]
            cls_embeddings = embeddings[mask]  # (N_cls, 13, 512)
            mean_emb = cls_embeddings.mean(dim=(0, 1))  # (512,)
            class_query_embeddings[cls] = F.normalize(mean_emb, dim=0)

    # Build per-sample lists
    band_emb_list = []
    query_emb_list = []
    class_label_list = []

    for cls in unique_classes:
        mask = [i for i, ln in enumerate(label_names) if ln == cls]
        if max_samples_per_class is not None:
            mask = mask[:max_samples_per_class]

        for idx in mask:
            band_emb_list.append(embeddings[idx])  # (13, 512)
            query_emb_list.append(class_query_embeddings[cls])
            class_label_list.append(cls)

    return compute_class_band_attribution(
        band_embeddings_list=band_emb_list,
        query_embeddings_list=query_emb_list,
        class_labels=class_label_list,
        sigma=sigma,
    )
