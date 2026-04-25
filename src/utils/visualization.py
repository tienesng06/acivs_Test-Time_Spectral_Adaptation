from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import rasterio
import torch

PathLike = Union[str, Path]

# EuroSAT-MS / Sentinel-2 convention used in your notebooks:
# image shape = [13, H, W], RGB = B04, B03, B02
DEFAULT_RGB_INDICES: Sequence[int] = (3, 2, 1)

__all__ = [
    "extract_rgb_bands",
    "stretch_for_display",
    "load_rgb_from_tif",
    "plot_multispectral_bands",
    "plot_query_and_topk",
    "plot_multiband_comparison",
]


def extract_rgb_bands(
    image: torch.Tensor,
    rgb_indices: Sequence[int] = DEFAULT_RGB_INDICES,
) -> torch.Tensor:
    """
    Extract RGB bands from a 13-channel multispectral tensor.

    Args:
        image:
            Tensor with shape [13, H, W] or [N, 13, H, W].
        rgb_indices:
            Zero-based indices for the RGB channels.
            Default = (3, 2, 1) -> B04, B03, B02.

    Returns:
        Tensor with shape [3, H, W] or [N, 3, H, W].
    """
    if image.ndim == 3:
        return image[list(rgb_indices), :, :]
    if image.ndim == 4:
        return image[:, list(rgb_indices), :, :]
    raise ValueError(f"Expected [13,H,W] or [N,13,H,W], got shape={tuple(image.shape)}")



def stretch_for_display(rgb_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert an RGB tensor into a display-friendly HWC numpy array.

    This mirrors the percentile stretch logic from your notebook.

    Args:
        rgb_tensor: Tensor with shape [3, H, W].

    Returns:
        Numpy array with shape [H, W, 3], clipped to [0, 1].
    """
    if rgb_tensor.ndim != 3 or rgb_tensor.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor [3,H,W], got shape={tuple(rgb_tensor.shape)}")

    rgb = rgb_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)
    rgb = np.clip((rgb - low) / (high - low + 1e-8), 0.0, 1.0)
    return rgb



def load_rgb_from_tif(
    path: PathLike,
    reflectance_scale: float = 10000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
    rgb_indices: Sequence[int] = DEFAULT_RGB_INDICES,
) -> torch.Tensor:
    """
    Load a EuroSAT-MS .tif file and return RGB channels only.

    Args:
        path:
            Path to a 13-band .tif image.
        reflectance_scale:
            Scale factor used to convert uint16 reflectance values.
        clamp_range:
            Min/max clamp applied after scaling.
        rgb_indices:
            Zero-based channel indices for RGB extraction.

    Returns:
        Tensor with shape [3, H, W].
    """
    path = Path(path)

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32) / reflectance_scale

    image = np.clip(image, clamp_range[0], clamp_range[1])
    image = torch.from_numpy(image)  # [13, H, W]
    rgb = extract_rgb_bands(image, rgb_indices=rgb_indices)
    return rgb


# ============================================================
# New visualization functions (Week 1 plan requirements)
# ============================================================

def plot_multispectral_bands(
    image: torch.Tensor,
    band_names: Sequence[str],
    max_cols: int = 4,
    figsize_per_band: tuple[float, float] = (3.0, 3.0),
    percentile: float = 2.0,
    cmap: str = "gray",
):
    """
    Visualize each spectral band as a separate grayscale subplot.

    Args:
        image: Tensor [B, H, W] — multispectral image.
        band_names: Sequence of B band name strings (e.g. ["B01", "B02", ...]).
        max_cols: Maximum number of subplot columns per row.
        figsize_per_band: (width, height) in inches per subplot cell.
        percentile: Percentile to use for contrast stretching (applied per band).
        cmap: Matplotlib colormap name for single-band display.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    image = image.detach().cpu().float()
    B = image.shape[0]
    if len(band_names) != B:
        raise ValueError(f"band_names length ({len(band_names)}) must match image bands ({B})")

    n_cols = min(B, max_cols)
    n_rows = math.ceil(B / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_band[0] * n_cols, figsize_per_band[1] * n_rows),
        squeeze=False,
    )

    for i in range(B):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        band_np = image[i].numpy()
        lo = np.percentile(band_np, percentile)
        hi = np.percentile(band_np, 100 - percentile)
        stretched = np.clip((band_np - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        ax.imshow(stretched, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(band_names[i], fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for i in range(B, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig


def plot_query_and_topk(
    query_image: torch.Tensor,
    topk_images: List[torch.Tensor],
    query_label: str,
    topk_labels: List[str],
    topk_scores: List[float],
    rgb_indices: Sequence[int] = DEFAULT_RGB_INDICES,
    title: Optional[str] = "Query + Top-K Retrievals",
    figsize_per_image: tuple[float, float] = (3.0, 3.5),
):
    """
    Display the query image alongside top-K retrieved gallery images.

    Correct retrievals (matching label) are framed in green; incorrect in red.

    Args:
        query_image: Tensor [C, H, W] — the query multispectral image.
        topk_images: List of K tensors [C, H, W] — retrieved gallery images.
        query_label: String label for the query image.
        topk_labels: List of K string labels for retrieved images.
        topk_scores: List of K cosine similarity scores.
        rgb_indices: Band indices for RGB composite (default: B04, B03, B02).
        title: Overall figure title (None to suppress).
        figsize_per_image: (width, height) inches per cell.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    k = len(topk_images)
    n_cols = k + 1  # query + k retrievals
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1]),
    )

    def _to_rgb_display(img_tensor: torch.Tensor) -> np.ndarray:
        rgb = extract_rgb_bands(img_tensor.detach().cpu().float(), rgb_indices)
        return stretch_for_display(rgb)

    # Query image
    axes[0].imshow(_to_rgb_display(query_image))
    axes[0].set_title(f"Query\n{query_label}", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#2196F3")
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Retrieved images
    for i, (img, label, score) in enumerate(zip(topk_images, topk_labels, topk_scores)):
        ax = axes[i + 1]
        ax.imshow(_to_rgb_display(img))
        correct = (label == query_label)
        border_color = "#4CAF50" if correct else "#F44336"
        ax.set_title(f"Rank {i+1}\n{label}\nsim={score:.3f}", fontsize=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_multiband_comparison(
    images: List[torch.Tensor],
    titles: List[str],
    rgb_indices: Sequence[int] = DEFAULT_RGB_INDICES,
    figsize_per_image: tuple[float, float] = (4.0, 4.0),
):
    """
    Side-by-side RGB composite comparison of multiple images.

    Useful for comparing retrieved results across different methods or
    for before/after visualization.

    Args:
        images: List of tensors [C, H, W] to compare.
        titles: List of title strings, one per image.
        rgb_indices: Band indices for RGB composite.
        figsize_per_image: (width, height) inches per subplot.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if len(images) != len(titles):
        raise ValueError(
            f"images ({len(images)}) and titles ({len(titles)}) must have the same length"
        )

    n = len(images)
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_image[0] * n, figsize_per_image[1]),
        squeeze=False,
    )

    def _to_rgb_display(img_tensor: torch.Tensor) -> np.ndarray:
        rgb = extract_rgb_bands(img_tensor.detach().cpu().float(), rgb_indices)
        return stretch_for_display(rgb)

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[0][i]
        ax.imshow(_to_rgb_display(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    return fig
