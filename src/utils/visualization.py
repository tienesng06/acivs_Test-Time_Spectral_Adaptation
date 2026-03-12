from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import rasterio
import torch

PathLike = Union[str, Path]

# EuroSAT-MS / Sentinel-2 convention used in your notebooks:
# image shape = [13, H, W], RGB = B04, B03, B02
DEFAULT_RGB_INDICES: Sequence[int] = (3, 2, 1)


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
