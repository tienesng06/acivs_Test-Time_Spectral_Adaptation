from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.utils.visualization import extract_rgb_bands


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)



def preprocess_rgb_for_clip(rgb_tensor: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Resize and normalize RGB tensor to CLIP input format.

    Args:
        rgb_tensor:
            Tensor with shape [3, H, W] or [N, 3, H, W].
        image_size:
            Target spatial size for CLIP.

    Returns:
        Tensor with shape [N, 3, image_size, image_size].
    """
    if rgb_tensor.ndim == 3:
        rgb_tensor = rgb_tensor.unsqueeze(0)

    if rgb_tensor.ndim != 4 or rgb_tensor.shape[1] != 3:
        raise ValueError(f"Expected [N,3,H,W] or [3,H,W], got shape={tuple(rgb_tensor.shape)}")

    rgb_tensor = rgb_tensor.float().clamp(0.0, 1.0)
    rgb_tensor = F.interpolate(
        rgb_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    mean = CLIP_MEAN.to(device=rgb_tensor.device, dtype=rgb_tensor.dtype)
    std = CLIP_STD.to(device=rgb_tensor.device, dtype=rgb_tensor.dtype)
    rgb_tensor = (rgb_tensor - mean) / std
    return rgb_tensor


@torch.no_grad()
def encode_test_gallery_rgb(
    loader,
    model,
    device: str | torch.device,
    image_size: int = 224,
    rgb_indices: Sequence[int] = (3, 2, 1),
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Encode a multispectral gallery using only RGB channels for CLIP.

    Expected batch format from dataloader:
        {
            "image": [N, 13, H, W],
            "label": [N],
            "label_name": list[str],
            "path": list[str],
        }

    Returns:
        {
            "features":   [N_gallery, D],
            "labels":     [N_gallery],
            "label_names": list[str],
            "paths":       list[str],
        }
    """
    all_features = []
    all_labels = []
    all_label_names = []
    all_paths = []

    iterator = tqdm(loader, desc="Encoding test gallery") if show_progress else loader

    for batch in iterator:
        image_13 = batch["image"]
        rgb = extract_rgb_bands(image_13, rgb_indices=rgb_indices)
        rgb = preprocess_rgb_for_clip(rgb, image_size=image_size).to(device)

        image_features = model.encode_image(rgb).float()
        image_features = F.normalize(image_features, dim=-1)

        all_features.append(image_features.cpu())
        all_labels.append(batch["label"].cpu())
        all_label_names.extend(list(batch["label_name"]))
        all_paths.extend(list(batch["path"]))

    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "label_names": all_label_names,
        "paths": all_paths,
    }
