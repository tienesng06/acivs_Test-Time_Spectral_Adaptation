from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


CLIP_MEAN = torch.tensor([0.48145466, 0.45782750, 0.40821073], dtype=torch.float32)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)


def get_device() -> torch.device:
    """
    MPS -> CUDA -> GPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_band_stack(
    image_13band: torch.Tensor,
    target_size: int = 224,
    reflectance_scale: float = 10000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Convert 1 ảnh multispectral (13, H, W) thành tensor CLIP-ready (13, 3, 224, 224).

    Steps:
    1. float32
    2. scale về [0,1] nếu dữ liệu đang là reflectance uint16-like
    3. lặp mỗi band thành 3 channel
    4. resize lên 224x224
    5. CLIP normalization
    """
    if image_13band.ndim != 3:
        raise ValueError(f"Expected shape (13, H, W), got {tuple(image_13band.shape)}")

    if image_13band.shape[0] != 13:
        raise ValueError(f"Expected 13 bands, got {image_13band.shape[0]}")

    x = image_13band.to(torch.float32)

    # Nếu dữ liệu chưa nằm trong [0,1], giả sử đang ở scale reflectance ~10000
    if x.max() > 1.0:
        x = x / reflectance_scale

    x = x.clamp(clamp_range[0], clamp_range[1])

    # (13, H, W) -> (13, 1, H, W) -> (13, 3, H, W)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)

    # Resize cho CLIP
    x = F.interpolate(
        x,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    # CLIP normalization
    mean = CLIP_MEAN.view(1, 3, 1, 1)
    std = CLIP_STD.view(1, 3, 1, 1)
    x = (x - mean) / std

    return x


@torch.inference_mode()
def encode_multispectral_bands(
    image_13band: torch.Tensor,
    clip_model,
    device: Optional[torch.device] = None,
    target_size: int = 224,
    reflectance_scale: float = 10000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
    l2_normalize: bool = True,
) -> torch.Tensor:
    """
    Encode 1 ảnh multispectral thành embedding per-band.

    Args:
        image_13band: tensor shape (13, H, W)
        clip_model: CLIP model đã load
        device: torch.device
    Returns:
        band_embeddings: tensor shape (13, 512)
    """
    if device is None:
        device = get_device()

    clip_model.eval()

    x = preprocess_band_stack(
        image_13band=image_13band,
        target_size=target_size,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
    ).to(device)

    # x shape: (13, 3, 224, 224)
    feats = clip_model.encode_image(x).float()  # (13, 512) với ViT-B/16

    if l2_normalize:
        feats = F.normalize(feats, dim=-1)

    return feats.cpu()


def preprocess_band_batch(
    images: torch.Tensor,
    target_size: int = 224,
    reflectance_scale: float = 10000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Convert batch multispectral:
        (N, 13, H, W) -> (N*13, 3, 224, 224)
    """
    if images.ndim != 4:
        raise ValueError(f"Expected shape (N, 13, H, W), got {tuple(images.shape)}")

    if images.shape[1] != 13:
        raise ValueError(f"Expected 13 bands, got {images.shape[1]}")

    x = images.to(torch.float32)

    if x.max() > 1.0:
        x = x / reflectance_scale

    x = x.clamp(clamp_range[0], clamp_range[1])

    # (N,13,H,W) -> (N,13,1,H,W) -> (N,13,3,H,W)
    x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)

    # gộp N và 13 band lại để encode hiệu quả hơn
    n, b, c, h, w = x.shape
    x = x.view(n * b, c, h, w)

    x = F.interpolate(
        x,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    mean = CLIP_MEAN.view(1, 3, 1, 1)
    std = CLIP_STD.view(1, 3, 1, 1)
    x = (x - mean) / std

    return x


@torch.inference_mode()
def encode_multispectral_batch(
    images: torch.Tensor,
    clip_model,
    device: Optional[torch.device] = None,
    target_size: int = 224,
    reflectance_scale: float = 10000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
    micro_batch_size: int = 64,
    l2_normalize: bool = True,
) -> torch.Tensor:
    """
    Encode batch ảnh multispectral.

    Args:
        images: (N, 13, H, W)
    Returns:
        embeddings: (N, 13, 512)
    """
    if device is None:
        device = get_device()

    clip_model.eval()

    x = preprocess_band_batch(
        images=images,
        target_size=target_size,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
    )

    all_feats = []
    for start in range(0, x.shape[0], micro_batch_size):
        chunk = x[start:start + micro_batch_size].to(device)
        feats = clip_model.encode_image(chunk).float()
        if l2_normalize:
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)  # (N*13, 512)

    n = images.shape[0]
    all_feats = all_feats.view(n, 13, -1)    # (N, 13, 512)

    return all_feats


def cache_band_embeddings_to_hdf5(
    loader,
    clip_model,
    output_path: str | Path,
    device: Optional[torch.device] = None,
    micro_batch_size: int = 64,
):
    """
    Encode toàn bộ dataset rồi cache ra HDF5.

    Yêu cầu batch từ loader nên có:
        batch["image"]      -> (N, 13, H, W)
        batch["label"]      -> (N,)
        batch["path"]       -> list[str]  (nếu có)
        batch["label_name"] -> list[str]  (nếu có)
    """
    if device is None:
        device = get_device()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = len(loader.dataset)
    embed_dim = 512

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as f:
        ds_embeddings = f.create_dataset(
            "embeddings",
            shape=(num_samples, 13, embed_dim),
            dtype="float32",
            chunks=(min(128, num_samples), 13, embed_dim),
            compression="gzip",
        )
        ds_labels = f.create_dataset(
            "labels",
            shape=(num_samples,),
            dtype="int64",
        )
        ds_paths = f.create_dataset(
            "paths",
            shape=(num_samples,),
            dtype=str_dtype,
        )
        ds_label_names = f.create_dataset(
            "label_names",
            shape=(num_samples,),
            dtype=str_dtype,
        )

        offset = 0

        for batch in tqdm(loader, desc="Caching per-band CLIP embeddings"):
            images = batch["image"]                  # (N, 13, H, W)
            labels = batch["label"]                  # (N,)
            paths = batch.get("path", [""] * len(labels))
            label_names = batch.get("label_name", [""] * len(labels))

            feats = encode_multispectral_batch(
                images=images,
                clip_model=clip_model,
                device=device,
                micro_batch_size=micro_batch_size,
            )  # (N, 13, 512)

            bs = images.shape[0]
            ds_embeddings[offset:offset + bs] = feats.numpy()
            ds_labels[offset:offset + bs] = labels.cpu().numpy()
            ds_paths[offset:offset + bs] = np.array(paths, dtype=object)
            ds_label_names[offset:offset + bs] = np.array(label_names, dtype=object)

            offset += bs