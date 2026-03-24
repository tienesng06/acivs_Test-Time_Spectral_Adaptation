from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

PathLike = Union[str, Path]


# ============================================================
# 1) Dataset metadata
# ============================================================

SENTINEL2_BANDS: List[str] = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12",
]

EUROSAT_CLASSES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_TEXT_MAP: Dict[str, str] = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway or major road",
    "Industrial": "industrial area",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential area",
    "River": "river",
    "SeaLake": "sea or lake",
}


# ============================================================
# 2) Main dataset
# ============================================================

class EuroSATMSDataset(Dataset):
    """
    Single dataset for the whole EuroSAT-MS directory.

    Expected folder structure:
        root/
            AnnualCrop/
                xxx.tif
                ...
            Forest/
                xxx.tif
                ...
            ...
            SeaLake/
                xxx.tif

    Each sample returns a dict:
        {
            "image": Tensor[13, H, W],
            "label": int,
            "label_name": str,
            "text": str,
            "path": str,
            "index": int,
        }
    """

    def __init__(
        self,
        root: PathLike,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        normalize: bool = True,
        reflectance_scale: float = 10000.0,
        clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        class_names: Optional[Sequence[str]] = None,
        text_template: str = "A satellite image of {class_text}.",
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.normalize = normalize
        self.reflectance_scale = reflectance_scale
        self.clamp_range = clamp_range
        self.class_names = list(class_names) if class_names is not None else list(EUROSAT_CLASSES)
        self.text_template = text_template

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.band_names: List[str] = list(SENTINEL2_BANDS)
        self.band_to_idx: Dict[str, int] = {band: i for i, band in enumerate(self.band_names)}

        self.class_to_idx: Dict[str, int] = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: class_name for class_name, idx in self.class_to_idx.items()
        }

        self.samples: List[Dict[str, object]] = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No .tif files found under: {self.root}")

    def _build_samples(self) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []

        missing_dirs = [c for c in self.class_names if not (self.root / c).exists()]
        if missing_dirs:
            raise FileNotFoundError(
                f"Missing class folders under {self.root}: {missing_dirs}"
            )

        for class_name in self.class_names:
            class_dir = self.root / class_name
            tif_files = sorted(class_dir.glob("*.tif"))

            for tif_path in tif_files:
                samples.append(
                    {
                        "path": tif_path,
                        "label": self.class_to_idx[class_name],
                        "label_name": class_name,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _build_text(self, label_name: str) -> str:
        class_text = CLASS_TEXT_MAP.get(label_name, label_name)
        return self.text_template.format(class_text=class_text)

    def _read_tif(self, tif_path: Path) -> Tensor:
        with rasterio.open(tif_path) as src:
            image = src.read().astype(np.float32)  # [C, H, W]

        if image.ndim != 3:
            raise ValueError(f"Expected 3D array [C,H,W], got shape {image.shape} in {tif_path}")

        if image.shape[0] != 13:
            raise ValueError(
                f"Expected 13 bands, but got {image.shape[0]} bands in file: {tif_path}"
            )

        if self.normalize:
            image = image / self.reflectance_scale

        if self.clamp_range is not None:
            low, high = self.clamp_range
            image = np.clip(image, low, high)

        return torch.from_numpy(image)

    def __getitem__(self, index: int) -> Dict[str, object]:
        meta = self.samples[index]
        tif_path = Path(meta["path"])
        label = int(meta["label"])
        label_name = str(meta["label_name"])

        image = self._read_tif(tif_path)

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,                     # Tensor [13, H, W]
            "label": label,                    # int
            "label_name": label_name,          # str
            "text": self._build_text(label_name),
            "path": str(tif_path),
            "index": index,
        }
        return sample

    def get_labels(self) -> List[int]:
        return [int(sample["label"]) for sample in self.samples]

    def get_paths(self) -> List[str]:
        return [str(sample["path"]) for sample in self.samples]

    def get_class_distribution(self) -> Dict[str, int]:
        counts = {class_name: 0 for class_name in self.class_names}
        for sample in self.samples:
            counts[str(sample["label_name"])] += 1
        return counts

    def get_text_prompts(self) -> List[str]:
        return [self._build_text(class_name) for class_name in self.class_names]


# ============================================================
# 3) Split by indices
# ============================================================

def make_stratified_split_indices(
    labels: Sequence[int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split indices in a class-balanced (stratified) way.

    Returns:
        {
            "train": [...],
            "val": [...],
            "test": [...],
        }
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)

    split_indices = {"train": [], "val": [], "test": []}

    unique_labels = sorted(np.unique(labels_np).tolist())

    for label in unique_labels:
        class_indices = np.where(labels_np == label)[0]
        class_indices = rng.permutation(class_indices)

        n = len(class_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx = class_indices[:n_train]
        val_idx = class_indices[n_train:n_train + n_val]
        test_idx = class_indices[n_train + n_val:]

        assert len(train_idx) + len(val_idx) + len(test_idx) == n
        assert len(test_idx) == n_test

        split_indices["train"].extend(train_idx.tolist())
        split_indices["val"].extend(val_idx.tolist())
        split_indices["test"].extend(test_idx.tolist())

    # Shuffle each split globally so batches are not grouped by class order
    for split_name in split_indices:
        split_indices[split_name] = rng.permutation(split_indices[split_name]).tolist()

    return split_indices


def build_eurosat_subsets(
    root: PathLike,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    normalize: bool = True,
    reflectance_scale: float = 10000.0,
    clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Build:
        - full dataset
        - split indices
        - train/val/test subsets
    """
    dataset = EuroSATMSDataset(
        root=root,
        transform=transform,
        normalize=normalize,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
    )

    split_indices = make_stratified_split_indices(
        labels=dataset.get_labels(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    subsets = {
        "train": Subset(dataset, split_indices["train"]),
        "val": Subset(dataset, split_indices["val"]),
        "test": Subset(dataset, split_indices["test"]),
    }

    return {
        "dataset": dataset,
        "indices": split_indices,
        "subsets": subsets,
    }


def build_eurosat_dataloaders(
    root: PathLike,
    batch_size: int = 16,
    num_workers: int = 0,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    normalize: bool = True,
    reflectance_scale: float = 10000.0,
    clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    pin_memory: bool = False,
) -> Dict[str, object]:
    """
    Return full dataset + subsets + dataloaders.
    """
    bundle = build_eurosat_subsets(
        root=root,
        transform=transform,
        normalize=normalize,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    subsets = bundle["subsets"]

    train_loader = DataLoader(
        subsets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        subsets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        subsets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    bundle["loaders"] = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return bundle


# ============================================================
# 4) Helpers for baseline step 4 (CLIP-RGB)
# ============================================================

def get_band_index(band_name: str) -> int:
    """
    Example:
        get_band_index("B04") -> 3
    """
    if band_name not in SENTINEL2_BANDS:
        raise ValueError(f"Unknown band name: {band_name}")
    return SENTINEL2_BANDS.index(band_name)


def extract_rgb_bands(image_13ch: Tensor) -> Tensor:
    """
    Extract Sentinel-2 RGB in paper baseline order:
        R = B04, G = B03, B = B02

    Supports:
        image_13ch: [13, H, W] or [N, 13, H, W]

    Returns:
        [3, H, W] or [N, 3, H, W]
    """
    r_idx = get_band_index("B04")
    g_idx = get_band_index("B03")
    b_idx = get_band_index("B02")

    if image_13ch.ndim == 3:
        return image_13ch[[r_idx, g_idx, b_idx], :, :]
    if image_13ch.ndim == 4:
        return image_13ch[:, [r_idx, g_idx, b_idx], :, :]

    raise ValueError(
        f"Expected image tensor with shape [13,H,W] or [N,13,H,W], got {tuple(image_13ch.shape)}"
    )


def replicate_single_band_to_rgb(
    image_13ch: Tensor,
    band: Union[int, str],
) -> Tensor:
    """
    For the later paper method:
    take one band and replicate it to 3 channels so frozen CLIP can ingest it.

    Supports:
        image_13ch: [13, H, W] or [N, 13, H, W]

    Returns:
        [3, H, W] or [N, 3, H, W]
    """
    band_idx = get_band_index(band) if isinstance(band, str) else int(band)

    if band_idx < 0 or band_idx >= 13:
        raise ValueError(f"Band index out of range: {band_idx}")

    if image_13ch.ndim == 3:
        single = image_13ch[band_idx:band_idx + 1, :, :]   # [1,H,W]
        return single.repeat(3, 1, 1)

    if image_13ch.ndim == 4:
        single = image_13ch[:, band_idx:band_idx + 1, :, :]  # [N,1,H,W]
        return single.repeat(1, 3, 1, 1)

    raise ValueError(
        f"Expected image tensor with shape [13,H,W] or [N,13,H,W], got {tuple(image_13ch.shape)}"
    )


def get_default_text_queries() -> List[str]:
    """
    Return one text prompt per EuroSAT class.
    Useful for the first CLIP-RGB baseline.
    """
    dataset_like_template = "A satellite image of {class_text}."
    return [
        dataset_like_template.format(
            class_text=CLASS_TEXT_MAP.get(class_name, class_name)
        )
        for class_name in EUROSAT_CLASSES
    ]


# ============================================================
# 5) Small utility for quick sanity-check
# ============================================================

def describe_split_sizes(indices: Dict[str, List[int]]) -> Dict[str, int]:
    return {split_name: len(idx_list) for split_name, idx_list in indices.items()}