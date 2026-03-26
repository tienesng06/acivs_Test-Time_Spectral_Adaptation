"""
BigEarthNet-S2 DataLoader for multispectral image retrieval.

Supports:
    - 12 Sentinel-2 bands (B10/Cirrus excluded)
    - Multi-label classification (top-10 most frequent CLC classes)
    - Multi-resolution upsampling (60m->120px, 20m->60px bilinear -> 120px)
    - HDF5 caching for fast I/O
    - Query (1k) / Gallery (10k) / Train (~89k) split from 100k subset

Expected BigEarthNet-S2 folder layout
--------------------------------------
    root/
        <tile_dir>/
            <patch_dir>/
                <patch_dir>_B01.tif
                <patch_dir>_B02.tif
                ...
                <patch_dir>_B12.tif
                <patch_dir>_labels_metadata.json
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Optional heavy dependencies
try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

try:
    import rasterio
except ImportError:
    rasterio = None  # type: ignore[assignment]

try:
    from scipy.ndimage import zoom as scipy_zoom
except ImportError:
    scipy_zoom = None  # type: ignore[assignment]

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)


# ============================================================
# 1) Dataset metadata
# ============================================================

BIGEARTH_BANDS: List[str] = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12",
]

NUM_BANDS: int = len(BIGEARTH_BANDS)  # 12

BAND_RESOLUTION: Dict[str, int] = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
    "B01": 60, "B09": 60,
}

TARGET_SIZE: int = 120

BIGEARTH_19_CLASSES: List[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

TOP10_CLASSES: List[str] = [
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Arable land",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Transitional woodland/shrub",
    "Urban fabric",
]

CLASS_TEXT_MAP: Dict[str, str] = {
    "Broad-leaved forest":       "broad-leaved forest",
    "Coniferous forest":         "coniferous forest",
    "Mixed forest":              "mixed forest",
    "Arable land":               "arable land for agriculture",
    "Pastures":                  "pasture land",
    "Complex cultivation patterns": "complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "agricultural land with natural vegetation",
    "Natural grassland and sparsely vegetated areas":
        "natural grassland and sparse vegetation",
    "Transitional woodland/shrub": "transitional woodland and shrubs",
    "Urban fabric":              "urban residential area",
}


# ============================================================
# 2) CLC-43 to 19-class mapping
# ============================================================

CLC43_TO_19: Dict[str, str] = {
    "Continuous urban fabric":                              "Urban fabric",
    "Discontinuous urban fabric":                           "Urban fabric",
    "Industrial or commercial units":                       "Industrial or commercial units",
    "Non-irrigated arable land":                            "Arable land",
    "Permanently irrigated land":                           "Arable land",
    "Rice fields":                                          "Arable land",
    "Vineyards":                                            "Permanent crops",
    "Fruit trees and berry plantations":                    "Permanent crops",
    "Olive groves":                                         "Permanent crops",
    "Pastures":                                             "Pastures",
    "Annual crops associated with permanent crops":         "Complex cultivation patterns",
    "Complex cultivation patterns":                         "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas":                                  "Agro-forestry areas",
    "Broad-leaved forest":                                  "Broad-leaved forest",
    "Coniferous forest":                                    "Coniferous forest",
    "Mixed forest":                                         "Mixed forest",
    "Natural grasslands":                                   "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas":                             "Natural grassland and sparsely vegetated areas",
    "Moors and heathland":                                  "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation":                            "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub":                          "Transitional woodland/shrub",
    "Beaches, dunes, sands":                                "Beaches, dunes, sands",
    "Inland marshes":                                       "Inland wetlands",
    "Peat bogs":                                            "Inland wetlands",
    "Salt marshes":                                         "Coastal wetlands",
    "Salines":                                              "Coastal wetlands",
    "Water courses":                                        "Inland waters",
    "Water bodies":                                         "Inland waters",
    "Coastal lagoons":                                      "Marine waters",
    "Estuaries":                                            "Marine waters",
    "Sea and ocean":                                        "Marine waters",
    "Mineral extraction sites":                             "Industrial or commercial units",
    "Dump sites":                                           "Industrial or commercial units",
    "Construction sites":                                   "Industrial or commercial units",
    "Green urban areas":                                    "Urban fabric",
    "Sport and leisure facilities":                         "Urban fabric",
    "Airports":                                             "Industrial or commercial units",
    "Port areas":                                           "Industrial or commercial units",
    "Road and rail networks and associated land":           "Industrial or commercial units",
    "Burnt areas":                                          "Natural grassland and sparsely vegetated areas",
    "Glaciers and perpetual snow":                          "Beaches, dunes, sands",
    "Bare rocks":                                           "Natural grassland and sparsely vegetated areas",
    "Intertidal flats":                                     "Coastal wetlands",
}


# ============================================================
# 3) Main dataset class
# ============================================================

class BigEarthNetDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet-S2 (Sentinel-2 multispectral patches).

    Each sample returns a dict::

        {
            "image":       Tensor[12, 120, 120],
            "labels":      Tensor[10],               # multi-hot over TOP10_CLASSES
            "label_names": List[str],
            "text":        str,                       # CLIP text prompt
            "patch_name":  str,
            "index":       int,
        }
    """

    def __init__(
        self,
        root: PathLike,
        split: str = "all",
        top_k_classes: Optional[List[str]] = None,
        max_samples: int = 100_000,
        query_size: int = 1_000,
        gallery_size: int = 10_000,
        use_cache: bool = False,
        cache_dir: Optional[PathLike] = None,
        normalize: bool = True,
        reflectance_scale: float = 10_000.0,
        clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        text_template: str = "A satellite image of {class_text}.",
        seed: int = 42,
        remove_snow_cloud_shadow: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.top_k_classes = list(top_k_classes or TOP10_CLASSES)
        self.max_samples = max_samples
        self.query_size = query_size
        self.gallery_size = gallery_size
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.root.parent
        self.normalize = normalize
        self.reflectance_scale = reflectance_scale
        self.clamp_range = clamp_range
        self.transform = transform
        self.text_template = text_template
        self.seed = seed
        self.remove_snow_cloud_shadow = remove_snow_cloud_shadow

        if self.split not in ("all", "train", "query", "gallery"):
            raise ValueError(
                f"split must be 'all', 'train', 'query', or 'gallery', got '{self.split}'"
            )

        self.band_names: List[str] = list(BIGEARTH_BANDS)
        self.band_to_idx: Dict[str, int] = {b: i for i, b in enumerate(self.band_names)}
        self.num_bands: int = NUM_BANDS

        self.class_to_idx: Dict[str, int] = {
            cls: i for i, cls in enumerate(self.top_k_classes)
        }
        self.idx_to_class: Dict[int, str] = {
            i: cls for cls, i in self.class_to_idx.items()
        }
        self.num_classes: int = len(self.top_k_classes)

        logger.info("Scanning BigEarthNet-S2 patches from %s ...", self.root)
        all_patches = self._scan_patches()
        logger.info("Found %d valid patches (before subset/split).", len(all_patches))

        self.samples: List[Dict[str, Any]] = self._build_subset(all_patches)
        logger.info("Split '%s' contains %d samples.", self.split, len(self.samples))

        if len(self.samples) == 0:
            warnings.warn(
                f"BigEarthNetDataset: 0 samples for split='{self.split}'. "
                f"Check root: {self.root}"
            )

    # ----------------------------------------------------------
    # Scanning
    # ----------------------------------------------------------

    def _scan_patches(self) -> List[Dict[str, Any]]:
        """Walk root and collect patches.

        Supports two formats:
        - **v1**: each patch dir contains ``*_labels_metadata.json``
        - **v2**: patch dirs contain only TIF band files (no JSON).
          In v2 mode labels are set to empty lists.
        """
        patches: List[Dict[str, Any]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"BigEarthNet root not found: {self.root}")

        # --- Try v1 format (JSON metadata per patch) ---
        json_files = sorted(self.root.rglob("*_labels_metadata.json"))

        if json_files:
            logger.info("  Detected BigEarthNet v1 format (%d JSON files).", len(json_files))
            return self._scan_patches_v1(json_files)

        # --- Fall back to v2 format (TIF-only, no JSON) ---
        logger.info("  No *_labels_metadata.json found. Trying v2/TIF-only scan...")
        b02_files = sorted(self.root.rglob("*_B02.tif"))

        if not b02_files:
            raise FileNotFoundError(
                f"No *_labels_metadata.json NOR *_B02.tif found under {self.root}. "
                "Check BigEarthNet-S2 extraction."
            )

        logger.info("  Detected BigEarthNet v2 format (%d B02.tif files).", len(b02_files))
        return self._scan_patches_v2(b02_files)

    # ---- v1 scanning (original, with JSON metadata) ----

    def _scan_patches_v1(
        self, json_files: List[Path]
    ) -> List[Dict[str, Any]]:
        patches: List[Dict[str, Any]] = []

        for json_path in json_files:
            patch_dir = json_path.parent
            patch_name = patch_dir.name

            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping %s: %s", json_path, e)
                continue

            # Skip snow / cloud / shadow
            if self.remove_snow_cloud_shadow:
                snow = meta.get("snow_ice_percentage", meta.get("snow", 0))
                cloud = meta.get("cloud_and_cloud_shadow_percentage",
                                 meta.get("cloud_cover", 0))
                try:
                    if float(snow) > 0 or float(cloud) > 0:
                        continue
                except (TypeError, ValueError):
                    pass

            # Parse labels
            raw_labels: List[str] = meta.get("labels", [])
            if not raw_labels:
                continue

            # Map CLC-43 to 19-class
            labels_19: List[str] = []
            for lbl in raw_labels:
                mapped = CLC43_TO_19.get(lbl, lbl)
                if mapped not in labels_19:
                    labels_19.append(mapped)

            # Keep only patches with at least one top-k label
            labels_top10 = [la for la in labels_19 if la in self.class_to_idx]
            if not labels_top10:
                continue

            # Quick check for B02 existence
            b02_path = patch_dir / f"{patch_name}_B02.tif"
            if not b02_path.exists():
                logger.warning("Missing B02.tif in %s, skipping.", patch_dir)
                continue

            patches.append({
                "patch_dir": patch_dir,
                "patch_name": patch_name,
                "labels_19": labels_19,
                "labels_top10": labels_top10,
            })

        return patches

    # ---- v2 scanning (TIF-only, no per-patch metadata) ----

    def _scan_patches_v2(
        self, b02_files: List[Path]
    ) -> List[Dict[str, Any]]:
        """Scan by locating *_B02.tif files.  Labels are empty."""
        patches: List[Dict[str, Any]] = []

        for b02_path in b02_files:
            patch_dir = b02_path.parent
            patch_name = patch_dir.name

            # Verify at least a few essential bands exist
            ok = True
            for band in ("B01", "B03", "B04"):
                if not (patch_dir / f"{patch_name}_{band}.tif").exists():
                    logger.warning("Missing %s in %s, skipping.", band, patch_dir)
                    ok = False
                    break
            if not ok:
                continue

            patches.append({
                "patch_dir": patch_dir,
                "patch_name": patch_name,
                "labels_19": [],
                "labels_top10": [],
            })

        return patches

    # ----------------------------------------------------------
    # Subset and split
    # ----------------------------------------------------------

    def _build_subset(
        self, all_patches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Subsample to max_samples, then split into train/query/gallery."""
        rng = np.random.default_rng(self.seed)

        if len(all_patches) > self.max_samples:
            indices = rng.choice(len(all_patches), self.max_samples, replace=False)
            indices.sort()
            subset = [all_patches[i] for i in indices]
        else:
            subset = list(all_patches)

        n = len(subset)
        perm = rng.permutation(n)

        q_size = min(self.query_size, n // 5)
        g_size = min(self.gallery_size, n // 2)

        query_idx = perm[:q_size]
        gallery_idx = perm[q_size: q_size + g_size]
        train_idx = perm[q_size + g_size:]

        split_map = {
            "query": query_idx,
            "gallery": gallery_idx,
            "train": train_idx,
            "all": np.arange(n),
        }

        chosen = split_map[self.split]
        return [subset[i] for i in sorted(chosen)]

    # ----------------------------------------------------------
    # Loading
    # ----------------------------------------------------------

    def _read_patch(self, patch_dir: Path, patch_name: str) -> Tensor:
        """Read 12 band TIF files and stack to (12, 120, 120)."""
        if rasterio is None:
            raise ImportError(
                "rasterio is required to read TIF files: pip install rasterio"
            )
        if scipy_zoom is None:
            raise ImportError(
                "scipy is required for band resampling: pip install scipy"
            )

        bands: List[np.ndarray] = []

        for band_name in BIGEARTH_BANDS:
            tif_path = patch_dir / f"{patch_name}_{band_name}.tif"

            if not tif_path.exists():
                raise FileNotFoundError(f"Band file not found: {tif_path}")

            with rasterio.open(tif_path) as src:
                band_data = src.read(1).astype(np.float32)

            h, w = band_data.shape
            if h != TARGET_SIZE or w != TARGET_SIZE:
                zoom_h = TARGET_SIZE / h
                zoom_w = TARGET_SIZE / w
                band_data = scipy_zoom(band_data, (zoom_h, zoom_w), order=1)

            bands.append(band_data)

        image = np.stack(bands, axis=0)  # (12, 120, 120)

        if self.normalize:
            image = image / self.reflectance_scale

        if self.clamp_range is not None:
            lo, hi = self.clamp_range
            image = np.clip(image, lo, hi)

        return torch.from_numpy(image)

    def _labels_to_multihot(self, labels_top10: List[str]) -> Tensor:
        """Convert class names to multi-hot tensor of length num_classes."""
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in labels_top10:
            if lbl in self.class_to_idx:
                vec[self.class_to_idx[lbl]] = 1.0
        return vec

    def _build_text(self, labels_top10: List[str]) -> str:
        """Build CLIP text prompt from patch labels."""
        parts = [CLASS_TEXT_MAP.get(la, la) for la in labels_top10]
        class_text = " and ".join(parts)
        return self.text_template.format(class_text=class_text)

    # ----------------------------------------------------------
    # HDF5 cache
    # ----------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.cache_dir / f"bigearth_cache_{self.split}.h5"

    def _write_cache(self) -> None:
        """Pack all samples into HDF5 for fast subsequent loads."""
        if h5py is None:
            raise ImportError("h5py required for caching: pip install h5py")

        cache_file = self._cache_path()
        if cache_file.exists():
            return

        logger.info("Creating HDF5 cache at %s ...", cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.samples)
        with h5py.File(cache_file, "w") as hf:
            img_ds = hf.create_dataset(
                "images",
                shape=(n, NUM_BANDS, TARGET_SIZE, TARGET_SIZE),
                dtype=np.float32,
                chunks=(1, NUM_BANDS, TARGET_SIZE, TARGET_SIZE),
                compression="gzip",
                compression_opts=4,
            )
            lbl_ds = hf.create_dataset(
                "labels", shape=(n, self.num_classes), dtype=np.float32,
            )

            for i, meta in enumerate(self.samples):
                img = self._read_patch(meta["patch_dir"], meta["patch_name"])
                lbl = self._labels_to_multihot(meta["labels_top10"])
                img_ds[i] = img.numpy()
                lbl_ds[i] = lbl.numpy()

                if (i + 1) % 5000 == 0:
                    logger.info("  cached %d / %d", i + 1, n)

        logger.info("HDF5 cache created: %s", cache_file)

    def _read_from_cache(self, index: int) -> Tuple[Tensor, Tensor]:
        """Read a single sample from HDF5 cache."""
        if h5py is None:
            raise ImportError("h5py required for caching: pip install h5py")

        cache_file = self._cache_path()
        with h5py.File(cache_file, "r") as hf:
            image = torch.from_numpy(hf["images"][index].copy())
            labels = torch.from_numpy(hf["labels"][index].copy())
        return image, labels

    # ----------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        meta = self.samples[index]

        if self.use_cache:
            cache_file = self._cache_path()
            if not cache_file.exists():
                self._write_cache()
            image, labels = self._read_from_cache(index)
        else:
            image = self._read_patch(meta["patch_dir"], meta["patch_name"])
            labels = self._labels_to_multihot(meta["labels_top10"])

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "labels": labels,
            "label_names": list(meta["labels_top10"]),
            "text": self._build_text(meta["labels_top10"]),
            "patch_name": meta["patch_name"],
            "index": index,
        }

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def get_class_distribution(self) -> Dict[str, int]:
        """Count patches per top-k class (multi-label aware)."""
        counts: Counter = Counter()
        for s in self.samples:
            for lbl in s["labels_top10"]:
                counts[lbl] += 1
        return {cls: counts.get(cls, 0) for cls in self.top_k_classes}

    def get_text_prompts(self) -> List[str]:
        """Return one CLIP text prompt per top-k class."""
        return [
            self.text_template.format(class_text=CLASS_TEXT_MAP.get(cls, cls))
            for cls in self.top_k_classes
        ]


# ============================================================
# 4) Builder functions
# ============================================================

def build_bigearth_subsets(
    root: PathLike,
    max_samples: int = 100_000,
    query_size: int = 1_000,
    gallery_size: int = 10_000,
    use_cache: bool = False,
    cache_dir: Optional[PathLike] = None,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    normalize: bool = True,
    reflectance_scale: float = 10_000.0,
    clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    seed: int = 42,
    remove_snow_cloud_shadow: bool = True,
) -> Dict[str, BigEarthNetDataset]:
    """Build train / query / gallery Dataset objects."""
    common = dict(
        root=root,
        max_samples=max_samples,
        query_size=query_size,
        gallery_size=gallery_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        transform=transform,
        normalize=normalize,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
        seed=seed,
        remove_snow_cloud_shadow=remove_snow_cloud_shadow,
    )
    return {
        "train": BigEarthNetDataset(split="train", **common),
        "query": BigEarthNetDataset(split="query", **common),
        "gallery": BigEarthNetDataset(split="gallery", **common),
    }


def build_bigearth_dataloaders(
    root: PathLike,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int = 100_000,
    query_size: int = 1_000,
    gallery_size: int = 10_000,
    use_cache: bool = False,
    cache_dir: Optional[PathLike] = None,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    normalize: bool = True,
    reflectance_scale: float = 10_000.0,
    clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    seed: int = 42,
    pin_memory: bool = False,
    remove_snow_cloud_shadow: bool = True,
) -> Dict[str, Any]:
    """Build train / query / gallery DataLoaders."""
    subsets = build_bigearth_subsets(
        root=root,
        max_samples=max_samples,
        query_size=query_size,
        gallery_size=gallery_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        transform=transform,
        normalize=normalize,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
        seed=seed,
        remove_snow_cloud_shadow=remove_snow_cloud_shadow,
    )

    loaders = {}
    for split_name, ds in subsets.items():
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    return {"datasets": subsets, "loaders": loaders}


# ============================================================
# 5) Helpers
# ============================================================

def get_band_index(band_name: str) -> int:
    """Return 0-based index of a band in BIGEARTH_BANDS."""
    if band_name not in BIGEARTH_BANDS:
        raise ValueError(f"Unknown band: {band_name}. Valid: {BIGEARTH_BANDS}")
    return BIGEARTH_BANDS.index(band_name)


def extract_rgb_bands(image_12ch: Tensor) -> Tensor:
    """
    Extract true-color RGB: R=B04, G=B03, B=B02.
    Supports [12, H, W] or [N, 12, H, W].
    """
    r = get_band_index("B04")
    g = get_band_index("B03")
    b = get_band_index("B02")

    if image_12ch.ndim == 3:
        return image_12ch[[r, g, b], :, :]
    if image_12ch.ndim == 4:
        return image_12ch[:, [r, g, b], :, :]
    raise ValueError(
        f"Expected [12,H,W] or [N,12,H,W], got {tuple(image_12ch.shape)}"
    )


def get_bigearth_text_queries(
    template: str = "A satellite image of {class_text}.",
) -> List[str]:
    """Return one CLIP text prompt per top-10 class."""
    return [
        template.format(class_text=CLASS_TEXT_MAP.get(cls, cls))
        for cls in TOP10_CLASSES
    ]
