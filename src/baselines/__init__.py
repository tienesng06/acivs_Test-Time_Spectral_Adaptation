from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "batched_pca_to_rgb",
    "build_gallery_patch_affinity_knn",
    "build_spectral_index_composite",
    "refine_similarity_matrix",
    "run_bigearth_ndvi_baseline",
    "run_bigearth_pca_baseline",
    "run_bigearth_rs_transclip_baseline",
    "run_eurosat_ndvi_baseline",
    "run_eurosat_pca_baseline",
    "run_eurosat_rs_transclip_baseline",
]


_EXPORT_MAP = {
    "batched_pca_to_rgb": ("src.baselines.pca_baseline", "batched_pca_to_rgb"),
    "run_bigearth_pca_baseline": ("src.baselines.pca_baseline", "run_bigearth_pca_baseline"),
    "run_eurosat_pca_baseline": ("src.baselines.pca_baseline", "run_eurosat_pca_baseline"),
    "build_spectral_index_composite": (
        "src.baselines.ndvi_baseline",
        "build_spectral_index_composite",
    ),
    "run_bigearth_ndvi_baseline": ("src.baselines.ndvi_baseline", "run_bigearth_ndvi_baseline"),
    "run_eurosat_ndvi_baseline": ("src.baselines.ndvi_baseline", "run_eurosat_ndvi_baseline"),
    "build_gallery_patch_affinity_knn": (
        "src.baselines.rs_transclip_baseline",
        "build_gallery_patch_affinity_knn",
    ),
    "refine_similarity_matrix": (
        "src.baselines.rs_transclip_baseline",
        "refine_similarity_matrix",
    ),
    "run_bigearth_rs_transclip_baseline": (
        "src.baselines.rs_transclip_baseline",
        "run_bigearth_rs_transclip_baseline",
    ),
    "run_eurosat_rs_transclip_baseline": (
        "src.baselines.rs_transclip_baseline",
        "run_eurosat_rs_transclip_baseline",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
