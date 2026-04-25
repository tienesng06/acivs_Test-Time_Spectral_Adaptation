from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.baselines.pca_baseline import load_openai_clip_model, save_csv_rows
from src.datasets.bigearth_loader import (
    bigearth_collate_fn,
    build_bigearth_subsets,
    get_band_index as get_bigearth_band_index,
)
from src.datasets.eurosat import (
    build_eurosat_dataloaders,
    get_band_index as get_eurosat_band_index,
)
from src.models.clip_utils import preprocess_rgb_for_clip
from src.models.per_band_encoder import get_device
from src.utils.metrics import evaluate_multilabel_image_retrieval, evaluate_text_to_image_retrieval
from src.utils.shared import finalize_metadata



def build_spectral_index_composite(
    images: torch.Tensor,
    *,
    nir_idx: int,
    red_idx: int,
    green_idx: int,
    savi_l: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build a 3-channel spectral-index composite: [NDVI, NDWI, SAVI].

    All three indices are clipped to [-1, 1] then linearly mapped to [0, 1]
    so the output can be passed directly to an RGB CLIP encoder.
    """
    squeeze_batch = images.ndim == 3
    if squeeze_batch:
        images = images.unsqueeze(0)

    if images.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] or [C,H,W], got {tuple(images.shape)}")

    x = images.detach().to(torch.float32).cpu()
    nir = x[:, nir_idx]
    red = x[:, red_idx]
    green = x[:, green_idx]

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    savi = ((1.0 + savi_l) * (nir - red)) / (nir + red + savi_l + eps)

    composite = torch.stack((ndvi, ndwi, savi), dim=1)
    composite = composite.clamp(-1.0, 1.0)
    composite = (composite + 1.0) * 0.5

    return composite.squeeze(0) if squeeze_batch else composite


@torch.no_grad()
def encode_loader_with_spectral_indices(
    loader,
    clip_model,
    device: torch.device,
    *,
    nir_idx: int,
    red_idx: int,
    green_idx: int,
    image_size: int = 224,
    metadata_keys: Sequence[str],
    desc: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Encode a loader after converting multispectral inputs to NDVI/NDWI/SAVI."""
    clip_model.eval()

    feature_chunks: List[torch.Tensor] = []
    metadata_store: Dict[str, List[Any]] = {key: [] for key in metadata_keys}

    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader

    for batch in iterator:
        composite = build_spectral_index_composite(
            batch["image"],
            nir_idx=nir_idx,
            red_idx=red_idx,
            green_idx=green_idx,
        )
        clip_inputs = preprocess_rgb_for_clip(composite, image_size=image_size).to(device)

        features = clip_model.encode_image(clip_inputs).float()
        features = F.normalize(features, dim=-1)
        feature_chunks.append(features.cpu())

        for key in metadata_keys:
            value = batch[key]
            if torch.is_tensor(value):
                metadata_store[key].append(value.cpu())
            else:
                metadata_store[key].append(value)

    result = {"features": torch.cat(feature_chunks, dim=0)}
    result.update(finalize_metadata(metadata_store))
    return result


def run_eurosat_ndvi_baseline(
    root: Path,
    clip_model,
    device: torch.device,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 224,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the NDVI/NDWI/SAVI CLIP baseline on EuroSAT-MS."""
    bundle = build_eurosat_dataloaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        normalize=True,
        reflectance_scale=10_000.0,
        clamp_range=(0.0, 1.0),
        transform=None,
    )
    dataset = bundle["dataset"]
    query_loader = bundle["loaders"]["val"]
    gallery_loader = bundle["loaders"]["test"]

    nir_idx = get_eurosat_band_index("B08")
    red_idx = get_eurosat_band_index("B04")
    green_idx = get_eurosat_band_index("B03")

    query = encode_loader_with_spectral_indices(
        query_loader,
        clip_model,
        device,
        nir_idx=nir_idx,
        red_idx=red_idx,
        green_idx=green_idx,
        image_size=image_size,
        metadata_keys=("label", "label_name", "path"),
        desc="Encoding EuroSAT NDVI composite query",
        show_progress=show_progress,
    )
    gallery = encode_loader_with_spectral_indices(
        gallery_loader,
        clip_model,
        device,
        nir_idx=nir_idx,
        red_idx=red_idx,
        green_idx=green_idx,
        image_size=image_size,
        metadata_keys=("label", "label_name", "path"),
        desc="Encoding EuroSAT NDVI composite gallery",
        show_progress=show_progress,
    )

    metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results = (
        evaluate_text_to_image_retrieval(
            query_features=query["features"],
            query_labels=query["label"],
            gallery_features=gallery["features"],
            gallery_labels=gallery["label"],
            ks=(1, 5, 10),
        )
    )

    summary = {
        "dataset": "EuroSAT_MS",
        "task": "image_to_image_single_label",
        "composite": "NDVI|NDWI|SAVI",
        "num_queries": int(query["label"].numel()),
        "num_gallery": int(gallery["label"].numel()),
        "num_classes": int(len(dataset.class_names)),
        "seed": seed,
        **{key: float(value) for key, value in metrics.items()},
    }

    for row, path, label_name in zip(per_query_results, query["path"], query["label_name"]):
        row["path"] = path
        row["label_name"] = label_name

    return {
        "summary": summary,
        "metrics": metrics,
        "query": query,
        "gallery": gallery,
        "similarity_matrix": similarity_matrix,
        "ranked_indices": ranked_indices,
        "ranked_relevance": ranked_relevance,
        "per_query_results": per_query_results,
    }


def run_bigearth_ndvi_baseline(
    root: Path,
    clip_model,
    device: torch.device,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int = 100_000,
    query_size: int = 1_000,
    gallery_size: int = 10_000,
    seed: int = 42,
    image_size: int = 224,
    remove_snow_cloud_shadow: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the NDVI/NDWI/SAVI CLIP baseline on BigEarthNet-S2."""
    subsets = build_bigearth_subsets(
        root=root,
        max_samples=max_samples,
        query_size=query_size,
        gallery_size=gallery_size,
        use_cache=False,
        normalize=True,
        reflectance_scale=10_000.0,
        clamp_range=(0.0, 1.0),
        seed=seed,
        remove_snow_cloud_shadow=remove_snow_cloud_shadow,
    )

    query_loader = DataLoader(
        subsets["query"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        collate_fn=bigearth_collate_fn,
    )
    gallery_loader = DataLoader(
        subsets["gallery"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        collate_fn=bigearth_collate_fn,
    )

    nir_idx = get_bigearth_band_index("B08")
    red_idx = get_bigearth_band_index("B04")
    green_idx = get_bigearth_band_index("B03")

    query = encode_loader_with_spectral_indices(
        query_loader,
        clip_model,
        device,
        nir_idx=nir_idx,
        red_idx=red_idx,
        green_idx=green_idx,
        image_size=image_size,
        metadata_keys=("labels", "label_names", "patch_name", "index"),
        desc="Encoding BigEarth NDVI composite query",
        show_progress=show_progress,
    )
    gallery = encode_loader_with_spectral_indices(
        gallery_loader,
        clip_model,
        device,
        nir_idx=nir_idx,
        red_idx=red_idx,
        green_idx=green_idx,
        image_size=image_size,
        metadata_keys=("labels", "label_names", "patch_name", "index"),
        desc="Encoding BigEarth NDVI composite gallery",
        show_progress=show_progress,
    )

    metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results = (
        evaluate_multilabel_image_retrieval(
            query_features=query["features"],
            query_labels=query["labels"],
            gallery_features=gallery["features"],
            gallery_labels=gallery["labels"],
            ks=(1, 5, 10),
        )
    )

    label_density_q = query["labels"].sum(dim=1).float().mean().item()
    label_density_g = gallery["labels"].sum(dim=1).float().mean().item()

    summary = {
        "dataset": "BigEarthNetS2",
        "task": "image_to_image_multi_label",
        "composite": "NDVI|NDWI|SAVI",
        "num_queries": int(query["labels"].shape[0]),
        "num_gallery": int(gallery["labels"].shape[0]),
        "num_classes": int(query["labels"].shape[1]),
        "seed": seed,
        "avg_labels_per_query": float(label_density_q),
        "avg_labels_per_gallery": float(label_density_g),
        **{key: float(value) for key, value in metrics.items()},
    }

    for row, patch_name, label_names in zip(
        per_query_results,
        query["patch_name"],
        query["label_names"],
    ):
        row["patch_name"] = patch_name
        row["label_names"] = "|".join(label_names)

    return {
        "summary": summary,
        "metrics": metrics,
        "query": query,
        "gallery": gallery,
        "similarity_matrix": similarity_matrix,
        "ranked_indices": ranked_indices,
        "ranked_relevance": ranked_relevance,
        "per_query_results": per_query_results,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NDVI/NDWI/SAVI CLIP baseline")
    parser.add_argument(
        "--datasets",
        choices=("eurosat", "bigearth", "both"),
        default="both",
        help="Which datasets to evaluate.",
    )
    parser.add_argument("--eurosat-root", type=Path, default=Path("data/EuroSAT_MS"))
    parser.add_argument("--bigearth-root", type=Path, default=Path("data/BigEarthNetS2"))
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("checkpoints/ViT-B-16.pt"))
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/ndvi_baseline_results.csv"),
    )
    parser.add_argument(
        "--eurosat-per-query-csv",
        type=Path,
        default=Path("results/ndvi_baseline_eurosat_per_query.csv"),
    )
    parser.add_argument(
        "--bigearth-per-query-csv",
        type=Path,
        default=Path("results/ndvi_baseline_bigearth_per_query.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eurosat-batch-size", type=int, default=64)
    parser.add_argument("--bigearth-batch-size", type=int, default=32)
    parser.add_argument("--bigearth-max-samples", type=int, default=100_000)
    parser.add_argument("--bigearth-query-size", type=int, default=1_000)
    parser.add_argument("--bigearth-gallery-size", type=int, default=10_000)
    parser.add_argument(
        "--keep-snow-cloud-shadow",
        action="store_true",
        help="Keep noisy BigEarthNet patches instead of filtering them out.",
    )
    parser.add_argument(
        "--hide-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)

    device = get_device()
    print(f"Using device: {device}")

    clip_model, _ = load_openai_clip_model(parsed.clip_checkpoint, device)

    summary_rows: List[Dict[str, Any]] = []

    if parsed.datasets in ("eurosat", "both"):
        print("\n[EuroSAT] Running NDVI/NDWI/SAVI baseline...")
        eurosat_result = run_eurosat_ndvi_baseline(
            root=parsed.eurosat_root,
            clip_model=clip_model,
            device=device,
            batch_size=parsed.eurosat_batch_size,
            num_workers=parsed.num_workers,
            seed=parsed.seed,
            show_progress=not parsed.hide_progress,
        )
        summary_rows.append(eurosat_result["summary"])
        save_csv_rows(eurosat_result["per_query_results"], parsed.eurosat_per_query_csv)
        print("EuroSAT metrics:")
        for key, value in eurosat_result["metrics"].items():
            print(f"  {key}: {value:.2f}%")

    if parsed.datasets in ("bigearth", "both"):
        print("\n[BigEarthNet] Running NDVI/NDWI/SAVI baseline...")
        bigearth_result = run_bigearth_ndvi_baseline(
            root=parsed.bigearth_root,
            clip_model=clip_model,
            device=device,
            batch_size=parsed.bigearth_batch_size,
            num_workers=parsed.num_workers,
            max_samples=parsed.bigearth_max_samples,
            query_size=parsed.bigearth_query_size,
            gallery_size=parsed.bigearth_gallery_size,
            seed=parsed.seed,
            remove_snow_cloud_shadow=not parsed.keep_snow_cloud_shadow,
            show_progress=not parsed.hide_progress,
        )
        summary_rows.append(bigearth_result["summary"])
        save_csv_rows(bigearth_result["per_query_results"], parsed.bigearth_per_query_csv)
        print("BigEarthNet metrics:")
        for key, value in bigearth_result["metrics"].items():
            print(f"  {key}: {value:.2f}%")

    save_csv_rows(summary_rows, parsed.results_csv)
    print(f"\nSaved summary results to {parsed.results_csv}")


if __name__ == "__main__":
    main()
