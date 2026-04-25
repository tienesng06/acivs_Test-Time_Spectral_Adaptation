from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.bigearth_loader import bigearth_collate_fn, build_bigearth_subsets
from src.datasets.eurosat import build_eurosat_dataloaders
from src.models.clip_utils import preprocess_rgb_for_clip
from src.models.per_band_encoder import get_device
from src.utils.metrics import (
    average_precision_from_relevance,
    evaluate_multilabel_image_retrieval,
    evaluate_text_to_image_retrieval,
)
from src.utils.shared import finalize_metadata, load_openai_clip_model, save_csv_rows


def fit_global_pca_from_loader(
    loader,
    *,
    desc: str,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Fit one PCA basis from all pixels in a loader.

    A global basis keeps channel semantics consistent across images, which is
    much more suitable for a frozen RGB CLIP encoder than fitting PCA per image.
    """
    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader

    sum_x: Optional[torch.Tensor] = None
    sum_xxt: Optional[torch.Tensor] = None
    total_pixels = 0

    for batch in iterator:
        images = batch["image"].detach().to(torch.float32).cpu()
        if images.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W] while fitting PCA, got {tuple(images.shape)}")

        flat = images.permute(0, 2, 3, 1).reshape(-1, images.shape[1]).to(torch.float64)

        if sum_x is None:
            channels = flat.shape[1]
            sum_x = torch.zeros(channels, dtype=torch.float64)
            sum_xxt = torch.zeros(channels, channels, dtype=torch.float64)

        sum_x += flat.sum(dim=0)
        sum_xxt += flat.T @ flat
        total_pixels += flat.shape[0]

    if sum_x is None or sum_xxt is None or total_pixels == 0:
        raise ValueError("Cannot fit PCA from an empty loader.")

    mean = sum_x / float(total_pixels)
    covariance = (sum_xxt / float(total_pixels)) - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5

    eigvals, eigvecs = torch.linalg.eigh(covariance)
    components = eigvecs[:, -3:].flip(dims=(-1,)).to(torch.float32)

    proj_min = torch.full((3,), float("inf"), dtype=torch.float32)
    proj_max = torch.full((3,), float("-inf"), dtype=torch.float32)

    iterator = tqdm(loader, desc=f"{desc} (range)") if show_progress else loader
    mean_f32 = mean.to(torch.float32)
    for batch in iterator:
        images = batch["image"].detach().to(torch.float32).cpu()
        flat = images.permute(0, 2, 3, 1).reshape(-1, images.shape[1])
        projected = (flat - mean_f32.view(1, -1)) @ components
        proj_min = torch.minimum(proj_min, projected.min(dim=0).values)
        proj_max = torch.maximum(proj_max, projected.max(dim=0).values)

    return {
        "mean": mean.to(torch.float32),
        "components": components,
        "explained_variance": eigvals[-3:].flip(dims=(0,)).to(torch.float32),
        "proj_min": proj_min,
        "proj_max": proj_max,
    }


def apply_global_pca_to_rgb(
    images: torch.Tensor,
    pca_state: Dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project multispectral images with a fitted global PCA basis."""
    squeeze_batch = images.ndim == 3
    if squeeze_batch:
        images = images.unsqueeze(0)

    if images.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] or [C,H,W], got {tuple(images.shape)}")

    n, c, h, w = images.shape
    mean = pca_state["mean"].to(torch.float32).cpu()
    components = pca_state["components"].to(torch.float32).cpu()
    proj_min = pca_state.get("proj_min")
    proj_max = pca_state.get("proj_max")

    if mean.shape[0] != c or components.shape[0] != c:
        raise ValueError(
            f"PCA state expects {mean.shape[0]} channels, but got image batch with {c}"
        )

    x = images.detach().to(torch.float32).cpu()
    x = x.permute(0, 2, 3, 1).reshape(n, h * w, c)
    projected = (x - mean.view(1, 1, c)) @ components

    if proj_min is not None and proj_max is not None:
        proj_min = proj_min.to(torch.float32).cpu().view(1, 1, 3)
        proj_max = proj_max.to(torch.float32).cpu().view(1, 1, 3)
        projected = (projected - proj_min) / (proj_max - proj_min + eps)
    else:
        comp_min = projected.min(dim=1, keepdim=True).values
        comp_max = projected.max(dim=1, keepdim=True).values
        projected = (projected - comp_min) / (comp_max - comp_min + eps)

    projected = projected.clamp(0.0, 1.0)

    rgb = projected.reshape(n, h, w, 3).permute(0, 3, 1, 2).contiguous()
    return rgb.squeeze(0) if squeeze_batch else rgb


def batched_pca_to_rgb(
    images: torch.Tensor,
    n_components: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Convert multispectral images to per-image PCA pseudo-RGB.

    For each image independently:
        (C, H, W) -> (H*W, C) -> PCA -> (H*W, 3) -> min-max to [0, 1]

    Args:
        images:
            Tensor with shape [C, H, W] or [N, C, H, W].
        n_components:
            Number of PCA components to keep. Baseline uses 3.

    Returns:
        Tensor with shape [3, H, W] or [N, 3, H, W], normalized to [0, 1].
    """
    squeeze_batch = images.ndim == 3
    if squeeze_batch:
        images = images.unsqueeze(0)

    if images.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] or [C,H,W], got {tuple(images.shape)}")

    n, c, h, w = images.shape
    if c < n_components:
        raise ValueError(
            f"PCA needs at least {n_components} channels, but got {c}"
        )

    x = images.detach().to(torch.float32).cpu()
    x = x.permute(0, 2, 3, 1).reshape(n, h * w, c)  # [N, P, C]

    x_centered = x - x.mean(dim=1, keepdim=True)
    denom = max((h * w) - 1, 1)
    cov = x_centered.transpose(1, 2) @ x_centered / float(denom)  # [N, C, C]

    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending eigenvalues
    top_vecs = eigvecs[:, :, -n_components:].flip(dims=(-1,))  # [N, C, 3]

    projected = x_centered @ top_vecs  # [N, P, 3]

    comp_min = projected.min(dim=1, keepdim=True).values
    comp_max = projected.max(dim=1, keepdim=True).values
    projected = (projected - comp_min) / (comp_max - comp_min + eps)
    projected = projected.clamp(0.0, 1.0)

    rgb = projected.reshape(n, h, w, n_components).permute(0, 3, 1, 2).contiguous()
    return rgb.squeeze(0) if squeeze_batch else rgb





@torch.no_grad()
def encode_loader_with_pca(
    loader,
    clip_model,
    device: torch.device,
    *,
    image_size: int = 224,
    pca_state: Optional[Dict[str, torch.Tensor]] = None,
    metadata_keys: Sequence[str],
    desc: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Encode a dataloader by first converting multispectral inputs to PCA pseudo-RGB."""
    clip_model.eval()

    feature_chunks: List[torch.Tensor] = []
    metadata_store: Dict[str, List[Any]] = {key: [] for key in metadata_keys}

    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader

    for batch in iterator:
        images = batch["image"]
        if pca_state is None:
            pca_rgb = batched_pca_to_rgb(images)
        else:
            pca_rgb = apply_global_pca_to_rgb(images, pca_state=pca_state)
        clip_inputs = preprocess_rgb_for_clip(pca_rgb, image_size=image_size).to(device)

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


# evaluate_multilabel_image_retrieval is imported from src.utils.metrics
# (canonical location per design doc Week 8 codebase structure)


def run_eurosat_pca_baseline(
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
    """Run PCA pseudo-RGB CLIP baseline on EuroSAT-MS image-to-image retrieval."""
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
    train_loader = bundle["loaders"]["train"]
    query_loader = bundle["loaders"]["val"]
    gallery_loader = bundle["loaders"]["test"]

    pca_state = fit_global_pca_from_loader(
        train_loader,
        desc="Fitting EuroSAT global PCA",
        show_progress=show_progress,
    )

    query = encode_loader_with_pca(
        query_loader,
        clip_model,
        device,
        image_size=image_size,
        pca_state=pca_state,
        metadata_keys=("label", "label_name", "path"),
        desc="Encoding EuroSAT PCA query",
        show_progress=show_progress,
    )
    gallery = encode_loader_with_pca(
        gallery_loader,
        clip_model,
        device,
        image_size=image_size,
        pca_state=pca_state,
        metadata_keys=("label", "label_name", "path"),
        desc="Encoding EuroSAT PCA gallery",
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
        "num_queries": int(query["label"].numel()),
        "num_gallery": int(gallery["label"].numel()),
        "num_classes": int(len(dataset.class_names)),
        "seed": seed,
        "pca_fit": "global_reference_pixels",
        **{k: float(v) for k, v in metrics.items()},
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


def run_bigearth_pca_baseline(
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
    """Run PCA pseudo-RGB CLIP baseline on BigEarthNet-S2 query/gallery splits."""
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
    train_loader = DataLoader(
        subsets["train"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        collate_fn=bigearth_collate_fn,
    )

    pca_state = fit_global_pca_from_loader(
        train_loader,
        desc="Fitting BigEarth global PCA",
        show_progress=show_progress,
    )

    query = encode_loader_with_pca(
        query_loader,
        clip_model,
        device,
        image_size=image_size,
        pca_state=pca_state,
        metadata_keys=("labels", "label_names", "patch_name", "index"),
        desc="Encoding BigEarth query PCA features",
        show_progress=show_progress,
    )
    gallery = encode_loader_with_pca(
        gallery_loader,
        clip_model,
        device,
        image_size=image_size,
        pca_state=pca_state,
        metadata_keys=("labels", "label_names", "patch_name", "index"),
        desc="Encoding BigEarth gallery PCA features",
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
        "num_queries": int(query["labels"].shape[0]),
        "num_gallery": int(gallery["labels"].shape[0]),
        "num_classes": int(query["labels"].shape[1]),
        "seed": seed,
        "pca_fit": "global_reference_pixels",
        "avg_labels_per_query": float(label_density_q),
        "avg_labels_per_gallery": float(label_density_g),
        **{k: float(v) for k, v in metrics.items()},
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


def save_csv_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """Save a list of dicts as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to save for {output_path}")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCA pseudo-RGB CLIP baseline")
    parser.add_argument(
        "--datasets",
        choices=("eurosat", "bigearth", "both"),
        default="both",
        help="Which datasets to evaluate.",
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        default=Path("data/EuroSAT_MS"),
    )
    parser.add_argument(
        "--bigearth-root",
        type=Path,
        default=Path("data/BigEarthNetS2"),
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=Path("checkpoints/ViT-B-16.pt"),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/pca_baseline_results.csv"),
    )
    parser.add_argument(
        "--eurosat-per-query-csv",
        type=Path,
        default=Path("results/pca_baseline_eurosat_per_query.csv"),
    )
    parser.add_argument(
        "--bigearth-per-query-csv",
        type=Path,
        default=Path("results/pca_baseline_bigearth_per_query.csv"),
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
        print("\n[EuroSAT] Running PCA pseudo-RGB baseline...")
        eurosat_result = run_eurosat_pca_baseline(
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
        print("\n[BigEarthNet] Running PCA pseudo-RGB baseline...")
        bigearth_result = run_bigearth_pca_baseline(
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
