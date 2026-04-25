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
from src.utils.metrics import average_precision_from_relevance
from src.utils.shared import finalize_metadata
from src.utils.visualization import extract_rgb_bands





def _extract_clip_global_and_patch_tokens(
    clip_model,
    clip_inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """
    Encode CLIP ViT inputs and return both global image features and patch tokens.

    This baseline relies on patch-level descriptors for cross-patch refinement,
    so it currently supports OpenAI CLIP ViT backbones only.
    """
    visual = getattr(clip_model, "visual", None)
    required_attrs = (
        "conv1",
        "class_embedding",
        "positional_embedding",
        "ln_pre",
        "transformer",
        "ln_post",
    )
    if visual is None or any(not hasattr(visual, attr) for attr in required_attrs):
        raise NotImplementedError(
            "RS-TransCLIP baseline currently requires an OpenAI CLIP ViT backbone."
        )

    # OpenAI CLIP often keeps vision weights in fp16 on MPS/CUDA. The manual
    # token path must therefore match the conv weight dtype explicitly.
    clip_inputs = clip_inputs.to(dtype=visual.conv1.weight.dtype)

    x = visual.conv1(clip_inputs)
    grid_h, grid_w = int(x.shape[-2]), int(x.shape[-1])

    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    class_embedding = visual.class_embedding.to(dtype=x.dtype, device=x.device)
    class_token = class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
    x = torch.cat([class_token, x], dim=1)

    positional_embedding = visual.positional_embedding.to(dtype=x.dtype, device=x.device)
    if positional_embedding.shape[0] != x.shape[1]:
        raise ValueError(
            "Positional embedding length does not match the CLIP input resolution. "
            f"Expected {positional_embedding.shape[0]} tokens, got {x.shape[1]}."
        )

    x = x + positional_embedding
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    x = visual.ln_post(x)

    if getattr(visual, "proj", None) is not None:
        projection = visual.proj.to(dtype=x.dtype, device=x.device)
        global_features = x[:, 0, :] @ projection
        patch_tokens = x[:, 1:, :] @ projection
    else:
        global_features = x[:, 0, :]
        patch_tokens = x[:, 1:, :]

    global_features = F.normalize(global_features.float(), dim=-1)
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    return global_features, patch_tokens, (grid_h, grid_w)


def pool_patch_tokens(
    patch_tokens: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    output_grid_size: int = 4,
) -> torch.Tensor:
    """
    Spatially resize ViT patch tokens to a compact grid.

    MPS requires adaptive pooling input sizes to be divisible by output sizes,
    which does not hold for CLIP's default 14x14 patch grid with output size 4.
    Bilinear resize is more portable here and still preserves coarse spatial
    structure for the downstream patch-affinity graph.
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens with shape [N, P, D], got {tuple(patch_tokens.shape)}")

    grid_h, grid_w = grid_size
    if grid_h * grid_w != patch_tokens.shape[1]:
        raise ValueError(
            f"grid_size={grid_size} is incompatible with {patch_tokens.shape[1]} patch tokens."
        )
    if output_grid_size <= 0:
        raise ValueError(f"output_grid_size must be > 0, got {output_grid_size}")

    pooled_grid_h = min(output_grid_size, grid_h)
    pooled_grid_w = min(output_grid_size, grid_w)

    x = patch_tokens.view(patch_tokens.shape[0], grid_h, grid_w, patch_tokens.shape[-1])
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(
        x,
        size=(pooled_grid_h, pooled_grid_w),
        mode="bilinear",
        align_corners=False,
    )
    x = x.flatten(2).transpose(1, 2).contiguous()
    return F.normalize(x, dim=-1)


def build_patch_descriptor(pooled_patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Compress pooled patch tokens into one patch-aware descriptor per image.

    The descriptor mixes the global average patch pattern with the strongest
    local response so the gallery graph reacts to both scene layout and
    distinctive local evidence.
    """
    if pooled_patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected pooled_patch_tokens with shape [N, P, D], got {tuple(pooled_patch_tokens.shape)}"
        )

    descriptor = pooled_patch_tokens.mean(dim=1) + pooled_patch_tokens.amax(dim=1)
    return F.normalize(descriptor.float(), dim=-1)


@torch.no_grad()
def encode_loader_with_rgb_patches(
    loader,
    clip_model,
    device: torch.device,
    *,
    rgb_indices: Sequence[int],
    image_size: int = 224,
    patch_pool_size: int = 4,
    metadata_keys: Sequence[str],
    desc: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Encode a loader into CLIP global features plus compact patch descriptors."""
    clip_model.eval()

    feature_chunks: List[torch.Tensor] = []
    descriptor_chunks: List[torch.Tensor] = []
    metadata_store: Dict[str, List[Any]] = {key: [] for key in metadata_keys}

    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader

    for batch in iterator:
        rgb = extract_rgb_bands(batch["image"], rgb_indices=rgb_indices)
        clip_inputs = preprocess_rgb_for_clip(rgb, image_size=image_size).to(device)

        global_features, patch_tokens, patch_grid = _extract_clip_global_and_patch_tokens(
            clip_model,
            clip_inputs,
        )
        pooled_patch_tokens = pool_patch_tokens(
            patch_tokens,
            grid_size=patch_grid,
            output_grid_size=patch_pool_size,
        )
        patch_descriptors = build_patch_descriptor(pooled_patch_tokens)

        feature_chunks.append(global_features.cpu())
        descriptor_chunks.append(patch_descriptors.cpu())

        for key in metadata_keys:
            value = batch[key]
            if torch.is_tensor(value):
                metadata_store[key].append(value.cpu())
            else:
                metadata_store[key].append(value)

    result = {
        "features": torch.cat(feature_chunks, dim=0),
        "patch_descriptors": torch.cat(descriptor_chunks, dim=0),
    }
    result.update(finalize_metadata(metadata_store))
    return result


@torch.no_grad()
def build_gallery_patch_affinity_knn(
    patch_descriptors: torch.Tensor,
    *,
    topk: int = 20,
    chunk_size: int = 256,
    compute_device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor | int]:
    """
    Build a row-stochastic KNN gallery affinity graph from patch descriptors.

    The output stores only top-k neighbors per gallery image, which is enough
    to implement `patch_affinity @ logits` without materializing a dense NxN matrix.
    """
    patch_descriptors = F.normalize(patch_descriptors.float(), dim=-1).cpu()
    num_gallery = int(patch_descriptors.shape[0])

    if num_gallery == 0:
        raise ValueError("Cannot build patch affinity from an empty gallery.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    if num_gallery == 1:
        return {
            "indices": torch.zeros((1, 1), dtype=torch.long),
            "weights": torch.ones((1, 1), dtype=torch.float32),
            "topk": 1,
        }

    topk = max(1, min(topk, num_gallery - 1))
    affinity_device = compute_device or torch.device("cpu")
    descriptors_device = patch_descriptors.to(affinity_device)

    knn_indices = torch.empty((num_gallery, topk), dtype=torch.long)
    knn_weights = torch.empty((num_gallery, topk), dtype=torch.float32)

    iterator = range(0, num_gallery, chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Building gallery patch affinity")

    for start in iterator:
        stop = min(start + chunk_size, num_gallery)
        block = descriptors_device[start:stop]

        similarity = block @ descriptors_device.T
        similarity = similarity.clamp(min=0.0)

        local_rows = torch.arange(stop - start, device=affinity_device)
        global_rows = torch.arange(start, stop, device=affinity_device)
        similarity[local_rows, global_rows] = -1.0

        top_values, top_indices = torch.topk(similarity, k=topk, dim=1)
        top_values = top_values.clamp(min=0.0)

        denom = top_values.sum(dim=1, keepdim=True)
        zero_rows = denom.squeeze(1) <= 0
        if zero_rows.any():
            top_values[zero_rows] = 1.0
            denom = top_values.sum(dim=1, keepdim=True)

        top_weights = top_values / (denom + 1e-6)

        knn_indices[start:stop] = top_indices.cpu()
        knn_weights[start:stop] = top_weights.cpu()

    return {
        "indices": knn_indices,
        "weights": knn_weights,
        "topk": topk,
    }


def propagate_logits_with_patch_affinity(
    logits: torch.Tensor,
    patch_affinity: Dict[str, torch.Tensor | int],
) -> torch.Tensor:
    """Apply sparse gallery graph propagation: `patch_affinity @ logits`."""
    squeeze_query = logits.ndim == 1
    if squeeze_query:
        logits = logits.unsqueeze(0)

    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [Q, N] or [N], got {tuple(logits.shape)}")

    indices = patch_affinity["indices"]
    weights = patch_affinity["weights"]
    if not isinstance(indices, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("patch_affinity must contain tensor 'indices' and 'weights'.")

    if logits.shape[1] != indices.shape[0]:
        raise ValueError(
            f"logits width ({logits.shape[1]}) must match patch affinity size ({indices.shape[0]})."
        )

    neighbor_logits = logits[:, indices]
    propagated = (neighbor_logits * weights.unsqueeze(0)).sum(dim=-1)
    return propagated.squeeze(0) if squeeze_query else propagated


def refine_similarity_matrix(
    logits: torch.Tensor,
    *,
    patch_affinity: Dict[str, torch.Tensor | int],
    alpha: float,
) -> torch.Tensor:
    """Refine gallery logits with cross-patch propagation."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    propagated = propagate_logits_with_patch_affinity(logits, patch_affinity)
    return (1.0 - alpha) * logits + alpha * propagated


def evaluate_single_label_retrieval_from_similarity(
    similarity_matrix: torch.Tensor,
    *,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Evaluate retrieval metrics from a precomputed single-label similarity matrix."""
    similarity_matrix = similarity_matrix.cpu().float()
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()

    ranked_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    expanded_gallery_labels = gallery_labels.unsqueeze(0).expand(query_labels.shape[0], -1)
    ranked_labels = torch.gather(expanded_gallery_labels, 1, ranked_indices)
    ranked_relevance = ranked_labels.eq(query_labels.unsqueeze(1))

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = ranked_relevance[:, :k].any(dim=1).float().mean().item() * 100.0

    ap_scores: List[float] = []
    first_hit_ranks: List[Optional[int]] = []

    for q_idx in range(query_labels.shape[0]):
        relevance_row = ranked_relevance[q_idx]
        ap_scores.append(average_precision_from_relevance(relevance_row))

        hit_positions = torch.where(relevance_row)[0]
        first_hit_rank = int(hit_positions[0].item()) + 1 if len(hit_positions) > 0 else None
        first_hit_ranks.append(first_hit_rank)

    metrics["mAP"] = sum(ap_scores) / max(len(ap_scores), 1) * 100.0

    per_query_results: List[Dict[str, Any]] = []
    for q_idx in range(query_labels.shape[0]):
        top1_index = int(ranked_indices[q_idx, 0].item())
        row: Dict[str, Any] = {
            "query_index": int(q_idx),
            "query_label": int(query_labels[q_idx].item()),
            "top1_gallery_index": top1_index,
            "top1_gallery_label": int(gallery_labels[top1_index].item()),
            "top1_score": float(similarity_matrix[q_idx, top1_index].item()),
            "AP_percent": ap_scores[q_idx] * 100.0,
            "first_hit_rank": first_hit_ranks[q_idx],
        }
        for k in ks:
            row[f"hit@{k}"] = bool(ranked_relevance[q_idx, :k].any().item())
        per_query_results.append(row)

    return metrics, ranked_indices, ranked_relevance, per_query_results


def evaluate_multilabel_retrieval_from_similarity(
    similarity_matrix: torch.Tensor,
    *,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Evaluate multi-label retrieval metrics from a precomputed similarity matrix."""
    similarity_matrix = similarity_matrix.cpu().float()
    query_labels = query_labels.cpu().bool()
    gallery_labels = gallery_labels.cpu().bool()

    if query_labels.ndim != 2 or gallery_labels.ndim != 2:
        raise ValueError("Multi-label evaluation expects labels of shape [N, C].")

    ranked_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    relevance_matrix = (query_labels.float() @ gallery_labels.float().T) > 0
    ranked_relevance = torch.gather(
        relevance_matrix.to(torch.int64),
        1,
        ranked_indices,
    ).bool()

    total_relevant = relevance_matrix.sum(dim=1).clamp(min=1)
    metrics: Dict[str, float] = {}

    for k in ks:
        topk_rel = ranked_relevance[:, :k].float()
        precision_q = topk_rel.mean(dim=1)
        recall_q = topk_rel.sum(dim=1) / total_relevant.float()
        f1_q = torch.where(
            (precision_q + recall_q) > 0,
            2.0 * precision_q * recall_q / (precision_q + recall_q),
            torch.zeros_like(precision_q),
        )

        metrics[f"R@{k}"] = ranked_relevance[:, :k].any(dim=1).float().mean().item() * 100.0
        metrics[f"P@{k}"] = precision_q.mean().item() * 100.0
        metrics[f"ML_Recall@{k}"] = recall_q.mean().item() * 100.0
        metrics[f"F1@{k}"] = f1_q.mean().item() * 100.0

    ap_scores: List[float] = []
    per_query_results: List[Dict[str, Any]] = []

    for q_idx in range(query_labels.shape[0]):
        relevance_row = ranked_relevance[q_idx]
        ap = average_precision_from_relevance(relevance_row)
        ap_scores.append(ap)

        hit_positions = torch.where(relevance_row)[0]
        first_hit_rank = int(hit_positions[0].item()) + 1 if len(hit_positions) > 0 else None
        top1_index = int(ranked_indices[q_idx, 0].item())

        row: Dict[str, Any] = {
            "query_index": int(q_idx),
            "num_active_labels": int(query_labels[q_idx].sum().item()),
            "num_relevant_gallery": int(relevance_matrix[q_idx].sum().item()),
            "top1_gallery_index": top1_index,
            "top1_score": float(similarity_matrix[q_idx, top1_index].item()),
            "AP_percent": ap * 100.0,
            "first_hit_rank": first_hit_rank,
        }
        for k in ks:
            row[f"hit@{k}"] = bool(ranked_relevance[q_idx, :k].any().item())
        per_query_results.append(row)

    metrics["mAP"] = sum(ap_scores) / max(len(ap_scores), 1) * 100.0
    return metrics, ranked_indices, ranked_relevance, per_query_results


def _search_best_alpha(
    base_similarity: torch.Tensor,
    *,
    patch_affinity: Dict[str, torch.Tensor | int],
    alphas: Sequence[float],
    evaluator,
    evaluator_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if not alphas:
        raise ValueError("alphas must contain at least one candidate value.")

    alpha_search_rows: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None

    for alpha in alphas:
        refined_similarity = refine_similarity_matrix(
            base_similarity,
            patch_affinity=patch_affinity,
            alpha=float(alpha),
        )
        metrics, ranked_indices, ranked_relevance, per_query_results = evaluator(
            refined_similarity,
            **evaluator_kwargs,
        )

        alpha_row = {
            "alpha": float(alpha),
            **{metric_name: float(metric_value) for metric_name, metric_value in metrics.items()},
        }
        alpha_search_rows.append(alpha_row)

        candidate = {
            "alpha": float(alpha),
            "metrics": metrics,
            "similarity_matrix": refined_similarity,
            "ranked_indices": ranked_indices,
            "ranked_relevance": ranked_relevance,
            "per_query_results": per_query_results,
        }

        if best_result is None:
            best_result = candidate
            continue

        best_key = (
            float(best_result["metrics"].get("R@1", 0.0)),
            float(best_result["metrics"].get("R@5", 0.0)),
            float(best_result["metrics"].get("mAP", 0.0)),
        )
        current_key = (
            float(metrics.get("R@1", 0.0)),
            float(metrics.get("R@5", 0.0)),
            float(metrics.get("mAP", 0.0)),
        )
        if current_key > best_key:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("Failed to evaluate any alpha candidates.")

    return {
        "best": best_result,
        "alpha_search": alpha_search_rows,
    }


def run_eurosat_rs_transclip_baseline(
    root: Path,
    clip_model,
    device: torch.device,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 224,
    patch_pool_size: int = 4,
    affinity_topk: int = 20,
    affinity_chunk_size: int = 256,
    alphas: Sequence[float] = (0.3, 0.5, 0.7),
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the RS-TransCLIP-style RGB baseline on EuroSAT-MS."""
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

    rgb_indices = (
        get_eurosat_band_index("B04"),
        get_eurosat_band_index("B03"),
        get_eurosat_band_index("B02"),
    )

    query = encode_loader_with_rgb_patches(
        query_loader,
        clip_model,
        device,
        rgb_indices=rgb_indices,
        image_size=image_size,
        patch_pool_size=patch_pool_size,
        metadata_keys=("label", "label_name", "path"),
        desc="Encoding EuroSAT RS-TransCLIP query",
        show_progress=show_progress,
    )
    gallery = encode_loader_with_rgb_patches(
        gallery_loader,
        clip_model,
        device,
        rgb_indices=rgb_indices,
        image_size=image_size,
        patch_pool_size=patch_pool_size,
        metadata_keys=("label", "label_name", "path"),
        desc="Encoding EuroSAT RS-TransCLIP gallery",
        show_progress=show_progress,
    )

    patch_affinity = build_gallery_patch_affinity_knn(
        gallery["patch_descriptors"],
        topk=affinity_topk,
        chunk_size=affinity_chunk_size,
        compute_device=device,
        show_progress=show_progress,
    )

    base_similarity = query["features"] @ gallery["features"].T
    base_metrics, _, _, _ = evaluate_single_label_retrieval_from_similarity(
        base_similarity,
        query_labels=query["label"],
        gallery_labels=gallery["label"],
        ks=(1, 5, 10),
    )

    search_result = _search_best_alpha(
        base_similarity,
        patch_affinity=patch_affinity,
        alphas=alphas,
        evaluator=evaluate_single_label_retrieval_from_similarity,
        evaluator_kwargs={
            "query_labels": query["label"],
            "gallery_labels": gallery["label"],
            "ks": (1, 5, 10),
        },
    )
    best = search_result["best"]

    summary = {
        "dataset": "EuroSAT_MS",
        "task": "image_to_image_single_label",
        "variant": "RS-TransCLIP_RGB_cross_patch",
        "num_queries": int(query["label"].numel()),
        "num_gallery": int(gallery["label"].numel()),
        "num_classes": int(len(dataset.class_names)),
        "seed": seed,
        "best_alpha": float(best["alpha"]),
        "alpha_candidates": "|".join(str(float(alpha)) for alpha in alphas),
        "patch_pool_size": int(patch_pool_size),
        "affinity_topk": int(patch_affinity["topk"]),
        **{f"base_{name}": float(value) for name, value in base_metrics.items()},
        **{name: float(value) for name, value in best["metrics"].items()},
    }

    for row, path, label_name in zip(best["per_query_results"], query["path"], query["label_name"]):
        row["path"] = path
        row["label_name"] = label_name
        row["alpha"] = float(best["alpha"])

    return {
        "summary": summary,
        "metrics": best["metrics"],
        "base_metrics": base_metrics,
        "best_alpha": float(best["alpha"]),
        "alpha_search": search_result["alpha_search"],
        "query": query,
        "gallery": gallery,
        "patch_affinity": patch_affinity,
        "similarity_matrix": best["similarity_matrix"],
        "ranked_indices": best["ranked_indices"],
        "ranked_relevance": best["ranked_relevance"],
        "per_query_results": best["per_query_results"],
    }


def run_bigearth_rs_transclip_baseline(
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
    patch_pool_size: int = 4,
    affinity_topk: int = 20,
    affinity_chunk_size: int = 256,
    alphas: Sequence[float] = (0.3, 0.5, 0.7),
    remove_snow_cloud_shadow: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the RS-TransCLIP-style RGB baseline on BigEarthNet-S2."""
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

    rgb_indices = (
        get_bigearth_band_index("B04"),
        get_bigearth_band_index("B03"),
        get_bigearth_band_index("B02"),
    )

    query = encode_loader_with_rgb_patches(
        query_loader,
        clip_model,
        device,
        rgb_indices=rgb_indices,
        image_size=image_size,
        patch_pool_size=patch_pool_size,
        metadata_keys=("labels", "label_names", "patch_name", "index"),
        desc="Encoding BigEarth RS-TransCLIP query",
        show_progress=show_progress,
    )
    gallery = encode_loader_with_rgb_patches(
        gallery_loader,
        clip_model,
        device,
        rgb_indices=rgb_indices,
        image_size=image_size,
        patch_pool_size=patch_pool_size,
        metadata_keys=("labels", "label_names", "patch_name", "index"),
        desc="Encoding BigEarth RS-TransCLIP gallery",
        show_progress=show_progress,
    )

    patch_affinity = build_gallery_patch_affinity_knn(
        gallery["patch_descriptors"],
        topk=affinity_topk,
        chunk_size=affinity_chunk_size,
        compute_device=device,
        show_progress=show_progress,
    )

    base_similarity = query["features"] @ gallery["features"].T
    base_metrics, _, _, _ = evaluate_multilabel_retrieval_from_similarity(
        base_similarity,
        query_labels=query["labels"],
        gallery_labels=gallery["labels"],
        ks=(1, 5, 10),
    )

    search_result = _search_best_alpha(
        base_similarity,
        patch_affinity=patch_affinity,
        alphas=alphas,
        evaluator=evaluate_multilabel_retrieval_from_similarity,
        evaluator_kwargs={
            "query_labels": query["labels"],
            "gallery_labels": gallery["labels"],
            "ks": (1, 5, 10),
        },
    )
    best = search_result["best"]

    label_density_q = query["labels"].sum(dim=1).float().mean().item()
    label_density_g = gallery["labels"].sum(dim=1).float().mean().item()

    summary = {
        "dataset": "BigEarthNetS2",
        "task": "image_to_image_multi_label",
        "variant": "RS-TransCLIP_RGB_cross_patch",
        "num_queries": int(query["labels"].shape[0]),
        "num_gallery": int(gallery["labels"].shape[0]),
        "num_classes": int(query["labels"].shape[1]),
        "seed": seed,
        "best_alpha": float(best["alpha"]),
        "alpha_candidates": "|".join(str(float(alpha)) for alpha in alphas),
        "patch_pool_size": int(patch_pool_size),
        "affinity_topk": int(patch_affinity["topk"]),
        "avg_labels_per_query": float(label_density_q),
        "avg_labels_per_gallery": float(label_density_g),
        **{f"base_{name}": float(value) for name, value in base_metrics.items()},
        **{name: float(value) for name, value in best["metrics"].items()},
    }

    for row, patch_name, label_names in zip(
        best["per_query_results"],
        query["patch_name"],
        query["label_names"],
    ):
        row["patch_name"] = patch_name
        row["label_names"] = "|".join(label_names)
        row["alpha"] = float(best["alpha"])

    return {
        "summary": summary,
        "metrics": best["metrics"],
        "base_metrics": base_metrics,
        "best_alpha": float(best["alpha"]),
        "alpha_search": search_result["alpha_search"],
        "query": query,
        "gallery": gallery,
        "patch_affinity": patch_affinity,
        "similarity_matrix": best["similarity_matrix"],
        "ranked_indices": best["ranked_indices"],
        "ranked_relevance": best["ranked_relevance"],
        "per_query_results": best["per_query_results"],
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RS-TransCLIP RGB cross-patch baseline")
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
        default=Path("results/rs_transclip_baseline_results.csv"),
    )
    parser.add_argument(
        "--eurosat-per-query-csv",
        type=Path,
        default=Path("results/rs_transclip_eurosat_per_query.csv"),
    )
    parser.add_argument(
        "--bigearth-per-query-csv",
        type=Path,
        default=Path("results/rs_transclip_bigearth_per_query.csv"),
    )
    parser.add_argument(
        "--eurosat-alpha-grid-csv",
        type=Path,
        default=Path("results/rs_transclip_eurosat_alpha_grid.csv"),
    )
    parser.add_argument(
        "--bigearth-alpha-grid-csv",
        type=Path,
        default=Path("results/rs_transclip_bigearth_alpha_grid.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eurosat-batch-size", type=int, default=64)
    parser.add_argument("--bigearth-batch-size", type=int, default=32)
    parser.add_argument("--bigearth-max-samples", type=int, default=100_000)
    parser.add_argument("--bigearth-query-size", type=int, default=1_000)
    parser.add_argument("--bigearth-gallery-size", type=int, default=10_000)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="Refinement weights to evaluate.",
    )
    parser.add_argument("--patch-pool-size", type=int, default=4)
    parser.add_argument("--affinity-topk", type=int, default=20)
    parser.add_argument("--affinity-chunk-size", type=int, default=256)
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
        print("\n[EuroSAT] Running RS-TransCLIP RGB cross-patch baseline...")
        eurosat_result = run_eurosat_rs_transclip_baseline(
            root=parsed.eurosat_root,
            clip_model=clip_model,
            device=device,
            batch_size=parsed.eurosat_batch_size,
            num_workers=parsed.num_workers,
            seed=parsed.seed,
            patch_pool_size=parsed.patch_pool_size,
            affinity_topk=parsed.affinity_topk,
            affinity_chunk_size=parsed.affinity_chunk_size,
            alphas=parsed.alphas,
            show_progress=not parsed.hide_progress,
        )
        summary_rows.append(eurosat_result["summary"])
        save_csv_rows(eurosat_result["per_query_results"], parsed.eurosat_per_query_csv)
        save_csv_rows(eurosat_result["alpha_search"], parsed.eurosat_alpha_grid_csv)
        print(f"EuroSAT best alpha: {eurosat_result['best_alpha']:.2f}")
        for key, value in eurosat_result["metrics"].items():
            print(f"  {key}: {value:.2f}%")

    if parsed.datasets in ("bigearth", "both"):
        print("\n[BigEarthNet] Running RS-TransCLIP RGB cross-patch baseline...")
        bigearth_result = run_bigearth_rs_transclip_baseline(
            root=parsed.bigearth_root,
            clip_model=clip_model,
            device=device,
            batch_size=parsed.bigearth_batch_size,
            num_workers=parsed.num_workers,
            max_samples=parsed.bigearth_max_samples,
            query_size=parsed.bigearth_query_size,
            gallery_size=parsed.bigearth_gallery_size,
            seed=parsed.seed,
            patch_pool_size=parsed.patch_pool_size,
            affinity_topk=parsed.affinity_topk,
            affinity_chunk_size=parsed.affinity_chunk_size,
            alphas=parsed.alphas,
            remove_snow_cloud_shadow=not parsed.keep_snow_cloud_shadow,
            show_progress=not parsed.hide_progress,
        )
        summary_rows.append(bigearth_result["summary"])
        save_csv_rows(bigearth_result["per_query_results"], parsed.bigearth_per_query_csv)
        save_csv_rows(bigearth_result["alpha_search"], parsed.bigearth_alpha_grid_csv)
        print(f"BigEarthNet best alpha: {bigearth_result['best_alpha']:.2f}")
        for key, value in bigearth_result["metrics"].items():
            print(f"  {key}: {value:.2f}%")

    save_csv_rows(summary_rows, parsed.results_csv)
    print(f"\nSaved summary results to {parsed.results_csv}")


if __name__ == "__main__":
    main()
