"""
CLIP-RGB baselines for EuroSAT-MS multispectral retrieval.

Two retrieval paradigms are provided:

    - run_clip_rgb_text_to_image_baseline():
        Legacy text-to-image paradigm. Query features are text embeddings
        (one per class). Used in early CLIP zero-shot evaluation.

    - run_clip_rgb_image_to_image_baseline():
        Standard image-to-image paradigm matching all other baselines.
        Both query and gallery are encoded as CLIP image features.
        This is the correct paradigm for comparing against PCA, NDVI,
        RS-TransCLIP, and Ours.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.models.clip_utils import encode_test_gallery_rgb
from src.utils.metrics import evaluate_text_to_image_retrieval


__all__ = [
    "run_clip_rgb_text_to_image_baseline",
    "run_clip_rgb_image_to_image_baseline",
]


@torch.no_grad()
def run_clip_rgb_text_to_image_baseline(
    test_loader,
    model,
    text_features: torch.Tensor,
    query_labels: torch.Tensor,
    device: str | torch.device,
    ks: Sequence[int] = (1, 5, 10),
    image_size: int = 224,
    rgb_indices: Sequence[int] = (3, 2, 1),
    show_progress: bool = True,
):
    """
    Text-to-image CLIP-RGB retrieval baseline (legacy).

    Uses one text embedding per class as the query, and encodes the gallery
    images with CLIP's image encoder. This is a zero-shot classification-style
    paradigm and does **not** match the image-to-image protocol used by the
    other baselines (PCA, NDVI, RS-TransCLIP, Ours).

    For a fair comparison against the other baselines, use
    ``run_clip_rgb_image_to_image_baseline()`` instead.

    Args:
        test_loader: Gallery dataloader for test images.
        model: CLIP model (image encoder used for gallery).
        text_features: L2-normalized text embeddings [Q, D], one per class.
        query_labels: Query class labels [Q].
        device: cpu / cuda / mps.
        ks: Recall cutoffs.
        image_size: CLIP input spatial resolution.
        rgb_indices: Zero-based channel indices for RGB extraction.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict with keys: gallery, metrics, similarity_matrix, ranked_indices,
        ranked_relevance, per_query_results.
    """
    gallery = encode_test_gallery_rgb(
        loader=test_loader,
        model=model,
        device=device,
        image_size=image_size,
        rgb_indices=rgb_indices,
        show_progress=show_progress,
    )

    metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results = (
        evaluate_text_to_image_retrieval(
            query_features=text_features,
            query_labels=query_labels,
            gallery_features=gallery["features"],
            gallery_labels=gallery["labels"],
            ks=ks,
        )
    )

    return {
        "gallery": gallery,
        "metrics": metrics,
        "similarity_matrix": similarity_matrix,
        "ranked_indices": ranked_indices,
        "ranked_relevance": ranked_relevance,
        "per_query_results": per_query_results,
    }


@torch.no_grad()
def run_clip_rgb_image_to_image_baseline(
    query_loader,
    gallery_loader,
    model,
    device: str | torch.device,
    ks: Sequence[int] = (1, 5, 10),
    image_size: int = 224,
    rgb_indices: Sequence[int] = (3, 2, 1),
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Image-to-image CLIP-RGB retrieval baseline.

    Encodes both query and gallery images with CLIP's image encoder (RGB  
    composite using ``rgb_indices`` bands). Both query and gallery are encoded  
    identically — this is the correct paradigm for comparing against PCA,  
    NDVI, RS-TransCLIP, and the 13-band pipeline (Ours).

    RGB composite is constructed as:
        image[[rgb_indices[0], rgb_indices[1], rgb_indices[2]], :, :]
    then fed to CLIP's visual encoder after bicubic resize to ``image_size``.

    Args:
        query_loader: DataLoader yielding query images with keys
            ``"image"`` [N, C, H, W], ``"label"`` [N], ``"label_name"``,
            ``"path"``, ``"index"``.
        gallery_loader: DataLoader with same structure for gallery images.
        model: Frozen OpenAI CLIP model (``clip.load()`` output).
        device: Target device (cpu / cuda / mps).
        ks: Recall@K cutoffs, e.g. (1, 5, 10).
        image_size: Spatial resolution for CLIP input (default 224).
        rgb_indices: Zero-based channel indices for RGB extraction.
            Default (3, 2, 1) = B04, B03, B02 for EuroSAT-MS / Sentinel-2.
        show_progress: Show tqdm progress bars.

    Returns:
        Dict with keys:
            - ``query``: dict with keys features [Q, D], label [Q],
              label_name List[str], path List[str], index [Q].
            - ``gallery``: same structure for [N] gallery images.
            - ``metrics``: dict {"R@1": %, "R@5": %, "R@10": %, "mAP": %}.
            - ``similarity_matrix``: [Q, N] cosine similarity.
            - ``ranked_indices``: [Q, N] gallery indices descending by sim.
            - ``ranked_relevance``: [Q, N] boolean correct-class mask.
            - ``per_query_results``: List of per-query summary dicts.
    """
    from src.models.clip_utils import preprocess_rgb_for_clip

    def _encode_split(loader, desc: str) -> Dict[str, Any]:
        feature_chunks: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        label_names: List[str] = []
        paths: List[str] = []
        indices: List[torch.Tensor] = []

        iterator = tqdm(loader, desc=desc) if show_progress else loader
        model.eval()

        for batch in iterator:
            images = batch["image"].to(torch.float32)
            # Build RGB composite [N, 3, H, W]
            rgb = images[:, list(rgb_indices), :, :]
            # Preprocess for CLIP (resize, normalize)
            clip_inputs = preprocess_rgb_for_clip(rgb, image_size=image_size).to(device)
            features = model.encode_image(clip_inputs).float()
            features = F.normalize(features, dim=-1)
            feature_chunks.append(features.cpu())

            labels.append(batch["label"].cpu())
            label_names.extend([str(n) for n in batch["label_name"]])
            paths.extend([str(p) for p in batch["path"]])

            batch_idx = batch["index"]
            if torch.is_tensor(batch_idx):
                indices.append(batch_idx.cpu())
            else:
                indices.append(torch.tensor(list(batch_idx), dtype=torch.long))

        return {
            "features": torch.cat(feature_chunks, dim=0),
            "label": torch.cat(labels, dim=0),
            "label_name": label_names,
            "path": paths,
            "index": torch.cat(indices, dim=0),
        }

    query_data = _encode_split(query_loader, desc="RGB-CLIP encode query")
    gallery_data = _encode_split(gallery_loader, desc="RGB-CLIP encode gallery")

    metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results = (
        evaluate_text_to_image_retrieval(
            query_features=query_data["features"],
            query_labels=query_data["label"],
            gallery_features=gallery_data["features"],
            gallery_labels=gallery_data["label"],
            ks=ks,
        )
    )

    return {
        "query": query_data,
        "gallery": gallery_data,
        "metrics": metrics,
        "similarity_matrix": similarity_matrix,
        "ranked_indices": ranked_indices,
        "ranked_relevance": ranked_relevance,
        "per_query_results": per_query_results,
    }
