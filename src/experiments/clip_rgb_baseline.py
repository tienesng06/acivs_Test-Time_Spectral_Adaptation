from __future__ import annotations

from typing import Sequence

import torch

from src.models.clip_utils import encode_test_gallery_rgb
from src.utils.metrics import evaluate_text_to_image_retrieval


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
    One-call helper for the EuroSAT-MS CLIP-RGB baseline.

    Args:
        test_loader:
            Gallery dataloader for test images.
        model:
            CLIP image encoder.
        text_features:
            Normalized text embeddings with shape [Q, D].
        query_labels:
            Query class labels with shape [Q].
        device:
            cpu / cuda / mps
        ks:
            Recall cutoffs.

    Returns:
        result dict containing gallery encodings + retrieval outputs.
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
