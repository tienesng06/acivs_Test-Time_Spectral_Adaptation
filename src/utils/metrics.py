"""
Retrieval evaluation metrics for multispectral image retrieval.

Supports both single-label (EuroSAT) and multi-label (BigEarthNet) retrieval.

Public API:
    - average_precision_from_relevance()   : AP for one ranked result list
    - evaluate_text_to_image_retrieval()   : single-label image retrieval metrics
    - evaluate_multilabel_image_retrieval(): multi-label image retrieval metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "average_precision_from_relevance",
    "evaluate_text_to_image_retrieval",
    "evaluate_multilabel_image_retrieval",
]


def average_precision_from_relevance(relevance_row: torch.Tensor) -> float:
    """
    Compute Average Precision for one ranked retrieval result.

    Args:
        relevance_row:
            Boolean or 0/1 tensor with shape [N_gallery], already sorted by rank.

    Returns:
        Average Precision in [0, 1].
    """
    relevance_row = relevance_row.bool().float()
    num_relevant = int(relevance_row.sum().item())

    if num_relevant == 0:
        return 0.0

    precision_at_k = torch.cumsum(relevance_row, dim=0) / torch.arange(
        1,
        relevance_row.numel() + 1,
        device=relevance_row.device,
        dtype=torch.float32,
    )
    ap = (precision_at_k * relevance_row).sum() / num_relevant
    return float(ap.item())


def evaluate_text_to_image_retrieval(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
):
    """
    Evaluate single-label image-to-image retrieval using cosine similarity.

    Assumes both query_features and gallery_features are already L2-normalized.
    For multi-label retrieval, use ``evaluate_multilabel_image_retrieval()`` instead.

    Args:
        query_features:   [Q, D]  — L2-normalized query embeddings
        query_labels:     [Q]     — integer class labels (1-D)
        gallery_features: [N, D]  — L2-normalized gallery embeddings
        gallery_labels:   [N]     — integer class labels (1-D)
        ks:
            Recall cutoffs, e.g. (1, 5, 10).

    Returns:
        metrics:
            Dict like {"R@1": ..., "R@5": ..., "mAP": ...} (values in %)
        similarity_matrix:
            [Q, N]
        ranked_indices:
            [Q, N], descending by similarity
        ranked_relevance:
            [Q, N], boolean relevance after ranking
        per_query_results:
            List of per-query summary dicts for inspection/debugging
    """
    query_features = query_features.cpu()
    query_labels = query_labels.cpu()
    gallery_features = gallery_features.cpu()
    gallery_labels = gallery_labels.cpu()

    similarity_matrix = query_features @ gallery_features.T
    ranked_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

    expanded_gallery_labels = gallery_labels.unsqueeze(0).expand(query_labels.shape[0], -1)
    ranked_labels = torch.gather(expanded_gallery_labels, 1, ranked_indices)
    ranked_relevance = ranked_labels.eq(query_labels.unsqueeze(1))

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = ranked_relevance[:, :k].any(dim=1).float().mean().item() * 100.0

    ap_scores: List[float] = []
    first_hit_ranks: List[Optional[int]] = []

    for q in range(query_features.shape[0]):
        relevance_row = ranked_relevance[q]
        ap_scores.append(average_precision_from_relevance(relevance_row))

        hit_positions = torch.where(relevance_row)[0]
        first_hit_ranks.append(
            int(hit_positions[0].item()) + 1 if len(hit_positions) > 0 else None
        )

    metrics["mAP"] = float(np.mean(ap_scores)) * 100.0

    per_query_results: List[Dict[str, Any]] = []
    for q in range(query_features.shape[0]):
        row: Dict[str, Any] = {
            "query_index": int(q),
            "query_label": int(query_labels[q].item()),
            "AP_percent": ap_scores[q] * 100.0,
            "first_hit_rank": first_hit_ranks[q],
        }
        for k in ks:
            row[f"hit@{k}"] = bool(ranked_relevance[q, :k].any().item())
        per_query_results.append(row)

    return metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results


def evaluate_multilabel_image_retrieval(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Evaluate image-to-image retrieval with multi-label relevance.

    A gallery image is considered *relevant* if it shares at least one active
    label with the query image (i.e., the binary label vectors have a non-zero
    dot product).

    Args:
        query_features:   [Q, D]  — L2-normalized query embeddings
        query_labels:     [Q, C]  — binary multi-hot label matrix (bool or 0/1)
        gallery_features: [N, D]  — L2-normalized gallery embeddings
        gallery_labels:   [N, C]  — binary multi-hot label matrix (bool or 0/1)
        ks:
            Evaluation cutoffs, e.g. (1, 5, 10).

    Returns:
        metrics:
            Dict with keys R@K, P@K, ML_Recall@K, F1@K, mAP (values in %)
        similarity_matrix:
            [Q, N] cosine similarity
        ranked_indices:
            [Q, N] gallery indices sorted descending by similarity
        ranked_relevance:
            [Q, N] boolean relevance after ranking
        per_query_results:
            List of per-query summary dicts

    Raises:
        ValueError: if label tensors are not 2-D.
    """
    query_features = F.normalize(query_features.cpu().float(), dim=-1)
    gallery_features = F.normalize(gallery_features.cpu().float(), dim=-1)
    query_labels = query_labels.cpu().bool()
    gallery_labels = gallery_labels.cpu().bool()

    if query_labels.ndim != 2 or gallery_labels.ndim != 2:
        raise ValueError(
            "Multi-label evaluation expects label tensors of shape [N, C]. "
            f"Got query_labels.ndim={query_labels.ndim}, "
            f"gallery_labels.ndim={gallery_labels.ndim}."
        )

    similarity_matrix = query_features @ gallery_features.T
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

    for q_idx in range(query_features.shape[0]):
        relevance_row = ranked_relevance[q_idx]
        ap = average_precision_from_relevance(relevance_row)
        ap_scores.append(ap)

        hit_positions = torch.where(relevance_row)[0]
        first_hit_rank = int(hit_positions[0].item()) + 1 if len(hit_positions) > 0 else None

        row: Dict[str, Any] = {
            "query_index": q_idx,
            "num_active_labels": int(query_labels[q_idx].sum().item()),
            "num_relevant_gallery": int(relevance_matrix[q_idx].sum().item()),
            "AP_percent": ap * 100.0,
            "first_hit_rank": first_hit_rank,
        }
        for k in ks:
            row[f"hit@{k}"] = bool(ranked_relevance[q_idx, :k].any().item())
        per_query_results.append(row)

    metrics["mAP"] = sum(ap_scores) / max(len(ap_scores), 1) * 100.0
    return metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results
