from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch



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
    Evaluate text-to-image retrieval using cosine similarity.

    Assumes both query_features and gallery_features are already L2-normalized.

    Args:
        query_features:  [Q, D]
        query_labels:    [Q]
        gallery_features:[N, D]
        gallery_labels:  [N]
        ks:
            Recall cutoffs, e.g. (1, 5, 10).

    Returns:
        metrics:
            Dict like {"R@1": ..., "R@5": ..., "mAP": ...}
        similarity_matrix:
            [Q, N]
        ranked_indices:
            [Q, N], descending by similarity
        ranked_relevance:
            [Q, N], boolean relevance after ranking
        per_query_results:
            Lightweight per-query summary for inspection/debugging
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

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = ranked_relevance[:, :k].any(dim=1).float().mean().item() * 100.0

    # Calculate for Multi-labeled datasets
    if query_labels.ndim > 1:
        for k in ks:
            hits_in_k = ranked_relevance[:, :k].float().sum(dim=1) 
            total_relevant = relevance_matrix.float().sum(dim=1) 
            
            precision_k = (hits_in_k / k).mean().item() * 100.0
            recall_k = (hits_in_k / torch.clamp(total_relevant, min=1)).mean().item() * 100.0
            
            f1_k = 0.0
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                
            metrics[f"P@{k}"] = precision_k
            metrics[f"ML_Recall@{k}"] = recall_k
            metrics[f"F1@{k}"] = f1_k

    ap_scores = []
    first_hit_ranks = []

    for q in range(query_features.shape[0]):
        relevance_row = ranked_relevance[q]
        ap_scores.append(average_precision_from_relevance(relevance_row))

        hit_positions = torch.where(relevance_row)[0]
        first_hit_ranks.append(int(hit_positions[0].item()) + 1 if len(hit_positions) > 0 else None)

    metrics["mAP"] = float(np.mean(ap_scores)) * 100.0

    per_query_results = []
    for q in range(query_features.shape[0]):
        row = {
            "query_index": int(q),
            "query_label": int(query_labels[q].item()),
            "AP_percent": ap_scores[q] * 100.0,
            "first_hit_rank": first_hit_ranks[q],
        }
        for k in ks:
            row[f"hit@{k}"] = bool(ranked_relevance[q, :k].any().item())
        per_query_results.append(row)

    return metrics, similarity_matrix, ranked_indices, ranked_relevance, per_query_results
