from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def _ensure_2d_query(query_embedding: torch.Tensor) -> torch.Tensor:
    """
    Ensure query embedding has shape (D,).
    Accepts (D,) or (1, D).
    """
    if query_embedding.ndim == 2:
        if query_embedding.shape[0] != 1:
            raise ValueError(
                f"query_embedding with ndim=2 must have shape (1, D), got {tuple(query_embedding.shape)}"
            )
        query_embedding = query_embedding.squeeze(0)

    if query_embedding.ndim != 1:
        raise ValueError(
            f"query_embedding must have shape (D,) or (1, D), got {tuple(query_embedding.shape)}"
        )

    return query_embedding


def compute_query_weights(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    sigma: float = 0.5,
    normalize_inputs: bool = True,
) -> torch.Tensor:
    """
    Compute query-conditioned weights for each spectral band.

    Args:
        band_embeddings: Tensor of shape (B, D)
        query_embedding: Tensor of shape (D,) or (1, D)
        sigma: temperature parameter for softmax
        normalize_inputs: if True, L2-normalize band and query embeddings first

    Returns:
        query_weights: Tensor of shape (B,), sum = 1
    """
    if band_embeddings.ndim != 2:
        raise ValueError(
            f"band_embeddings must have shape (B, D), got {tuple(band_embeddings.shape)}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    query_embedding = _ensure_2d_query(query_embedding)

    if normalize_inputs:
        band_embeddings = F.normalize(band_embeddings, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)

    # (B,)
    query_scores = (band_embeddings @ query_embedding) / sigma
    query_weights = torch.softmax(query_scores, dim=0)

    return query_weights


def compute_pairwise_similarity(
    band_embeddings: torch.Tensor,
    normalize_inputs: bool = True,
    clamp_min_zero: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise similarity matrix S between bands.

    Args:
        band_embeddings: Tensor of shape (B, D)
        normalize_inputs: if True, use cosine-like similarity via L2 normalization
        clamp_min_zero: if True, clamp negative similarities to 0 to keep graph non-negative

    Returns:
        S: Tensor of shape (B, B)
    """
    if band_embeddings.ndim != 2:
        raise ValueError(
            f"band_embeddings must have shape (B, D), got {tuple(band_embeddings.shape)}"
        )

    if normalize_inputs:
        band_embeddings = F.normalize(band_embeddings, dim=-1)

    S = band_embeddings @ band_embeddings.T  # (B, B)

    if clamp_min_zero:
        S = torch.clamp(S, min=0.0)

    return S


def symmetric_normalize(
    A: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric normalization: A_norm = D^{-1/2} A D^{-1/2}

    Args:
        A: affinity matrix of shape (B, B)
        eps: small constant to avoid division by zero

    Returns:
        A_norm: normalized affinity matrix of shape (B, B)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {tuple(A.shape)}")

    degree = A.sum(dim=1)  # (B,)
    inv_sqrt_degree = torch.pow(degree + eps, -0.5)
    D_inv_sqrt = torch.diag(inv_sqrt_degree)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


def compute_affinity_graph(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    sigma: float = 0.5,
    tau: Optional[float] = None,
    normalize_inputs: bool = True,
    clamp_similarity_nonnegative: bool = True,
    return_details: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute query-conditioned affinity graph for spectral bands.

    WeeklyTasks formulation:
        1) query_weights = softmax((band @ query) / sigma)
        2) S = band @ band.T  [optionally softmax-normalized with tau]
        3) A = S * (w_i * w_j)
        4) A_norm = D^{-1/2} A D^{-1/2}

    Args:
        band_embeddings: Tensor of shape (B, D)
        query_embedding: Tensor of shape (D,) or (1, D)
        sigma: softmax temperature for query alignment (Eq. 1)
        tau: softmax temperature for affinity normalization (optional).
            When None (default), the raw pairwise similarity is used as-is
            (original behavior, backward-compatible).
            When a float, applies row-wise softmax(S / tau) to the
            pairwise similarity matrix before weighting:
              - small tau → sharper (concentrate on most similar band pairs)
              - large tau → softer (more uniform distribution)
            Typical range for sensitivity scan: [0.05, 0.1, 0.2, 0.5].
        normalize_inputs: L2-normalize embeddings before dot products
        clamp_similarity_nonnegative: clamp S >= 0 to keep graph stable
            (ignored when tau is not None, as softmax output is always >= 0)
        return_details: if True, also return intermediate tensors

    Returns:
        A_norm
        optionally details dict containing:
            - query_weights
            - similarity_matrix
            - affinity_raw
            - degree
    """
    query_weights = compute_query_weights(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        sigma=sigma,
        normalize_inputs=normalize_inputs,
    )  # (B,)

    similarity_matrix = compute_pairwise_similarity(
        band_embeddings=band_embeddings,
        normalize_inputs=normalize_inputs,
        clamp_min_zero=(clamp_similarity_nonnegative and tau is None),
    )  # (B, B)

    # Optional: softmax-normalize pairwise similarity with temperature tau
    # (tau=None → keep raw similarity, backward-compatible)
    if tau is not None:
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        similarity_matrix = torch.softmax(similarity_matrix / tau, dim=-1)

    # Outer product of weights: (B, 1) * (1, B) -> (B, B)
    weight_matrix = query_weights.unsqueeze(1) * query_weights.unsqueeze(0)

    affinity_raw = similarity_matrix * weight_matrix
    affinity_raw = 0.5 * (affinity_raw + affinity_raw.T)  # enforce exact symmetry

    A_norm = symmetric_normalize(affinity_raw)

    if not return_details:
        return A_norm

    details = {
        "query_weights": query_weights,
        "similarity_matrix": similarity_matrix,
        "affinity_raw": affinity_raw,
        "degree": affinity_raw.sum(dim=1),
    }
    return A_norm, details