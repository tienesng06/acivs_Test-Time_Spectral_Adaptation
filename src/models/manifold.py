"""
Manifold consistency module for test-time spectral adaptation.

Implements k-NN graph construction and manifold consistency loss
(Equation 6 from the paper) to preserve local structure of band
embeddings during weighted fusion.

Core functions:
    - build_knn_graph()           : k nearest-neighbor indices via L2 distance
    - manifold_consistency_loss() : penalty for distorting local neighborhood
    - compute_fused_embedding()   : weighted sum of band embeddings
    - check_knn_preservation()    : diagnostic for before/after fusion
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# 1) Input validation
# ============================================================

def _validate_band_embeddings(band_embeddings: torch.Tensor) -> None:
    """Ensure band_embeddings is 2-D (B, D)."""
    if not isinstance(band_embeddings, torch.Tensor):
        raise TypeError("band_embeddings must be a torch.Tensor")
    if band_embeddings.ndim != 2:
        raise ValueError(
            f"band_embeddings must be 2D (B, D), got shape {tuple(band_embeddings.shape)}"
        )
    if band_embeddings.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 bands, got {band_embeddings.shape[0]}"
        )


def _validate_fused_embedding(
    fused_embedding: torch.Tensor,
    expected_dim: int,
) -> None:
    """Ensure fused_embedding is 1-D with matching dimensionality."""
    if fused_embedding.ndim == 2 and fused_embedding.shape[0] == 1:
        pass  # will squeeze later
    elif fused_embedding.ndim != 1:
        raise ValueError(
            f"fused_embedding must be 1D (D,) or (1, D), got {tuple(fused_embedding.shape)}"
        )
    d = fused_embedding.shape[-1]
    if d != expected_dim:
        raise ValueError(
            f"fused_embedding dim {d} doesn't match band dim {expected_dim}"
        )


# ============================================================
# 2) k-NN graph construction
# ============================================================

def build_knn_graph(
    band_embeddings: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Build a k-nearest-neighbor graph over band embeddings.

    For each of the B bands, find k closest neighbors (excluding self)
    based on L2 (Euclidean) distance.

    Args:
        band_embeddings: (B, D) tensor of per-band embeddings
        k: number of nearest neighbors (default 5)

    Returns:
        knn_indices: (B, k) long tensor of neighbor indices
    """
    _validate_band_embeddings(band_embeddings)

    B = band_embeddings.shape[0]
    if k >= B:
        raise ValueError(
            f"k={k} must be < number of bands B={B}"
        )

    # Pairwise L2 distances: (B, B)
    dist_matrix = torch.cdist(
        band_embeddings.unsqueeze(0),   # (1, B, D)
        band_embeddings.unsqueeze(0),   # (1, B, D)
    ).squeeze(0)                        # (B, B)

    # topk with largest=False → smallest distances first
    # Take k+1 to exclude self (distance = 0), then drop the first column
    _, topk_indices = torch.topk(dist_matrix, k + 1, dim=1, largest=False)
    knn_indices = topk_indices[:, 1:]   # (B, k)

    return knn_indices


def build_knn_graph_with_distances(
    band_embeddings: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same as build_knn_graph but also returns neighbor distances.

    Returns:
        knn_indices:   (B, k) long tensor
        knn_distances: (B, k) float tensor
    """
    _validate_band_embeddings(band_embeddings)
    B = band_embeddings.shape[0]
    if k >= B:
        raise ValueError(f"k={k} must be < B={B}")

    dist_matrix = torch.cdist(
        band_embeddings.unsqueeze(0),
        band_embeddings.unsqueeze(0),
    ).squeeze(0)

    topk_vals, topk_idx = torch.topk(dist_matrix, k + 1, dim=1, largest=False)
    return topk_idx[:, 1:], topk_vals[:, 1:]


# ============================================================
# 3) Manifold consistency loss
# ============================================================

def manifold_consistency_loss(
    fused_embedding: torch.Tensor,
    band_embeddings: torch.Tensor,
    knn_indices: torch.Tensor,
    lambda_m: float = 0.1,
) -> torch.Tensor:
    """
    Manifold consistency loss (Equation 6).

    Penalizes large L2 distances from the fused embedding to the k-NN
    neighbors of each band, thereby preserving local manifold structure.

    Formula::

        loss = lambda_m * (1/B) * sum_i [ mean_j || fused - neighbor_j(i) ||_2 ]

    This is fully vectorized (no Python for-loop).

    Args:
        fused_embedding:  (D,) or (1, D) — weighted fusion of band embeddings
        band_embeddings:  (B, D) — per-band CLIP embeddings
        knn_indices:      (B, k) — neighbor indices from build_knn_graph()
        lambda_m:         loss weight (default 0.1)

    Returns:
        loss: scalar tensor (differentiable)
    """
    _validate_band_embeddings(band_embeddings)
    B, D = band_embeddings.shape

    # Flatten fused_embedding to (D,)
    if fused_embedding.ndim == 2:
        fused_embedding = fused_embedding.squeeze(0)
    _validate_fused_embedding(fused_embedding, D)

    k = knn_indices.shape[1]

    # Gather neighbor embeddings: (B, k, D)
    # knn_indices is (B, k), need to expand to (B, k, D) for gather
    idx_expanded = knn_indices.unsqueeze(-1).expand(B, k, D)    # (B, k, D)
    neighbors = band_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, B, D)
    neighbors = torch.gather(neighbors, dim=1, index=idx_expanded)  # (B, k, D)

    # Distance from fused to each neighbor: (B, k)
    fused_expanded = fused_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    dist = torch.norm(fused_expanded - neighbors, dim=2)        # (B, k)

    # Mean distance per band, then average over bands
    per_band_mean = dist.mean(dim=1)    # (B,)
    loss = lambda_m * per_band_mean.mean()

    return loss


# ============================================================
# 4) Fused embedding helper
# ============================================================

def compute_fused_embedding(
    band_embeddings: torch.Tensor,
    weights: torch.Tensor,
    normalize_output: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute weighted fusion of band embeddings.

    Args:
        band_embeddings: (B, D)
        weights:         (B,) — should sum to 1 (softmax output)
        normalize_output: if True, L2-normalize the fused embedding
        eps: numerical stability

    Returns:
        fused: (D,) embedding
    """
    _validate_band_embeddings(band_embeddings)
    B, D = band_embeddings.shape

    if weights.ndim != 1 or weights.shape[0] != B:
        raise ValueError(
            f"weights must be (B={B},), got shape {tuple(weights.shape)}"
        )

    # (B, 1) * (B, D) → sum over B → (D,)
    fused = (weights.unsqueeze(1) * band_embeddings).sum(dim=0)

    if normalize_output:
        fused = F.normalize(fused, dim=0, eps=eps)

    return fused


# ============================================================
# 5) Diagnostics
# ============================================================

def check_knn_preservation(
    band_embeddings: torch.Tensor,
    fused_embedding: torch.Tensor,
    knn_indices: torch.Tensor,
    k_check: Optional[int] = None,
) -> Dict[str, float]:
    """
    Measure how well the fused embedding preserves k-NN structure.

    For each band i, check what fraction of i's original k-NN
    neighbors are also among the closest-k neighbors of the
    fused embedding.

    Args:
        band_embeddings: (B, D)
        fused_embedding: (D,)
        knn_indices:     (B, k) — original k-NN indices
        k_check:         number of neighbors to check (default: same as k)

    Returns:
        dict with:
            - preservation_rate: mean fraction of neighbors preserved
            - mean_neighbor_dist: mean L2 from fused to all neighbors
    """
    _validate_band_embeddings(band_embeddings)
    B, D = band_embeddings.shape

    if fused_embedding.ndim == 2:
        fused_embedding = fused_embedding.squeeze(0)

    k = knn_indices.shape[1]
    if k_check is None:
        k_check = k

    # Distances from fused embedding to all bands: (B,)
    dists_to_bands = torch.norm(
        band_embeddings - fused_embedding.unsqueeze(0), dim=1
    )

    # Top-k_check closest bands to fused embedding
    _, fused_nn = torch.topk(dists_to_bands, k_check, largest=False)
    fused_nn_set = set(fused_nn.tolist())

    # Check overlap for each band
    preserved = 0
    total = 0
    for i in range(B):
        orig_neighbors = set(knn_indices[i].tolist())
        preserved += len(orig_neighbors & fused_nn_set)
        total += len(orig_neighbors)

    preservation_rate = preserved / max(total, 1)

    # Mean distance from fused to all original neighbors
    idx_expanded = knn_indices.unsqueeze(-1).expand(B, k, D)
    neighbors = band_embeddings.unsqueeze(0).expand(B, -1, -1)
    neighbors = torch.gather(neighbors, dim=1, index=idx_expanded)
    fused_exp = fused_embedding.unsqueeze(0).unsqueeze(0)
    mean_dist = torch.norm(fused_exp - neighbors, dim=2).mean().item()

    return {
        "preservation_rate": preservation_rate,
        "mean_neighbor_dist": mean_dist,
    }
