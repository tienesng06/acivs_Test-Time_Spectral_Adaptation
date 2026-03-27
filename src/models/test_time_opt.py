"""
Test-time optimization of spectral fusion weights.

Implements the core test-time adaptation loop: starting from initial
Fiedler-vector-based weights, optimize fusion weights via a few
gradient descent steps on the manifold consistency loss.

Core functions:
    - optimize_fusion_weights()    : Adam-based weight optimization
    - grid_search_hyperparams()    : search over num_steps × lr space
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.manifold import (
    build_knn_graph,
    compute_fused_embedding,
    manifold_consistency_loss,
)


# ============================================================
# 1) Optimization result dataclass
# ============================================================

@dataclass
class OptimizationResult:
    """Container for optimize_fusion_weights outputs."""

    optimized_weights: torch.Tensor    # (B,) softmax-normalized
    fused_embedding: torch.Tensor      # (D,) final fused + L2-normalized
    loss_history: List[float] = field(default_factory=list)
    num_steps_run: int = 0
    elapsed_ms: float = 0.0


# ============================================================
# 2) Core optimization function
# ============================================================

def optimize_fusion_weights(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    w_init: torch.Tensor,
    *,
    num_steps: int = 5,
    lr: float = 0.01,
    lambda_m: float = 0.1,
    k: int = 5,
    grad_clip: Optional[float] = 1.0,
    early_stop_tol: float = 1e-6,
    normalize_output: bool = True,
) -> OptimizationResult:
    """
    Optimize spectral fusion weights via test-time gradient descent.

    Starting from initial weights (e.g. Fiedler magnitude weights),
    run a few Adam steps to minimize manifold consistency loss,
    thereby adapting the fusion to the specific query at test time.

    Args:
        band_embeddings:  (B, D) per-band CLIP embeddings (detached)
        query_embedding:  (D,) or (1, D) query text/image embedding
        w_init:           (B,) initial weights (e.g. from Fiedler vector)
        num_steps:        number of Adam optimization steps (default 5)
        lr:               learning rate for Adam (default 0.01)
        lambda_m:         manifold loss weight (default 0.1)
        k:                k for k-NN graph (default 5)
        grad_clip:        max gradient norm (None to disable)
        early_stop_tol:   stop if |loss_prev - loss_cur| < tol
        normalize_output: L2-normalize the final fused embedding

    Returns:
        OptimizationResult with optimized weights, fused embedding,
        loss history, steps run, and elapsed time in ms.
    """
    t_start = time.perf_counter()

    # Validate inputs
    if band_embeddings.ndim != 2:
        raise ValueError(
            f"band_embeddings must be 2D (B, D), got {tuple(band_embeddings.shape)}"
        )
    B, D = band_embeddings.shape

    if query_embedding.ndim == 2:
        query_embedding = query_embedding.squeeze(0)
    if query_embedding.ndim != 1 or query_embedding.shape[0] != D:
        raise ValueError(
            f"query_embedding must be (D={D},), got {tuple(query_embedding.shape)}"
        )

    if w_init.ndim != 1 or w_init.shape[0] != B:
        raise ValueError(
            f"w_init must be (B={B},), got {tuple(w_init.shape)}"
        )

    # Ensure band_embeddings are detached (fixed during optimization)
    band_emb = band_embeddings.detach().clone()

    # Build k-NN graph once (not part of the optimization loop)
    knn_indices = build_knn_graph(band_emb, k=k)

    # Initialize learnable weight logits from initial weights
    # Use log of w_init as logits so softmax(logits) ≈ w_init
    w_logits = torch.log(w_init.detach().clamp(min=1e-8)).requires_grad_(True)
    optimizer = torch.optim.Adam([w_logits], lr=lr)

    loss_history: List[float] = []
    steps_run = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # Softmax-normalize weights → [0, 1], sum = 1
        w_norm = torch.softmax(w_logits, dim=0)

        # Weighted fusion
        fused_emb = compute_fused_embedding(
            band_emb, w_norm, normalize_output=False
        )

        # Manifold consistency loss
        loss = manifold_consistency_loss(
            fused_emb, band_emb, knn_indices, lambda_m=lambda_m
        )

        loss.backward()

        # Gradient clipping for stability
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([w_logits], grad_clip)

        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)
        steps_run += 1

        # Early stopping: if loss barely changed
        if step > 0 and abs(loss_history[-2] - current_loss) < early_stop_tol:
            break

    # Final weights after optimization
    with torch.no_grad():
        final_weights = torch.softmax(w_logits, dim=0)
        final_fused = compute_fused_embedding(
            band_emb, final_weights, normalize_output=normalize_output
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return OptimizationResult(
        optimized_weights=final_weights.detach(),
        fused_embedding=final_fused.detach(),
        loss_history=loss_history,
        num_steps_run=steps_run,
        elapsed_ms=elapsed_ms,
    )


# ============================================================
# 3) Grid search over hyperparameters
# ============================================================

@dataclass
class GridSearchEntry:
    """One entry in grid search results."""

    num_steps: int
    lr: float
    final_loss: float
    loss_decrease_pct: float
    elapsed_ms: float
    optimized_weights: torch.Tensor


def grid_search_hyperparams(
    band_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    w_init: torch.Tensor,
    *,
    num_steps_choices: Tuple[int, ...] = (3, 5, 7),
    lr_choices: Tuple[float, ...] = (0.005, 0.01, 0.02),
    lambda_m: float = 0.1,
    k: int = 5,
    grad_clip: Optional[float] = 1.0,
) -> List[GridSearchEntry]:
    """
    Grid search over num_steps and lr for optimize_fusion_weights.

    Args:
        band_embeddings, query_embedding, w_init: same as optimize_fusion_weights
        num_steps_choices: candidate num_steps values (default [3, 5, 7])
        lr_choices:        candidate learning rates (default [0.005, 0.01, 0.02])
        lambda_m, k, grad_clip: forwarded to optimize_fusion_weights

    Returns:
        list of GridSearchEntry sorted by final_loss ascending
    """
    results: List[GridSearchEntry] = []

    for ns in num_steps_choices:
        for lr_val in lr_choices:
            opt_result = optimize_fusion_weights(
                band_embeddings=band_embeddings,
                query_embedding=query_embedding,
                w_init=w_init,
                num_steps=ns,
                lr=lr_val,
                lambda_m=lambda_m,
                k=k,
                grad_clip=grad_clip,
            )

            if len(opt_result.loss_history) >= 2:
                decrease_pct = (
                    (1.0 - opt_result.loss_history[-1] / opt_result.loss_history[0])
                    * 100.0
                )
            else:
                decrease_pct = 0.0

            results.append(GridSearchEntry(
                num_steps=ns,
                lr=lr_val,
                final_loss=opt_result.loss_history[-1],
                loss_decrease_pct=decrease_pct,
                elapsed_ms=opt_result.elapsed_ms,
                optimized_weights=opt_result.optimized_weights,
            ))

    # Sort by final_loss ascending (best first)
    results.sort(key=lambda e: e.final_loss)
    return results
