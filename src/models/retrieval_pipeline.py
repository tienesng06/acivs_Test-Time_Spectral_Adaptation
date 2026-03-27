"""
End-to-end multispectral retrieval pipeline.

Integrates all 6 pipeline stages into a single class:
    1. Receive per-band CLIP embeddings  (B, D)
    2. Receive query embedding            (D,)
    3. Compute affinity graph             → A_norm (B, B)
    4. Compute Fiedler magnitude weights  → w_init (B,)
    5. Optimize fusion weights via Adam   → w_opt  (B,)
    6. Weighted fusion + L2-normalize     → fused  (D,)

Usage:
    pipeline = MultispectralRetrievalPipeline()
    result   = pipeline.retrieve(band_embeddings, query_embedding)
    # result.fused_embedding  → (D,)
    # result.weights          → (B,)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.affinity_graph import compute_affinity_graph
from src.models.fiedler import compute_fiedler_magnitude_weights
from src.models.manifold import compute_fused_embedding
from src.models.test_time_opt import optimize_fusion_weights, OptimizationResult


# ============================================================
# Result container
# ============================================================

@dataclass
class PipelineResult:
    """Output of a single retrieve() call."""

    fused_embedding: torch.Tensor          # (D,) L2-normalized
    weights: torch.Tensor                  # (B,) softmax-normalized, sum=1
    fiedler_weights: torch.Tensor          # (B,) initial Fiedler weights
    affinity_matrix: torch.Tensor          # (B, B) normalized affinity
    loss_history: List[float] = field(default_factory=list)
    elapsed_ms: float = 0.0               # total wall-clock time
    optimization_ms: float = 0.0          # time in test-time opt only


@dataclass
class BatchPipelineResult:
    """Output of retrieve_batch()."""

    fused_embeddings: torch.Tensor         # (N, D)
    weights: torch.Tensor                  # (N, B)
    per_sample_ms: List[float] = field(default_factory=list)
    mean_ms: float = 0.0
    total_ms: float = 0.0


# ============================================================
# Pipeline class
# ============================================================

class MultispectralRetrievalPipeline:
    """
    End-to-end spectral fusion retrieval pipeline.

    Combines affinity graph, Fiedler initialization, test-time
    optimization, and weighted fusion into a single callable.
    """

    def __init__(
        self,
        *,
        # Affinity graph params
        sigma: float = 0.5,
        normalize_inputs: bool = True,
        # Fiedler params
        normalized_laplacian: bool = True,
        # Test-time optimization params
        num_steps: int = 5,
        lr: float = 0.01,
        lambda_m: float = 0.1,
        k: int = 5,
        grad_clip: Optional[float] = 1.0,
        early_stop_tol: float = 1e-6,
        # Output params
        l2_normalize_output: bool = True,
    ):
        self.sigma = sigma
        self.normalize_inputs = normalize_inputs
        self.normalized_laplacian = normalized_laplacian
        self.num_steps = num_steps
        self.lr = lr
        self.lambda_m = lambda_m
        self.k = k
        self.grad_clip = grad_clip
        self.early_stop_tol = early_stop_tol
        self.l2_normalize_output = l2_normalize_output

    # --------------------------------------------------------
    # Core: single-sample retrieve
    # --------------------------------------------------------

    def retrieve(
        self,
        band_embeddings: torch.Tensor,
        query_embedding: torch.Tensor,
        *,
        return_details: bool = True,
    ) -> PipelineResult:
        """
        Run 6-step pipeline on pre-encoded embeddings.

        Args:
            band_embeddings: (B, D) per-band CLIP embeddings
            query_embedding: (D,) or (1, D) query embedding

        Returns:
            PipelineResult with fused embedding, weights, timing, etc.
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

        # ------ Step 3: Compute affinity graph ------
        A_norm = compute_affinity_graph(
            band_embeddings=band_embeddings,
            query_embedding=query_embedding,
            sigma=self.sigma,
            normalize_inputs=self.normalize_inputs,
            return_details=False,
        )

        # ------ Step 4: Compute Fiedler init weights ------
        w_fiedler = compute_fiedler_magnitude_weights(
            A=A_norm,
            normalized=self.normalized_laplacian,
        )

        # ------ Step 5: Test-time optimization ------
        t_opt_start = time.perf_counter()

        opt_result: OptimizationResult = optimize_fusion_weights(
            band_embeddings=band_embeddings,
            query_embedding=query_embedding,
            w_init=w_fiedler,
            num_steps=self.num_steps,
            lr=self.lr,
            lambda_m=self.lambda_m,
            k=self.k,
            grad_clip=self.grad_clip,
            early_stop_tol=self.early_stop_tol,
            normalize_output=self.l2_normalize_output,
        )

        t_opt_end = time.perf_counter()
        optimization_ms = (t_opt_end - t_opt_start) * 1000.0

        # ------ Step 6: fused embedding from OptimizationResult ------
        # opt_result already contains the softmax-normalized weights
        # and the L2-normed fused embedding
        fused_embedding = opt_result.fused_embedding
        optimized_weights = opt_result.optimized_weights

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        return PipelineResult(
            fused_embedding=fused_embedding,
            weights=optimized_weights,
            fiedler_weights=w_fiedler.detach(),
            affinity_matrix=A_norm.detach(),
            loss_history=opt_result.loss_history,
            elapsed_ms=total_ms,
            optimization_ms=optimization_ms,
        )

    # --------------------------------------------------------
    # Batch retrieve
    # --------------------------------------------------------

    def retrieve_batch(
        self,
        batch_band_embeddings: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> BatchPipelineResult:
        """
        Run pipeline on a batch of N samples.

        Args:
            batch_band_embeddings: (N, B, D) per-sample per-band embeddings
            query_embedding: (D,) query embedding (shared across batch)

        Returns:
            BatchPipelineResult
        """
        if batch_band_embeddings.ndim != 3:
            raise ValueError(
                f"batch_band_embeddings must be 3D (N, B, D), "
                f"got {tuple(batch_band_embeddings.shape)}"
            )

        N, B, D = batch_band_embeddings.shape
        fused_list = []
        weights_list = []
        times_list = []

        t_batch_start = time.perf_counter()

        for i in range(N):
            result = self.retrieve(
                band_embeddings=batch_band_embeddings[i],
                query_embedding=query_embedding,
            )
            fused_list.append(result.fused_embedding)
            weights_list.append(result.weights)
            times_list.append(result.elapsed_ms)

        t_batch_end = time.perf_counter()
        total_ms = (t_batch_end - t_batch_start) * 1000.0
        mean_ms = total_ms / max(N, 1)

        return BatchPipelineResult(
            fused_embeddings=torch.stack(fused_list, dim=0),   # (N, D)
            weights=torch.stack(weights_list, dim=0),          # (N, B)
            per_sample_ms=times_list,
            mean_ms=mean_ms,
            total_ms=total_ms,
        )

    # --------------------------------------------------------
    # Full E2E: raw image + text → fused embedding
    # --------------------------------------------------------

    def retrieve_from_raw(
        self,
        image_13band: torch.Tensor,
        query_text: str,
        clip_model,
        clip_tokenize_fn=None,
        device: Optional[torch.device] = None,
    ) -> PipelineResult:
        """
        Complete end-to-end: encode raw 13-band image + text query,
        then run 6-step pipeline.

        Args:
            image_13band: (13, H, W) raw multispectral image
            query_text: text query string
            clip_model: loaded CLIP model
            clip_tokenize_fn: CLIP tokenizer (e.g. clip.tokenize)
            device: torch.device for inference

        Returns:
            PipelineResult
        """
        # Lazy import to avoid hard dependency when not needed
        from src.models.per_band_encoder import (
            encode_multispectral_bands,
            get_device,
        )

        if device is None:
            device = get_device()

        # Step 1: encode 13 bands → (13, D)
        band_embeddings = encode_multispectral_bands(
            image_13band=image_13band,
            clip_model=clip_model,
            device=device,
        )

        # Step 2: encode query text → (D,)
        if clip_tokenize_fn is None:
            try:
                import clip
                clip_tokenize_fn = clip.tokenize
            except ImportError:
                raise ImportError(
                    "clip package required for text encoding. "
                    "Install with: pip install git+https://github.com/openai/CLIP.git"
                )

        with torch.no_grad():
            text_tokens = clip_tokenize_fn([query_text]).to(device)
            query_embedding = clip_model.encode_text(text_tokens).float()
            query_embedding = F.normalize(query_embedding, dim=-1)
            query_embedding = query_embedding.squeeze(0).cpu()

        # Steps 3-6: pipeline
        return self.retrieve(
            band_embeddings=band_embeddings,
            query_embedding=query_embedding,
        )

    # --------------------------------------------------------
    # String representation
    # --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MultispectralRetrievalPipeline("
            f"sigma={self.sigma}, "
            f"num_steps={self.num_steps}, "
            f"lr={self.lr}, "
            f"lambda_m={self.lambda_m}, "
            f"k={self.k})"
        )
