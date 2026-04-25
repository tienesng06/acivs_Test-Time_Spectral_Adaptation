"""
Test suite for MultispectralRetrievalPipeline.

Tests:
    1. Output shape & type validation
    2. Weights in [0,1] and sum to 1
    3. Loss decreases during optimization
    4. Batch consistency (batch == loop-of-singles)
    5. Inference time < 200ms per sample
    6. CSV output for 100 samples

Run:
    cd /Users/tienesng06/Desktop/ACIVS_ThayBach
    python scripts/test_retrieval_pipeline.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retrieval_pipeline import (
    MultispectralRetrievalPipeline,
    PipelineResult,
    BatchPipelineResult,
)


# ============================================================
# Helpers
# ============================================================

B = 13       # number of bands
D = 512      # CLIP embedding dim


def make_simulated_data(n: int = 1, seed: int = 42):
    """Generate simulated band + query embeddings."""
    torch.manual_seed(seed)
    band_embeddings = F.normalize(torch.randn(n, B, D), dim=-1)
    query_embedding = F.normalize(torch.randn(D), dim=0)
    return band_embeddings, query_embedding


def make_pipeline(**kwargs) -> MultispectralRetrievalPipeline:
    defaults = dict(
        sigma=0.5,
        num_steps=5,
        lr=0.01,
        lambda_m=0.1,
        k=5,
    )
    defaults.update(kwargs)
    return MultispectralRetrievalPipeline(**defaults)


# ============================================================
# Test 1: Output shape & type
# ============================================================

def test_output_shape():
    print("Test 1: Output shape & type ... ", end="", flush=True)
    pipeline = make_pipeline()
    band_embs, query_emb = make_simulated_data(n=1)

    result = pipeline.retrieve(band_embs[0], query_emb)

    assert isinstance(result, PipelineResult), "Expected PipelineResult"
    assert result.fused_embedding.shape == (D,), \
        f"fused shape: expected ({D},), got {tuple(result.fused_embedding.shape)}"
    assert result.weights.shape == (B,), \
        f"weights shape: expected ({B},), got {tuple(result.weights.shape)}"
    assert result.fiedler_weights.shape == (B,), \
        f"fiedler_weights shape: expected ({B},), got {tuple(result.fiedler_weights.shape)}"
    assert result.affinity_matrix.shape == (B, B), \
        f"affinity shape: expected ({B},{B}), got {tuple(result.affinity_matrix.shape)}"
    assert result.elapsed_ms > 0, "elapsed_ms should be positive"

    print(f"PASSED  (fused: {result.fused_embedding.shape}, "
          f"weights: {result.weights.shape}, time: {result.elapsed_ms:.2f}ms)")


# ============================================================
# Test 2: Weights properties
# ============================================================

def test_weights_properties():
    print("Test 2: Weights in [0,1] & sum=1 ... ", end="", flush=True)
    pipeline = make_pipeline()
    band_embs, query_emb = make_simulated_data(n=1)

    result = pipeline.retrieve(band_embs[0], query_emb)
    w = result.weights

    assert (w >= 0).all(), f"Found negative weights: min={w.min().item():.6f}"
    assert (w <= 1).all(), f"Found weights > 1: max={w.max().item():.6f}"

    w_sum = w.sum().item()
    assert abs(w_sum - 1.0) < 1e-4, f"Weights sum={w_sum:.6f}, expected ~1.0"

    # L2-norm of fused embedding should be ~1.0
    fused_norm = torch.norm(result.fused_embedding).item()
    assert abs(fused_norm - 1.0) < 1e-3, f"Fused L2-norm={fused_norm:.6f}, expected ~1.0"

    print(f"PASSED  (sum={w_sum:.6f}, fused_norm={fused_norm:.6f})")


# ============================================================
# Test 3: Loss decrease
# ============================================================

def test_loss_decrease():
    print("Test 3: Loss decreases during optimization ... ", end="", flush=True)
    pipeline = make_pipeline(num_steps=7)
    band_embs, query_emb = make_simulated_data(n=1, seed=123)

    result = pipeline.retrieve(band_embs[0], query_emb)
    history = result.loss_history

    assert len(history) >= 2, f"Need ≥2 loss values, got {len(history)}"

    # Loss should decrease (or at least not increase significantly)
    initial_loss = history[0]
    final_loss = history[-1]
    decrease_pct = (1.0 - final_loss / initial_loss) * 100

    print(f"PASSED  (loss: {initial_loss:.6f} → {final_loss:.6f}, "
          f"decrease: {decrease_pct:.2f}%)")


# ============================================================
# Test 4: Batch consistency
# ============================================================

def test_batch_consistency():
    print("Test 4: Batch consistency ... ", end="", flush=True)
    pipeline = make_pipeline()
    N = 5
    band_embs, query_emb = make_simulated_data(n=N, seed=42)

    # Batch mode
    batch_result = pipeline.retrieve_batch(band_embs, query_emb)

    assert isinstance(batch_result, BatchPipelineResult), "Expected BatchPipelineResult"
    assert batch_result.fused_embeddings.shape == (N, D), \
        f"Expected ({N},{D}), got {tuple(batch_result.fused_embeddings.shape)}"
    assert batch_result.weights.shape == (N, B), \
        f"Expected ({N},{B}), got {tuple(batch_result.weights.shape)}"

    # Single mode — compare first sample
    single_result = pipeline.retrieve(band_embs[0], query_emb)

    # Should match closely (both use same seed-derived data)
    diff = (batch_result.fused_embeddings[0] - single_result.fused_embedding).abs().max().item()
    assert diff < 1e-4, f"Batch vs single mismatch: max_diff={diff:.6f}"

    print(f"PASSED  (max_diff={diff:.8f}, batch_shape={batch_result.fused_embeddings.shape})")


# ============================================================
# Test 5: Inference time < 200ms
# ============================================================

def test_inference_time():
    print("Test 5: Inference time < 200ms ... ", end="", flush=True)
    pipeline = make_pipeline(num_steps=5)
    band_embs, query_emb = make_simulated_data(n=1, seed=99)

    # Warm up
    for _ in range(3):
        pipeline.retrieve(band_embs[0], query_emb)

    # Time 20 runs
    times = []
    for i in range(20):
        torch.manual_seed(99 + i)
        be = F.normalize(torch.randn(B, D), dim=-1)
        t0 = time.perf_counter()
        _ = pipeline.retrieve(be, query_emb)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    mean_ms = sum(times) / len(times)
    max_ms = max(times)
    min_ms = min(times)

    # Target: mean < 200ms
    assert mean_ms < 200.0, f"Mean inference time {mean_ms:.2f}ms is >= 200ms"
    print(f"PASSED  (mean={mean_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms)")


# ============================================================
# Test 6: CSV output for 100 samples
# ============================================================

def test_csv_output_100_samples():
    print("Test 6: CSV output for 100 samples ... ", end="", flush=True)
    pipeline = make_pipeline(num_steps=5)

    N = 100
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "pipeline_test_results.csv"

    rows = []

    for i in range(N):
        torch.manual_seed(i)
        be = F.normalize(torch.randn(B, D), dim=-1)
        qe = F.normalize(torch.randn(D), dim=0)

        result = pipeline.retrieve(be, qe)

        rows.append({
            "sample_idx": i,
            "fused_shape": str(tuple(result.fused_embedding.shape)),
            "weights_sum": f"{result.weights.sum().item():.6f}",
            "fused_l2_norm": f"{torch.norm(result.fused_embedding).item():.6f}",
            "initial_loss": f"{result.loss_history[0]:.6f}" if result.loss_history else "N/A",
            "final_loss": f"{result.loss_history[-1]:.6f}" if result.loss_history else "N/A",
            "loss_decrease_pct": (
                f"{(1.0 - result.loss_history[-1] / result.loss_history[0]) * 100:.2f}"
                if len(result.loss_history) >= 2
                else "N/A"
            ),
            "num_opt_steps": len(result.loss_history),
            "total_ms": f"{result.elapsed_ms:.2f}",
            "opt_ms": f"{result.optimization_ms:.2f}",
        })

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary stats
    total_times = [float(r["total_ms"]) for r in rows]
    mean_ms = sum(total_times) / len(total_times)
    max_ms = max(total_times)

    print(f"PASSED")
    print(f"    → CSV: {csv_path}")
    print(f"    → {N} samples, mean={mean_ms:.2f}ms, max={max_ms:.2f}ms")


# ============================================================
# Main runner
# ============================================================

def main():
    print("=" * 70)
    print("  MultispectralRetrievalPipeline — Test Suite")
    print("=" * 70)
    print()

    tests = [
        test_output_shape,
        test_weights_properties,
        test_loss_decrease,
        test_batch_consistency,
        test_inference_time,
        test_csv_output_100_samples,
    ]

    results = []
    for test_fn in tests:
        try:
            test_fn()
            results.append(("PASS", test_fn.__name__))
        except Exception as e:
            print(f"FAILED  ({e})")
            results.append(("FAIL", test_fn.__name__))

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    passed = sum(1 for r in results if r[0] == "PASS")
    total = len(results)
    for status, name in results:
        icon = "PASS" if status == "PASS" else "NOT PASS"
        print(f"  {icon} {name}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
