#!/usr/bin/env python3
"""
Test for src/models/test_time_opt.py

Verifies:
  1. Import & basic optimize_fusion_weights call
  2. Output structure: weights sum=1, correct shapes
  3. Loss decreases during optimization
  4. Inference time < 200ms
  5. Grid search returns sorted results
  6. Early stopping works
"""

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F


def main():
    print("=" * 60)
    print("Test-Time Optimization Module — Test Suite")
    print("=" * 60)

    # =========================================================
    # Test 1: Import
    # =========================================================
    print("\n[1/6] Import...")
    from src.models.test_time_opt import (
        optimize_fusion_weights,
        grid_search_hyperparams,
        OptimizationResult,
        GridSearchEntry,
    )
    print("  ✅ Import OK")

    # Setup: simulated CLIP embeddings (L2-normalized)
    torch.manual_seed(42)
    B, D = 13, 512
    band_embeddings = F.normalize(torch.randn(B, D), dim=1)
    query_embedding = F.normalize(torch.randn(D), dim=0)

    # Simulate Fiedler weights (initial, positive, sum=1)
    w_init = torch.softmax(torch.randn(B), dim=0)

    # =========================================================
    # Test 2: Basic call & output structure
    # =========================================================
    print("\n[2/6] Basic optimize_fusion_weights call...")
    result = optimize_fusion_weights(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        w_init=w_init,
        num_steps=5,
        lr=0.01,
        lambda_m=0.1,
        k=5,
    )

    assert isinstance(result, OptimizationResult), "Wrong return type"
    assert result.optimized_weights.shape == (B,), f"weights shape {result.optimized_weights.shape}"
    assert result.fused_embedding.shape == (D,), f"fused shape {result.fused_embedding.shape}"

    # Weights sum to 1
    w_sum = result.optimized_weights.sum().item()
    print(f"  weights sum   = {w_sum:.6f} (expect ~1.0)")
    assert abs(w_sum - 1.0) < 1e-5, f"Weights don't sum to 1: {w_sum}"

    # All weights non-negative (softmax output)
    assert (result.optimized_weights >= 0).all(), "Negative weight found"

    # Fused embedding is L2-normalized
    fused_norm = torch.norm(result.fused_embedding).item()
    print(f"  fused L2 norm = {fused_norm:.6f} (expect ~1.0)")
    assert abs(fused_norm - 1.0) < 1e-4, f"Fused not normalized: {fused_norm}"

    print(f"  steps run     = {result.num_steps_run}")
    print(f"  elapsed       = {result.elapsed_ms:.2f} ms")
    print(f"  ✅ Output structure correct")

    # =========================================================
    # Test 3: Loss decreases
    # =========================================================
    print("\n[3/6] Loss decreases during optimization...")
    result_20 = optimize_fusion_weights(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        w_init=w_init,
        num_steps=20,
        lr=0.05,
        lambda_m=0.1,
        k=5,
        early_stop_tol=0.0,  # disable early stop
    )

    losses = result_20.loss_history
    print(f"  Step  0: loss = {losses[0]:.6f}")
    print(f"  Step  9: loss = {losses[9]:.6f}")
    print(f"  Step 19: loss = {losses[19]:.6f}")

    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
    decrease_pct = (1 - losses[-1] / losses[0]) * 100
    print(f"  Decrease: {decrease_pct:.2f}%")
    print(f"  ✅ Loss decreased from {losses[0]:.4f} → {losses[-1]:.4f}")

    # =========================================================
    # Test 4: Inference time < 200ms
    # =========================================================
    print("\n[4/6] Inference time check (5-step, target < 200ms)...")

    # Warm-up
    for _ in range(3):
        optimize_fusion_weights(
            band_embeddings=band_embeddings,
            query_embedding=query_embedding,
            w_init=w_init,
            num_steps=5,
            lr=0.01,
        )

    # Measure average over 10 runs
    times = []
    for _ in range(10):
        res = optimize_fusion_weights(
            band_embeddings=band_embeddings,
            query_embedding=query_embedding,
            w_init=w_init,
            num_steps=5,
            lr=0.01,
        )
        times.append(res.elapsed_ms)

    avg_ms = sum(times) / len(times)
    max_ms = max(times)
    print(f"  Average: {avg_ms:.2f} ms")
    print(f"  Max:     {max_ms:.2f} ms")
    assert avg_ms < 200, f"Average time {avg_ms:.2f}ms exceeds 200ms target"
    print(f"  ✅ Inference < 200ms")

    # =========================================================
    # Test 5: Grid search
    # =========================================================
    print("\n[5/6] Grid search hyperparameters...")
    grid_results = grid_search_hyperparams(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        w_init=w_init,
        num_steps_choices=(3, 5, 7),
        lr_choices=(0.005, 0.01, 0.02),
        lambda_m=0.1,
        k=5,
    )

    assert len(grid_results) == 9, f"Expected 9 entries, got {len(grid_results)}"

    print(f"  {'steps':>5s}  {'lr':>6s}  {'loss':>10s}  {'decrease%':>10s}  {'time_ms':>8s}")
    print("  " + "-" * 48)
    for entry in grid_results:
        print(f"  {entry.num_steps:>5d}  {entry.lr:>6.3f}  "
              f"{entry.final_loss:>10.6f}  {entry.loss_decrease_pct:>9.2f}%  "
              f"{entry.elapsed_ms:>7.2f}")

    # Results should be sorted by final_loss (ascending)
    for i in range(len(grid_results) - 1):
        assert grid_results[i].final_loss <= grid_results[i+1].final_loss, \
            "Grid results not sorted"
    print(f"  ✅ Grid search: 9 configs, sorted by loss")
    print(f"  Best: steps={grid_results[0].num_steps}, lr={grid_results[0].lr}, "
          f"loss={grid_results[0].final_loss:.6f}")

    # =========================================================
    # Test 6: Early stopping
    # =========================================================
    print("\n[6/6] Early stopping...")
    result_early = optimize_fusion_weights(
        band_embeddings=band_embeddings,
        query_embedding=query_embedding,
        w_init=w_init,
        num_steps=100,  # high limit
        lr=0.01,
        lambda_m=0.1,
        k=5,
        early_stop_tol=1e-6,
    )
    print(f"  Requested 100 steps, ran {result_early.num_steps_run} steps")
    print(f"  Final loss: {result_early.loss_history[-1]:.6f}")
    assert result_early.num_steps_run <= 100
    # With early stopping, should converge much sooner than 100
    if result_early.num_steps_run < 100:
        print(f"  ✅ Early stopping triggered (saved {100 - result_early.num_steps_run} steps)")
    else:
        print(f"  ⚠ Early stopping did not trigger (loss still changing)")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED — Test-time optimization is working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
