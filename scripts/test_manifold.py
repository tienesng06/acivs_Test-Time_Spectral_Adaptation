#!/usr/bin/env python3
"""
Test for src/models/manifold.py

Verifies:
  1. Import & shapes — build_knn_graph returns (B, k) indices
  2. manifold_consistency_loss returns differentiable scalar
  3. Loss decreases during gradient optimization (critical check)
  4. Loss < 1.0 after optimization
  5. k-NN preservation diagnostic
  6. Edge cases (k=1, k=B-1)
"""

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch


def main():
    print("=" * 60)
    print("Manifold Consistency Module — Test Suite")
    print("=" * 60)

    # =========================================================
    # Test 1: Import & build_knn_graph shape
    # =========================================================
    print("\n[1/6] Import & build_knn_graph shape...")
    from src.models.manifold import (
        build_knn_graph,
        build_knn_graph_with_distances,
        manifold_consistency_loss,
        compute_fused_embedding,
        check_knn_preservation,
    )
    print("  ✅ Import OK")

    torch.manual_seed(42)
    B, D = 13, 512  # 13 bands, 512-dim CLIP embeddings
    # L2-normalize to match real CLIP embeddings (unit vectors)
    band_embeddings = torch.randn(B, D)
    band_embeddings = torch.nn.functional.normalize(band_embeddings, dim=1)
    k = 5

    knn_indices = build_knn_graph(band_embeddings, k=k)
    print(f"  knn_indices shape: {tuple(knn_indices.shape)} (expect ({B}, {k}))")
    assert knn_indices.shape == (B, k), f"Shape mismatch: {knn_indices.shape}"
    assert knn_indices.min() >= 0, "Negative index"
    assert knn_indices.max() < B, f"Index {knn_indices.max()} >= B={B}"

    # No self-loops
    for i in range(B):
        assert i not in knn_indices[i].tolist(), f"Self-loop found for band {i}"

    print(f"  ✅ Shape ({B}, {k}), no self-loops, valid indices")

    # =========================================================
    # Test 2: build_knn_graph_with_distances
    # =========================================================
    print("\n[2/6] build_knn_graph_with_distances...")
    knn_idx, knn_dist = build_knn_graph_with_distances(band_embeddings, k=k)
    assert knn_idx.shape == (B, k)
    assert knn_dist.shape == (B, k)
    assert (knn_dist >= 0).all(), "Negative distances found"
    # Distances should be sorted ascending
    for i in range(B):
        diffs = knn_dist[i, 1:] - knn_dist[i, :-1]
        assert (diffs >= -1e-6).all(), f"Distances not sorted for band {i}"
    print(f"  ✅ Distances shape ({B}, {k}), all non-negative, sorted ascending")

    # =========================================================
    # Test 3: manifold_consistency_loss — shape & differentiability
    # =========================================================
    print("\n[3/6] manifold_consistency_loss basic check...")

    # Create a simple fused embedding
    weights = torch.softmax(torch.randn(B), dim=0)
    fused = compute_fused_embedding(band_embeddings, weights, normalize_output=False)
    print(f"  fused shape: {tuple(fused.shape)} (expect ({D},))")
    assert fused.shape == (D,)

    loss = manifold_consistency_loss(fused, band_embeddings, knn_indices, lambda_m=0.1)
    print(f"  loss = {loss.item():.6f}")
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"  ✅ Scalar loss, positive, value = {loss.item():.4f}")

    # =========================================================
    # Test 4: Loss decreases during optimization (CRITICAL)
    # =========================================================
    print("\n[4/6] Loss decreases with gradient optimization...")

    # Detach band_embeddings so they're fixed; only weights are learnable
    band_emb_fixed = band_embeddings.detach()
    knn_idx_fixed = build_knn_graph(band_emb_fixed, k=k)

    # Initialize learnable weights
    w = torch.randn(B, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=0.1)

    losses = []
    num_steps = 50

    for step in range(num_steps):
        optimizer.zero_grad()

        w_norm = torch.softmax(w, dim=0)
        fused_emb = compute_fused_embedding(
            band_emb_fixed, w_norm, normalize_output=False
        )

        loss = manifold_consistency_loss(
            fused_emb, band_emb_fixed, knn_idx_fixed, lambda_m=0.1
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"  Step  0: loss = {losses[0]:.6f}")
    print(f"  Step 24: loss = {losses[24]:.6f}")
    print(f"  Step 49: loss = {losses[49]:.6f}")

    # Check loss decreased
    assert losses[-1] < losses[0], (
        f"Loss did NOT decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
    )
    decrease_pct = (1 - losses[-1] / losses[0]) * 100
    print(f"  Decrease: {decrease_pct:.1f}%")
    print(f"  ✅ Loss decreased from {losses[0]:.4f} → {losses[-1]:.4f}")

    # =========================================================
    # Test 5: Final loss < 1.0
    # =========================================================
    print("\n[5/6] Final loss < 1.0 check...")
    final_loss = losses[-1]
    print(f"  Final loss = {final_loss:.6f}")
    assert final_loss < 1.0, f"Final loss {final_loss:.4f} >= 1.0"
    print(f"  ✅ Loss < 1.0")

    # =========================================================
    # Test 6: k-NN preservation diagnostic
    # =========================================================
    print("\n[6/6] k-NN preservation diagnostic...")

    final_w = torch.softmax(w, dim=0).detach()
    final_fused = compute_fused_embedding(
        band_emb_fixed, final_w, normalize_output=False
    )

    diagnostics = check_knn_preservation(
        band_emb_fixed, final_fused, knn_idx_fixed
    )
    print(f"  preservation_rate  = {diagnostics['preservation_rate']:.3f}")
    print(f"  mean_neighbor_dist = {diagnostics['mean_neighbor_dist']:.4f}")
    print(f"  ✅ Diagnostics computed successfully")

    # =========================================================
    # Edge cases
    # =========================================================
    print("\n[Bonus] Edge cases...")

    # k=1
    knn_k1 = build_knn_graph(band_embeddings, k=1)
    assert knn_k1.shape == (B, 1)
    loss_k1 = manifold_consistency_loss(fused, band_embeddings, knn_k1, lambda_m=0.1)
    print(f"  k=1: shape={tuple(knn_k1.shape)}, loss={loss_k1.item():.4f} ✅")

    # k=B-1 (maximum)
    knn_max = build_knn_graph(band_embeddings, k=B - 1)
    assert knn_max.shape == (B, B - 1)
    loss_max = manifold_consistency_loss(fused, band_embeddings, knn_max, lambda_m=0.1)
    print(f"  k={B-1}: shape={tuple(knn_max.shape)}, loss={loss_max.item():.4f} ✅")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED — Manifold consistency module is working!")
    print("=" * 60)
    print(f"\nLoss trajectory: {losses[0]:.4f} → {losses[-1]:.4f} ({decrease_pct:.1f}% decrease)")
    print(f"Verified: shapes, differentiability, loss decrease, final < 1.0")


if __name__ == "__main__":
    main()
