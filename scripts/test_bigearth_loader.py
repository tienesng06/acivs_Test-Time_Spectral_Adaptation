#!/usr/bin/env python3
"""
End-to-end test: Load mock BigEarthNet data through BigEarthNetDataset.

Verifies:
  1. No import errors
  2. Dataset scanning works (finds patches + parses labels)
  3. __getitem__ returns correct shapes
  4. Multi-hot labels are valid
  5. DataLoader batching works
  6. Class distribution is non-empty
  7. RGB extraction helper works
"""

import sys
import os

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DUMMY_ROOT = os.path.join(PROJECT_ROOT, "data", "Dummy_BigEarthS2")


def main():
    print("=" * 60)
    print("BigEarthNet DataLoader — End-to-End Test on Mock Data")
    print("=" * 60)

    # --- 1. Import ---
    print("\n[1/7] Importing module...")
    from src.datasets.bigearth_loader import (
        BigEarthNetDataset,
        BIGEARTH_BANDS,
        TOP10_CLASSES,
        bigearth_collate_fn,
        extract_rgb_bands,
        get_bigearth_text_queries,
    )
    print(f"     Import OK")
    print(f"     Bands: {len(BIGEARTH_BANDS)} → {BIGEARTH_BANDS[:4]}...")
    print(f"     Classes: {len(TOP10_CLASSES)}")

    # --- 2. Load all splits ---
    print("\n[2/7] Creating datasets (split=all, train, query, gallery)...")
    ds_all = BigEarthNetDataset(
        root=DUMMY_ROOT,
        split="all",
        max_samples=100,  # dummy has only 30
        query_size=5,
        gallery_size=10,
    )
    ds_train = BigEarthNetDataset(
        root=DUMMY_ROOT, split="train", max_samples=100, query_size=5, gallery_size=10
    )
    ds_query = BigEarthNetDataset(
        root=DUMMY_ROOT, split="query", max_samples=100, query_size=5, gallery_size=10
    )
    ds_gallery = BigEarthNetDataset(
        root=DUMMY_ROOT, split="gallery", max_samples=100, query_size=5, gallery_size=10
    )

    print(f"  all:     {len(ds_all):>4d} samples")
    print(f"  train:   {len(ds_train):>4d} samples")
    print(f"  query:   {len(ds_query):>4d} samples")
    print(f"  gallery: {len(ds_gallery):>4d} samples")
    total = len(ds_train) + len(ds_query) + len(ds_gallery)
    print(f"  sum:     {total:>4d} (should == {len(ds_all)})")
    assert total == len(ds_all), f"Split sizes don't add up: {total} != {len(ds_all)}"
    print(f"  Split sizes consistent")

    # --- 3. __getitem__ shape check ---
    print("\n[3/7] Checking __getitem__ output shapes...")
    sample = ds_all[0]
    img = sample["image"]
    lbl = sample["labels"]
    print(f"  image.shape  = {tuple(img.shape)}  (expect (12, 120, 120))")
    print(f"  labels.shape = {tuple(lbl.shape)}  (expect (10,))")
    print(f"  image dtype  = {img.dtype}")
    print(f"  labels dtype = {lbl.dtype}")
    print(f"  patch_name   = {sample['patch_name']}")
    print(f"  text         = {sample['text']}")
    print(f"  label_names  = {sample['label_names']}")

    assert img.shape == (12, 120, 120), f"Image shape mismatch: {img.shape}"
    assert lbl.shape == (10,), f"Label shape mismatch: {lbl.shape}"
    assert img.dtype.is_floating_point, f"Image should be float, got {img.dtype}"
    print(f"  Shapes correct")

    # --- 4. Pixel value range ---
    print("\n[4/7] Checking pixel value range...")
    print(f"  min={img.min().item():.4f}  max={img.max().item():.4f}")
    assert img.min() >= 0.0, f"Pixel min {img.min()} < 0"
    assert img.max() <= 1.0, f"Pixel max {img.max()} > 1"
    print(f"  Values in [0, 1]")

    # --- 5. Multi-hot labels ---
    print("\n[5/7] Checking multi-hot labels...")
    print(f"  labels vector = {lbl.tolist()}")
    assert lbl.sum() >= 1.0, "At least one label should be active"
    # Check all samples
    for i in range(len(ds_all)):
        s = ds_all[i]
        assert s["labels"].sum() >= 1.0, f"Sample {i} has no active labels"
        assert s["image"].shape == (12, 120, 120), f"Sample {i} shape mismatch"
    print(f"  All {len(ds_all)} samples have valid multi-hot labels and shapes")

    # --- 6. DataLoader batching ---
    print("\n[6/7] Testing DataLoader batching...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds_all,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=bigearth_collate_fn,
    )
    batch = next(iter(loader))
    B = batch["image"].shape[0]
    print(f"  batch image:  {tuple(batch['image'].shape)}")
    print(f"  batch labels: {tuple(batch['labels'].shape)}")
    print(f"  batch label_names lens: {[len(x) for x in batch['label_names']]}")
    assert batch["image"].shape == (B, 12, 120, 120)
    assert batch["labels"].shape == (B, 10)
    assert isinstance(batch["label_names"], list)
    assert all(isinstance(names, list) for names in batch["label_names"])
    print(f"  Batching works")

    # --- 7. Helpers ---
    print("\n[7/7] Testing helper functions...")

    # extract_rgb_bands
    rgb = extract_rgb_bands(batch["image"])
    print(f"  extract_rgb_bands: {tuple(rgb.shape)} (expect ({B}, 3, 120, 120))")
    assert rgb.shape == (B, 3, 120, 120)

    # Single image
    rgb_single = extract_rgb_bands(sample["image"])
    print(f"  extract_rgb_bands (single): {tuple(rgb_single.shape)} (expect (3, 120, 120))")
    assert rgb_single.shape == (3, 120, 120)

    # Class distribution
    dist = ds_all.get_class_distribution()
    print(f"  class_distribution: {dist}")
    assert sum(dist.values()) > 0, "Distribution should not be empty"

    # Text queries
    prompts = get_bigearth_text_queries()
    print(f"  text_queries: {len(prompts)} prompts")
    assert len(prompts) == 10

    print(f"  All helpers work")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED — BigEarthNet DataLoader is working!")
    print("=" * 60)
    print(f"\nReady to use with real BigEarthNet-S2 data.")
    print(f"Just change root to 'data/BigEarthNet-S2/' and set max_samples=100_000.")


if __name__ == "__main__":
    main()
