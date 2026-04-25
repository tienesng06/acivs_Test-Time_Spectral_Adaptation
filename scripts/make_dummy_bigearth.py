#!/usr/bin/env python3
"""
Generate a small dummy BigEarthNet-S2 dataset for testing the DataLoader.

Creates ~20 fake patches in data/Dummy_BigEarthS2/ with:
  - 12 Sentinel-2 band TIF files per patch (correct resolutions)
  - _labels_metadata.json with multi-label annotations

Usage:
    python scripts/make_dummy_bigearth.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# --- Try rasterio first, fallback to tifffile ---
try:
    import rasterio
    from rasterio.transform import from_bounds

    USE_RASTERIO = True
except ImportError:
    USE_RASTERIO = False

try:
    import tifffile
except ImportError:
    tifffile = None

if not USE_RASTERIO and tifffile is None:
    print("ERROR: Need either 'rasterio' or 'tifffile' to write TIF files.")
    print("  pip install rasterio   OR   pip install tifffile")
    sys.exit(1)


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "Dummy_BigEarthS2"

# Sentinel-2 bands (excluding B10)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
         "B8A", "B09", "B11", "B12"]

# Native pixel sizes per band
BAND_SIZES = {
    "B02": (120, 120), "B03": (120, 120), "B04": (120, 120), "B08": (120, 120),  # 10m
    "B05": (60, 60), "B06": (60, 60), "B07": (60, 60), "B8A": (60, 60),          # 20m
    "B11": (60, 60), "B12": (60, 60),                                              # 20m
    "B01": (20, 20), "B09": (20, 20),                                              # 60m
}

# CLC Level-3 labels (original BigEarthNet format)
CLC_LABELS_POOL = [
    ["Mixed forest"],
    ["Broad-leaved forest"],
    ["Coniferous forest"],
    ["Non-irrigated arable land"],
    ["Pastures"],
    ["Complex cultivation patterns"],
    ["Discontinuous urban fabric"],
    ["Natural grasslands"],
    ["Transitional woodland-shrub"],
    ["Land principally occupied by agriculture, with significant areas of natural vegetation"],
    # Multi-label examples
    ["Mixed forest", "Transitional woodland-shrub"],
    ["Pastures", "Non-irrigated arable land"],
    ["Broad-leaved forest", "Natural grasslands"],
    ["Discontinuous urban fabric", "Complex cultivation patterns"],
    ["Coniferous forest", "Pastures"],
]

# Fake tile names
TILE_NAMES = ["S2A_MSIL2A_20190613T101031", "S2B_MSIL2A_20200415T100559"]

NUM_PATCHES = 30  # total dummy patches to create
SEED = 42


# ============================================================
# TIF writing
# ============================================================

def write_tif_rasterio(path: Path, data: np.ndarray) -> None:
    """Write a 2D float32 array as a single-band GeoTIFF."""
    h, w = data.shape
    transform = from_bounds(0, 0, w * 10, h * 10, w, h)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def write_tif_tifffile(path: Path, data: np.ndarray) -> None:
    """Write a 2D float32 array as a plain TIFF (no georeference)."""
    tifffile.imwrite(str(path), data.astype(np.float32))


write_tif = write_tif_rasterio if USE_RASTERIO else write_tif_tifffile


# ============================================================
# Main
# ============================================================

def main() -> None:
    rng = np.random.default_rng(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    patch_count = 0
    for i in range(NUM_PATCHES):
        # Pick a tile
        tile = TILE_NAMES[i % len(TILE_NAMES)]
        tile_dir = OUTPUT_DIR / tile
        tile_dir.mkdir(exist_ok=True)

        # Patch name
        h_order = i // 10
        v_order = i % 10
        patch_name = f"{tile}_{h_order}_{v_order}"
        patch_dir = tile_dir / patch_name
        patch_dir.mkdir(exist_ok=True)

        # --- Write 12 band TIF files ---
        for band in BANDS:
            h, w = BAND_SIZES[band]
            # Simulate realistic Sentinel-2 L2A reflectance (0–10000 range)
            data = rng.integers(0, 8000, size=(h, w)).astype(np.float32)
            # Add some spatial structure (gradient + noise)
            gradient = np.linspace(500, 3000, h).reshape(-1, 1) * np.ones((1, w))
            data = data * 0.5 + gradient.astype(np.float32) * 0.5

            tif_path = patch_dir / f"{patch_name}_{band}.tif"
            write_tif(tif_path, data)

        # --- Write labels_metadata.json ---
        labels = CLC_LABELS_POOL[i % len(CLC_LABELS_POOL)]
        metadata = {
            "labels": labels,
            "acquisition_date": "2019-06-13",
            "coordinates": {"ulx": 0, "uly": 0, "lrx": 1200, "lry": 1200},
            "tile_source": tile,
            "projection": "EPSG:32632",
        }
        json_path = patch_dir / f"{patch_name}_labels_metadata.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        patch_count += 1

    print(f"\n✅ Created {patch_count} dummy patches in {OUTPUT_DIR}")
    print(f"   Tiles: {TILE_NAMES}")
    print(f"   Bands per patch: {len(BANDS)}")
    print(f"   Total files: {patch_count * (len(BANDS) + 1)} (TIFs + JSONs)")

    # Quick verification
    patch_dirs = list(OUTPUT_DIR.rglob("*_labels_metadata.json"))
    print(f"\n   Scan check: found {len(patch_dirs)} patches via rglob")


if __name__ == "__main__":
    main()
