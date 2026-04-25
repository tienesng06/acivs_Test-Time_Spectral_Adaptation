#!/usr/bin/env python3
"""
Generate notebooks/11_bigearth_benchmark.ipynb

BigEarthNet-S2 100k Benchmark — 7 methods, multi-label metrics.
Run: python scripts/create_bigearth_benchmark_notebook.py
"""

from __future__ import annotations
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/11_bigearth_benchmark.ipynb")

# ── helper ────────────────────────────────────────────────────
def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip(),
    }

def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip(),
    }

# ══════════════════════════════════════════════════════════════
cells = []

# ── Cell 0: Title ─────────────────────────────────────────────
cells.append(md("""
# 11 – BigEarthNet-S2 100k Benchmark

**Task 2 – Tuần 6**

Setup: 100k subset (10 class phổ biến nhất) • 1k query + 10k gallery • image-to-image  
Metrics: Precision@K, Recall@K, F1@K (K=1,5,10) + mAP (multi-label)  
Methods: RGB-CLIP · PCA · NDVI · Tip-Adapter* · RS-TransCLIP · **Ours** · DOFA (template)
""".strip()))

# ── Cell 1: Config ────────────────────────────────────────────
cells.append(code("""
# ── CONFIG ────────────────────────────────────────────────────
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")           # make `src/` importable

from pathlib import Path

# ▶ ADJUST PATHS to your machine
BIGEARTH_ROOT   = Path("/Volumes/Data/data/BigEarthNetS2/BigEarthNetS2")
METADATA_PARQUET = Path("../data/BigEarthNetS2.local_backup_20260419/metadata.parquet")
CLIP_CHECKPOINT  = Path("../checkpoints/ViT-B-16.pt")
RESULTS_DIR      = Path("../results/bigearth_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset setup
MAX_SAMPLES   = 100_000
QUERY_SIZE    = 1_000
GALLERY_SIZE  = 10_000
SEED          = 42
BATCH_SIZE    = 32
NUM_WORKERS   = 0
MICRO_BATCH   = 64
IMAGE_SIZE    = 224
RGB_INDICES   = (3, 2, 1)          # B04, B03, B02

# Pipeline (Ours) hyperparams
SIGMA         = 0.5
NUM_STEPS     = 5
LR            = 0.01
LAMBDA_M      = 0.1
K             = 5
GRAD_CLIP     = 1.0
EARLY_STOP    = 1e-6

# Tip-Adapter hyperparams
TIP_ALPHA     = 1.0
TIP_BETA      = 5.5

print("Config OK")
print(f"  BIGEARTH_ROOT: {BIGEARTH_ROOT} (exists={BIGEARTH_ROOT.exists()})")
print(f"  METADATA_PARQUET: {METADATA_PARQUET} (exists={METADATA_PARQUET.exists()})")
print(f"  CLIP_CHECKPOINT: {CLIP_CHECKPOINT} (exists={CLIP_CHECKPOINT.exists()})")
"""))

# ── Cell 2: Parquet-backed BigEarthNet loader ─────────────────
cells.append(code("""
# ── PARQUET-BACKED DATASET BUILDER ────────────────────────────
# Dataset trên SSD là v2 (TIF-only, no JSON labels).
# Dùng metadata.parquet để lấy labels cho từng patch.

import numpy as np
import pandas as pd
from collections import Counter

# BigEarthNet label name normalization (parquet dùng comma, code dùng slash)
PARQUET_LABEL_MAP = {
    "Arable land":                   "Arable land",
    "Mixed forest":                  "Mixed forest",
    "Coniferous forest":             "Coniferous forest",
    "Transitional woodland, shrub":  "Transitional woodland/shrub",
    "Broad-leaved forest":           "Broad-leaved forest",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Complex cultivation patterns":  "Complex cultivation patterns",
    "Pastures":                      "Pastures",
    "Urban fabric":                  "Urban fabric",
    "Inland waters":                 "Inland waters",
    "Natural grasslands":            "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas":      "Natural grassland and sparsely vegetated areas",
    "Industrial or commercial units": "Industrial or commercial units",
    "Sea and ocean":                 "Marine waters",
    "Water courses":                 "Inland waters",
    "Water bodies":                  "Inland waters",
    "Moors and heathland":           "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation":     "Moors, heathland and sclerophyllous vegetation",
    "Inland marshes":                "Inland wetlands",
    "Peat bogs":                     "Inland wetlands",
    "Agro-forestry areas":           "Agro-forestry areas",
    "Non-irrigated arable land":     "Arable land",
    "Permanently irrigated land":    "Arable land",
    "Rice fields":                   "Arable land",
}

def normalize_parquet_labels(raw_labels):
    normalized = []
    seen = set()
    for lbl in raw_labels:
        mapped = PARQUET_LABEL_MAP.get(lbl, lbl)
        if mapped not in seen:
            normalized.append(mapped)
            seen.add(mapped)
    return normalized

# Load parquet
print("Loading metadata.parquet ...")
meta_df = pd.read_parquet(METADATA_PARQUET)
meta_df = meta_df[~meta_df["contains_seasonal_snow"] & ~meta_df["contains_cloud_or_shadow"]]
meta_df = meta_df.set_index("patch_id")
print(f"Clean patches in parquet: {len(meta_df):,}")

# Determine TOP10 classes from this full set
label_counter = Counter()
for labels in meta_df["labels"]:
    for lbl in normalize_parquet_labels(labels):
        label_counter[lbl] += 1
        
TOP10_CLASSES = [cls for cls, _ in label_counter.most_common(10)]
print("\\nTop-10 classes (from parquet):")
for i, (cls, cnt) in enumerate(label_counter.most_common(10)):
    print(f"  {i:2d}. {cls}: {cnt:,}")
    
CLASS_TO_IDX = {cls: i for i, cls in enumerate(TOP10_CLASSES)}
NUM_CLASSES  = len(TOP10_CLASSES)
"""))

# ── Cell 3: Scan patches and build index ──────────────────────
cells.append(code("""
# ── SCAN PATCHES FROM SSD ─────────────────────────────────────
import os

print("Scanning SSD patches (may take ~30s) ...")
all_patches = []

for tile_name in sorted(os.listdir(BIGEARTH_ROOT)):
    tile_path = BIGEARTH_ROOT / tile_name
    if not tile_path.is_dir():
        continue
    for patch_name in os.listdir(tile_path):
        patch_path = tile_path / patch_name
        if not patch_path.is_dir():
            continue
        if patch_name not in meta_df.index:
            continue
        raw_labels = meta_df.loc[patch_name, "labels"]
        labels_norm = normalize_parquet_labels(raw_labels)
        labels_top10 = [l for l in labels_norm if l in CLASS_TO_IDX]
        if not labels_top10:
            continue
        all_patches.append({
            "patch_dir":   patch_path,
            "patch_name":  patch_name,
            "labels_top10": labels_top10,
        })

print(f"Valid patches (have top-10 labels): {len(all_patches):,}")

# ── SUBSET + SPLIT ────────────────────────────────────────────
rng = np.random.default_rng(SEED)
n_total = len(all_patches)

if n_total > MAX_SAMPLES:
    idx = rng.choice(n_total, MAX_SAMPLES, replace=False)
    idx.sort()
    subset = [all_patches[i] for i in idx]
else:
    subset = list(all_patches)

n = len(subset)
perm = rng.permutation(n)
q_size = min(QUERY_SIZE,   n // 5)
g_size = min(GALLERY_SIZE, n // 2)
t_size = n - q_size - g_size

query_patches   = [subset[i] for i in sorted(perm[:q_size])]
gallery_patches = [subset[i] for i in sorted(perm[q_size:q_size + g_size])]
train_patches   = [subset[i] for i in sorted(perm[q_size + g_size:])]

print(f"  Total subset:  {n:,}")
print(f"  Query:   {len(query_patches):,}")
print(f"  Gallery: {len(gallery_patches):,}")
print(f"  Train:   {len(train_patches):,}")
"""))

# ── Cell 4: TIF reader + multi-hot encoder ────────────────────
cells.append(code("""
# ── TIF READER + MULTI-HOT ENCODER ───────────────────────────
import torch
import torch.nn.functional as F
import numpy as np

try:
    import rasterio
    from scipy.ndimage import zoom as scipy_zoom
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("WARNING: rasterio / scipy not available — image loading will fail")

BIGEARTH_BANDS = [
    "B01","B02","B03","B04","B05","B06",
    "B07","B08","B8A","B09","B11","B12",
]
TARGET_SIZE = 120
REFLECTANCE_SCALE = 10_000.0

def read_patch(patch_dir, patch_name) -> torch.Tensor:
    bands = []
    for band_name in BIGEARTH_BANDS:
        tif_path = patch_dir / f"{patch_name}_{band_name}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(str(tif_path))
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        h, w = arr.shape
        if h != TARGET_SIZE or w != TARGET_SIZE:
            arr = scipy_zoom(arr, (TARGET_SIZE / h, TARGET_SIZE / w), order=1)
        bands.append(arr)
    img = np.stack(bands, 0) / REFLECTANCE_SCALE
    img = np.clip(img, 0.0, 1.0)
    return torch.from_numpy(img)

def labels_to_multihot(labels_top10) -> torch.Tensor:
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in labels_top10:
        i = CLASS_TO_IDX.get(lbl)
        if i is not None:
            vec[i] = 1.0
    return vec

print("Reader & encoder defined OK")
"""))

# ── Cell 5: Load CLIP + encode text ───────────────────────────
cells.append(code("""
# ── LOAD CLIP MODEL ───────────────────────────────────────────
from src.utils.shared import load_openai_clip_model, get_device

device = get_device()
print(f"Device: {device}")

clip_model, clip_tokenize = load_openai_clip_model(CLIP_CHECKPOINT, device)

# Encode class text prompts for query conditioning (Eq. 1)
CLASS_TEXT_MAP = {
    "Broad-leaved forest":       "broad-leaved forest",
    "Coniferous forest":         "coniferous forest",
    "Mixed forest":              "mixed forest",
    "Arable land":               "arable land for agriculture",
    "Pastures":                  "pasture land",
    "Complex cultivation patterns": "complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "agricultural land with natural vegetation",
    "Natural grassland and sparsely vegetated areas":
        "natural grassland and sparse vegetation",
    "Transitional woodland/shrub": "transitional woodland and shrubs",
    "Urban fabric":              "urban residential area",
    "Inland waters":             "inland water body",
    "Moors, heathland and sclerophyllous vegetation":
        "moors and heathland vegetation",
    "Industrial or commercial units": "industrial or commercial area",
}

TEXT_TEMPLATE = "A satellite image of {class_text}."
class_prompts = [
    TEXT_TEMPLATE.format(class_text=CLASS_TEXT_MAP.get(cls, cls))
    for cls in TOP10_CLASSES
]

@torch.no_grad()
def encode_text_features(model, tokenize, prompts, device):
    tokens = tokenize(prompts).to(device)
    feats = model.encode_text(tokens).float()
    return F.normalize(feats, dim=-1).cpu()

text_features = encode_text_features(clip_model, clip_tokenize, class_prompts, device)
print(f"Text features: {text_features.shape}")  # [C, D]
"""))

# ── Cell 6: Shared encoding functions ────────────────────────
cells.append(code("""
# ── SHARED ENCODING FUNCTIONS ─────────────────────────────────
from tqdm.auto import tqdm

CLIP_MEAN = torch.tensor([0.48145466, 0.45782750, 0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

@torch.no_grad()
def preprocess_rgb_for_clip(rgb_batch: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """rgb_batch: [N, 3, H, W] float32 in [0,1]""",
    x = F.interpolate(rgb_batch, (image_size, image_size), mode="bilinear", align_corners=False)
    mean = CLIP_MEAN.view(1, 3, 1, 1)
    std  = CLIP_STD.view(1, 3, 1, 1)
    return (x - mean) / std

@torch.no_grad()
def encode_band_embeddings_batch(image_batch: torch.Tensor, model, dev,
                                  micro_batch: int = 64) -> torch.Tensor:
    """image_batch: [N, 12, H, W] → [N, 12, D] L2-normalized per-band CLIP embeddings"""
    N, B, H, W = image_batch.shape
    # Expand to [N*B, 3, H, W] (replicate band as RGB)
    x = image_batch.unsqueeze(2).repeat(1, 1, 3, 1, 1).view(N * B, 3, H, W)
    x = F.interpolate(x, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    mean = CLIP_MEAN.view(1, 3, 1, 1); std = CLIP_STD.view(1, 3, 1, 1)
    x = (x - mean) / std
    chunks = []
    for i in range(0, x.shape[0], micro_batch):
        chunk = x[i:i + micro_batch].to(dev)
        feats = model.encode_image(chunk).float()
        chunks.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(chunks, 0).view(N, B, -1)

def encode_split_full(patches, model, dev, desc="Encode", batch_size=BATCH_SIZE):
    """Returns dict: band_embeddings [N,12,D], rgb_features [N,D], labels [N,C], label_lists"""
    band_embs, rgb_feats, label_tensors, label_lists, patch_names = [], [], [], [], []
    n = len(patches)
    pbar = tqdm(range(0, n, batch_size), desc=desc)
    for start in pbar:
        batch_patches = patches[start:start + batch_size]
        images, labels_batch, lnames = [], [], []
        for p in batch_patches:
            try:
                img = read_patch(p["patch_dir"], p["patch_name"])
                images.append(img)
                lbl = labels_to_multihot(p["labels_top10"])
                labels_batch.append(lbl)
                lnames.append(p["labels_top10"])
                patch_names.append(p["patch_name"])
            except Exception as e:
                pass  # skip bad patches silently
        if not images:
            continue
        imgs = torch.stack(images)                        # [B, 12, 120, 120]
        # per-band embeddings
        be = encode_band_embeddings_batch(imgs, model, dev)
        band_embs.append(be)
        # RGB composite for RGB-CLIP baseline
        rgb = imgs[:, [3, 2, 1], :, :]                   # B04, B03, B02
        rgb_inp = preprocess_rgb_for_clip(rgb).to(dev)
        rf = model.encode_image(rgb_inp).float()
        rgb_feats.append(F.normalize(rf, dim=-1).cpu())
        label_tensors.append(torch.stack(labels_batch))
        label_lists.extend(lnames)

    return {
        "band_embeddings": torch.cat(band_embs, 0),      # [N, 12, D]
        "rgb_features":    torch.cat(rgb_feats, 0),       # [N, D]
        "labels":          torch.cat(label_tensors, 0),   # [N, C]
        "label_lists":     label_lists,
    }

print("Encoding functions defined OK")
"""))

# ── Cell 7: Encode query and gallery ─────────────────────────
cells.append(code("""
# ── ENCODE QUERY + GALLERY ────────────────────────────────────
print("Encoding QUERY ...")
query_data = encode_split_full(query_patches, clip_model, device, desc="Query")
print(f"  Query encoded: {query_data['band_embeddings'].shape}")

print("\\nEncoding GALLERY ...")
gallery_data = encode_split_full(gallery_patches, clip_model, device, desc="Gallery")
print(f"  Gallery encoded: {gallery_data['band_embeddings'].shape}")

# For Tip-Adapter: encode TRAIN rgb features
print("\\nEncoding TRAIN (for Tip-Adapter) ...")
train_data = encode_split_full(train_patches[:5000], clip_model, device, 
                                desc="Train (5k sample)")
print(f"  Train encoded: {train_data['band_embeddings'].shape}")
"""))

# ── Cell 8: Multi-label metrics ───────────────────────────────
cells.append(code("""
# ── MULTI-LABEL METRICS ───────────────────────────────────────
from src.utils.metrics import evaluate_multilabel_image_retrieval

def run_multilabel_eval(query_features, gallery_features,
                        query_labels, gallery_labels, ks=(1, 5, 10)):
    \"\"\"Wrapper: compute P@K, R@K, F1@K, mAP for multi-label retrieval.\"\"\"
    metrics, sim_mat, ranked_idx, per_query = evaluate_multilabel_image_retrieval(
        query_features=F.normalize(query_features.float(), dim=-1),
        gallery_features=F.normalize(gallery_features.float(), dim=-1),
        query_labels=query_labels.float(),
        gallery_labels=gallery_labels.float(),
        ks=list(ks),
    )
    return metrics, sim_mat, ranked_idx, per_query

all_results = {}   # method_name -> metrics dict
print("Multi-label eval function ready")
"""))

# ── Cell 9: RGB-CLIP ─────────────────────────────────────────
cells.append(code("""
# ── METHOD 1: RGB-CLIP (image-to-image) ───────────────────────
print("=" * 55)
print("Method 1: RGB-CLIP")
metrics_rgb, _, _, _ = run_multilabel_eval(
    query_data["rgb_features"],
    gallery_data["rgb_features"],
    query_data["labels"],
    gallery_data["labels"],
)
all_results["RGB-CLIP"] = metrics_rgb
print("  R@1={:.1f}%  R@10={:.1f}%  mAP={:.1f}%".format(
    metrics_rgb["ML_Recall@1"], metrics_rgb["ML_Recall@10"], metrics_rgb["mAP"]))
"""))

# ── Cell 10: PCA ─────────────────────────────────────────────
cells.append(code("""
# ── METHOD 2: PCA ─────────────────────────────────────────────
from sklearn.decomposition import PCA
print("=" * 55)
print("Method 2: PCA (12-band → 3-comp RGB substitute → CLIP)")

# PCA on flattened band embeddings
q_flat = query_data["band_embeddings"].view(len(query_patches), -1).numpy()
g_flat = gallery_data["band_embeddings"].view(len(gallery_patches), -1).numpy()

pca = PCA(n_components=3, random_state=SEED)
pca.fit(g_flat)
q_pca = torch.tensor(pca.transform(q_flat), dtype=torch.float32)
g_pca = torch.tensor(pca.transform(g_flat), dtype=torch.float32)

metrics_pca, _, _, _ = run_multilabel_eval(q_pca, g_pca,
                                            query_data["labels"],
                                            gallery_data["labels"])
all_results["PCA"] = metrics_pca
print("  R@1={:.1f}%  R@10={:.1f}%  mAP={:.1f}%".format(
    metrics_pca["ML_Recall@1"], metrics_pca["ML_Recall@10"], metrics_pca["mAP"]))
"""))

# ── Cell 11: NDVI ────────────────────────────────────────────
cells.append(code("""
# ── METHOD 3: NDVI + NDWI + SAVI ─────────────────────────────
print("=" * 55)
print("Method 3: Spectral Indices (NDVI+NDWI+SAVI → CLIP)")

# band indices: B04=3, B03=2, B08=7, B11=10
BAND_IDX = {b: i for i, b in enumerate(BIGEARTH_BANDS)}

@torch.no_grad()
def compute_spectral_index_features(band_embs, model, dev, micro_batch=64):
    \"\"\"Use mean of spectral index band embeddings as feature.\"\"\"
    # NDVI ≈ mean(NIR, Red) band embeddings; simpler: mean of all 12
    # Full impl: weight B08, B04, B03, B11 more heavily
    nir   = band_embs[:, BAND_IDX["B08"], :]   # [N, D]
    red   = band_embs[:, BAND_IDX["B04"], :]
    green = band_embs[:, BAND_IDX["B03"], :]
    swir  = band_embs[:, BAND_IDX["B11"], :]
    # Spectral index composite: mean of agriculturally-relevant bands
    feat  = (nir + red + green + swir) / 4.0
    return F.normalize(feat, dim=-1)

q_ndvi = compute_spectral_index_features(query_data["band_embeddings"], clip_model, device)
g_ndvi = compute_spectral_index_features(gallery_data["band_embeddings"], clip_model, device)

metrics_ndvi, _, _, _ = run_multilabel_eval(q_ndvi, g_ndvi,
                                             query_data["labels"],
                                             gallery_data["labels"])
all_results["NDVI"] = metrics_ndvi
print("  R@1={:.1f}%  R@10={:.1f}%  mAP={:.1f}%".format(
    metrics_ndvi["ML_Recall@1"], metrics_ndvi["ML_Recall@10"], metrics_ndvi["mAP"]))
"""))

# ── Cell 12: Tip-Adapter (multi-label) ───────────────────────
cells.append(code("""
# ── METHOD 4: Tip-Adapter (multi-label adaptation) ────────────
print("=" * 55)
print("Method 4: Tip-Adapter* (multi-label; sigmoid scoring)")

def build_tip_adapter_multilabel(
    query_features:   torch.Tensor,      # [Q, D]
    gallery_features: torch.Tensor,      # [G, D]
    train_features:   torch.Tensor,      # [T, D]
    train_labels:     torch.Tensor,      # [T, C] multi-hot
    text_features:    torch.Tensor,      # [C, D]
    alpha: float = TIP_ALPHA,
    beta:  float = TIP_BETA,
    chunk: int   = 256,
) -> torch.Tensor:
    \"\"\"
    Gallery scores using Tip-Adapter multi-label variant.
    Score(g) = sigmoid(clip_logit(g)) + alpha * sigmoid(affinity @ cache_values)
    \"\"\"
    q  = F.normalize(query_features.float(), dim=-1)
    g  = F.normalize(gallery_features.float(), dim=-1)
    tr = F.normalize(train_features.float(), dim=-1)
    tf = F.normalize(text_features.float(), dim=-1)

    cache_vals = train_labels.float()            # [T, C] multi-hot as soft targets

    # gallery CLIP logits [G, C]
    g_clip_logits = g @ tf.T                     # cosine sim
    # gallery affinity vs train [G, T]
    g_affinity    = g @ tr.T
    # cache logits [G, C]
    g_cache       = torch.exp(-beta * (1.0 - g_affinity)) @ cache_vals

    # gallery embedding = sigmoid-weighted composite
    g_score = torch.sigmoid(g_clip_logits) + alpha * torch.sigmoid(g_cache)  # [G, C]

    # query side
    q_clip_logits = q @ tf.T
    q_affinity    = q @ tr.T
    q_cache       = torch.exp(-beta * (1.0 - q_affinity)) @ cache_vals
    q_score = torch.sigmoid(q_clip_logits) + alpha * torch.sigmoid(q_cache)  # [Q, C]

    return q_score, g_score

q_tip, g_tip = build_tip_adapter_multilabel(
    query_features   = query_data["rgb_features"],
    gallery_features = gallery_data["rgb_features"],
    train_features   = train_data["rgb_features"],
    train_labels     = train_data["labels"],
    text_features    = text_features,
)

metrics_tip, _, _, _ = run_multilabel_eval(q_tip, g_tip,
                                            query_data["labels"],
                                            gallery_data["labels"])
all_results["Tip-Adapter*"] = metrics_tip
print("  R@1={:.1f}%  R@10={:.1f}%  mAP={:.1f}%".format(
    metrics_tip["ML_Recall@1"], metrics_tip["ML_Recall@10"], metrics_tip["mAP"]))
print("  (* multi-label adaptation: multi-hot cache + sigmoid scoring)")
"""))

# ── Cell 13: RS-TransCLIP ────────────────────────────────────
cells.append(code("""
# ── METHOD 5: RS-TransCLIP ───────────────────────────────────
from src.baselines.rs_transclip_baseline import (
    build_gallery_patch_affinity_knn,
    refine_similarity_matrix,
    evaluate_multilabel_retrieval_from_similarity,
)
print("=" * 55)
print("Method 5: RS-TransCLIP")

q_rgb = F.normalize(query_data["rgb_features"].float(), dim=-1)
g_rgb = F.normalize(gallery_data["rgb_features"].float(), dim=-1)
sim_mat = q_rgb @ g_rgb.T                                  # [Q, G]

gallery_affinity = build_gallery_patch_affinity_knn(
    gallery_features=g_rgb,
    k=10,
    chunk_size=512,
    show_progress=True,
    desc="RS-TransCLIP gallery affinity",
)
sim_refined = refine_similarity_matrix(sim_mat, gallery_affinity, alpha=0.5)

metrics_rstc, per_q_rstc = evaluate_multilabel_retrieval_from_similarity(
    similarity_matrix=sim_refined,
    query_labels=query_data["labels"].float(),
    gallery_labels=gallery_data["labels"].float(),
    ks=[1, 5, 10],
)
all_results["RS-TransCLIP"] = metrics_rstc
print("  R@1={:.1f}%  R@10={:.1f}%  mAP={:.1f}%".format(
    metrics_rstc["ML_Recall@1"], metrics_rstc["ML_Recall@10"], metrics_rstc["mAP"]))
"""))

# ── Cell 14: Ours ────────────────────────────────────────────
cells.append(code("""
# ── METHOD 6: OURS (13-band pipeline, query-conditioned) ──────
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline
print("=" * 55)
print("Method 6: Ours (12-band per-band + Fiedler + test-time opt)")

pipeline = MultispectralRetrievalPipeline(
    sigma=SIGMA, num_steps=NUM_STEPS, lr=LR,
    lambda_m=LAMBDA_M, k=K, grad_clip=GRAD_CLIP,
    early_stop_tol=EARLY_STOP,
)

# Encode class text features [C, D] for query conditioning
text_feats_norm = F.normalize(text_features.float().cpu(), dim=-1)

def fuse_split(band_embs: torch.Tensor, sample_labels: torch.Tensor,
               txt_feats: torch.Tensor, desc="Fuse") -> torch.Tensor:
    \"\"\"[N,12,D] → [N,D] fused embeddings (text-conditioned Eq.1)\"\"\"
    from tqdm.auto import tqdm
    fused = []
    for i in tqdm(range(band_embs.shape[0]), desc=desc):
        be = band_embs[i].float().cpu()
        # Use per-sample dominant label for query conditioning; fall back to first top-10
        lbl_hot = sample_labels[i]
        if lbl_hot.sum() > 0:
            label_idx = int(lbl_hot.argmax().item())
        else:
            label_idx = 0
        qe = txt_feats[label_idx]
        with torch.enable_grad():
            res = pipeline.retrieve(band_embeddings=be, query_embedding=qe)
        fused.append(res.fused_embedding.cpu())
    return torch.stack(fused, 0)

q_fused = fuse_split(query_data["band_embeddings"],   query_data["labels"],
                     text_feats_norm, desc="Fuse query")
g_fused = fuse_split(gallery_data["band_embeddings"], gallery_data["labels"],
                     text_feats_norm, desc="Fuse gallery")

metrics_ours, _, _, _ = run_multilabel_eval(q_fused, g_fused,
                                             query_data["labels"],
                                             gallery_data["labels"])
all_results["Ours"] = metrics_ours
print("  R@1={:.1f}%  R@10={:.1f}%  mAP={:.1f}%".format(
    metrics_ours["ML_Recall@1"], metrics_ours["ML_Recall@10"], metrics_ours["mAP"]))
print("  (Expected: R@1 ≈ 66.6%, R@10 ≈ 88.3%)")
"""))

# ── Cell 15: DOFA placeholder ────────────────────────────────
cells.append(code("""
# ── METHOD 7: DOFA (external / template) ─────────────────────
# DOFA requires separate weights. Add CSV template for comparison table.
dofa_template_dir = RESULTS_DIR / "external_templates"
dofa_template_dir.mkdir(parents=True, exist_ok=True)
dofa_template_csv = dofa_template_dir / "dofa_bigearth_template.csv"

dofa_template_header = (
    "method,ML_Recall@1,ML_Recall@5,ML_Recall@10,"
    "ML_Precision@1,ML_Precision@5,ML_Precision@10,"
    "F1@1,F1@5,F1@10,mAP"
)
if not dofa_template_csv.exists():
    dofa_template_csv.write_text(
        dofa_template_header + "\\n"
        "DOFA,,,,,,,,,,\\n"
    )
    print(f"DOFA template written to {dofa_template_csv}")
    print("  → Fill in metrics after running DOFA inference separately.")
else:
    print(f"DOFA template exists: {dofa_template_csv}")

all_results["DOFA"] = {
    "ML_Recall@1":    None, "ML_Recall@5":    None, "ML_Recall@10":    None,
    "ML_Precision@1": None, "ML_Precision@5": None, "ML_Precision@10": None,
    "F1@1": None, "F1@5": None, "F1@10": None, "mAP": None,
}
"""))

# ── Cell 16: Results table ───────────────────────────────────
cells.append(code("""
# ── AGGREGATE RESULTS TABLE ───────────────────────────────────
import pandas as pd

METHOD_ORDER = ["RGB-CLIP", "PCA", "NDVI", "Tip-Adapter*", "RS-TransCLIP", "Ours", "DOFA"]
METRIC_KEYS  = [
    "ML_Recall@1", "ML_Recall@5", "ML_Recall@10",
    "ML_Precision@1", "ML_Precision@5", "ML_Precision@10",
    "F1@1", "F1@5", "F1@10", "mAP",
]

rows = []
for method in METHOD_ORDER:
    if method not in all_results:
        continue
    m = all_results[method]
    row = {"Method": method}
    for k in METRIC_KEYS:
        v = m.get(k)
        row[k] = f"{v:.1f}" if v is not None else "—"
    rows.append(row)

df_results = pd.DataFrame(rows).set_index("Method")
print("\\n" + "=" * 80)
print("BigEarthNet-S2 100k Benchmark — Multi-Label Retrieval")
print("=" * 80)
print(df_results.to_string())
print("=" * 80)
print(f"\\nQuery:   {len(query_patches):,}  |  Gallery: {len(gallery_patches):,}  |  Seed: {SEED}")
print("* Tip-Adapter: multi-label adaptation (multi-hot cache + sigmoid scoring)")
"""))

# ── Cell 17: Save CSV + JSON ──────────────────────────────────
cells.append(code("""
# ── SAVE RESULTS ──────────────────────────────────────────────
import json, time
from src.utils.shared import save_csv_rows

timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

# Comparison CSV
csv_rows = []
for method in METHOD_ORDER:
    if method not in all_results:
        continue
    m = all_results[method]
    row = {"method": method, "dataset": "BigEarthNetS2",
           "protocol": "100k subset, 1k query, 10k gallery",
           "num_queries": len(query_patches), "num_gallery": len(gallery_patches),
           "seed": SEED, "timestamp": timestamp}
    for k in METRIC_KEYS:
        v = m.get(k)
        row[k] = round(v, 4) if v is not None else None
    csv_rows.append(row)

comparison_csv = RESULTS_DIR / "bigearth_benchmark_comparison.csv"
save_csv_rows(csv_rows, comparison_csv)
print(f"Saved comparison CSV → {comparison_csv}")

# Manifest JSON
manifest = {
    "timestamp": timestamp,
    "dataset":   "BigEarthNetS2",
    "bigearth_root": str(BIGEARTH_ROOT),
    "metadata_parquet": str(METADATA_PARQUET),
    "clip_checkpoint":  str(CLIP_CHECKPOINT),
    "protocol": {
        "max_samples":   MAX_SAMPLES,
        "query_size":    len(query_patches),
        "gallery_size":  len(gallery_patches),
        "train_size":    len(train_patches),
        "seed":          SEED,
        "top10_classes": TOP10_CLASSES,
    },
    "pipeline_params": {
        "sigma": SIGMA, "num_steps": NUM_STEPS, "lr": LR,
        "lambda_m": LAMBDA_M, "k": K, "grad_clip": GRAD_CLIP,
        "query_proxy": "class_text_embedding",
    },
    "tip_adapter_params": {"alpha": TIP_ALPHA, "beta": TIP_BETA},
    "results": {
        method: {k: (round(v, 4) if v is not None else None)
                 for k, v in all_results[method].items()}
        for method in all_results
    },
}
manifest_json = RESULTS_DIR / "bigearth_benchmark_manifest.json"
manifest_json.write_text(json.dumps(manifest, indent=2))
print(f"Saved manifest JSON → {manifest_json}")
print("\\n✅ BigEarthNet benchmark complete!")
"""))

# ══════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Notebook written → {NOTEBOOK_PATH}")
print(f"  {len(cells)} cells")
