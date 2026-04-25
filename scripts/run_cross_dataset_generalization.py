#!/usr/bin/env python
"""
Cross-Dataset Generalization (Day 48-49, Task 3 Tuần 7)

Zero-shot: 'Ours' pipeline tuned on EuroSAT → applied to BigEarthNet.
Uses metadata.parquet for BigEarthNet labels (v2 TIF format on SSD).
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.eurosat import build_eurosat_subsets, EUROSAT_CLASSES, CLASS_TEXT_MAP as ES_TEXT
from src.datasets.bigearth_loader import (
    TOP10_CLASSES, CLASS_TEXT_MAP as BE_TEXT, BIGEARTH_BANDS, CLC43_TO_19,
    bigearth_collate_fn,
)
from src.models.per_band_encoder import encode_multispectral_batch
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline
from src.models.band_attribution import compute_class_band_attribution
from src.utils.metrics import evaluate_text_to_image_retrieval, evaluate_multilabel_image_retrieval
from src.utils.shared import get_device, load_openai_clip_model, save_csv_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# EuroSAT-tuned hyperparams (locked from Task 2 Tuần 7)
HP = dict(sigma=0.5, tau=None, lambda_m=0.1, k=5, num_steps=5, lr=0.01)
CKPT = REPO_ROOT / "checkpoints" / "ViT-B-16.pt"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bigearth_root",  default="/Volumes/Data/data/BigEarthNetS2/BigEarthNetS2")
    p.add_argument("--eurosat_root",   default="/Volumes/Data/data/EuroSAT_MS")
    p.add_argument("--metadata_parquet", default=str(REPO_ROOT/"data/BigEarthNetS2.local_backup_20260419/metadata.parquet"))
    p.add_argument("--output_dir",     default="results/cross_dataset")
    p.add_argument("--checkpoint",     default=str(CKPT))
    p.add_argument("--device",         default="auto", choices=["auto","mps","cuda","cpu"])
    p.add_argument("--query_size",     type=int, default=1000)
    p.add_argument("--gallery_size",   type=int, default=10000)
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--micro_batch",    type=int, default=32)
    p.add_argument("--attr_per_class", type=int, default=30)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--skip_bigearth_encoding", action="store_true",
                   help="Skip Phase 1 BigEarthNet encoding (use existing CSVs). Resume mode.")
    p.add_argument("--skip_eurosat_resolution", action="store_true")
    p.add_argument("--skip_attribution",        action="store_true")
    return p.parse_args()


# ── BigEarthNet parquet-based Dataset ─────────────────────────────────────────
class BigEarthParquetDataset(Dataset):
    """Lightweight dataset: labels from parquet, images from SSD TIF files."""

    def __init__(self, patch_df: pd.DataFrame, bigearth_root: Path):
        self.df = patch_df.reset_index(drop=True)
        self.root = bigearth_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import rasterio
        from scipy.ndimage import zoom as szoom
        row = self.df.iloc[idx]
        patch_id = row["patch_id"]
        # BigEarthNet patch folder: root / tile_name / patch_id /
        # tile_name = patch_id up to the last _XX_YY suffix removed → keep tile part
        # patch_id = S2A_..._T33UUP_26_57 → tile = S2A_..._T33UUP
        parts = patch_id.rsplit("_", 2)
        tile_name = parts[0]
        patch_dir = self.root / tile_name / patch_id

        bands = []
        for b in BIGEARTH_BANDS:
            tif = patch_dir / f"{patch_id}_{b}.tif"
            with rasterio.open(tif) as src:
                data = src.read(1).astype(np.float32)
            h, w = data.shape
            if h != 120 or w != 120:
                data = szoom(data, (120/h, 120/w), order=1)
            bands.append(data)
        image = torch.from_numpy(np.stack(bands) / 10000.0).clamp(0, 1)

        labels_top10 = row["labels_top10"]
        multihot = torch.zeros(len(TOP10_CLASSES))
        cls2idx = {c: i for i, c in enumerate(TOP10_CLASSES)}
        for lbl in labels_top10:
            if lbl in cls2idx:
                multihot[cls2idx[lbl]] = 1.0

        return {
            "image":       image,
            "labels":      multihot,
            "label_names": list(labels_top10),
            "patch_id":    patch_id,
        }


def be_collate(batch):
    return {
        "image":       torch.stack([b["image"] for b in batch]),
        "labels":      torch.stack([b["labels"] for b in batch]),
        "label_names": [b["label_names"] for b in batch],
        "patch_id":    [b["patch_id"] for b in batch],
    }


def load_bigearth_splits(parquet_path: Path, bigearth_root: Path,
                          query_size: int, gallery_size: int, seed: int):
    """Load parquet, filter, split into query/gallery. Returns DataLoaders."""
    log.info("Loading metadata.parquet ...")
    df = pd.read_parquet(parquet_path)
    df = df[~df["contains_seasonal_snow"] & ~df["contains_cloud_or_shadow"]].reset_index(drop=True)

    def map_top10(lbls):
        out = []
        for l in lbls:
            m = CLC43_TO_19.get(l, l)
            if m in TOP10_CLASSES and m not in out:
                out.append(m)
        return out

    df["labels_top10"] = df["labels"].apply(map_top10)
    df = df[df["labels_top10"].apply(len) > 0].reset_index(drop=True)
    log.info("Usable patches: %d", len(df))

    # Filter to patches that exist on disk (check tile exists)
    log.info("Checking patch availability on SSD ...")
    def exists(pid):
        parts = pid.rsplit("_", 2)
        p = bigearth_root / parts[0] / pid / f"{pid}_B02.tif"
        return p.exists()

    # Sample first to avoid checking all 418k
    rng = np.random.default_rng(seed)
    total_needed = query_size + gallery_size
    # Shuffle and take first max_check candidates
    max_check = min(len(df), total_needed * 5)
    candidate_idx = rng.choice(len(df), max_check, replace=False)
    valid_idx = []
    for i in tqdm(candidate_idx, desc="  Checking patches on disk"):
        pid = df.iloc[i]["patch_id"]
        parts = pid.rsplit("_", 2)
        p = bigearth_root / parts[0] / pid / f"{pid}_B02.tif"
        if p.exists():
            valid_idx.append(i)
        if len(valid_idx) >= total_needed:
            break

    if len(valid_idx) < total_needed:
        log.warning("Only %d valid patches found (need %d). Reducing sizes.", len(valid_idx), total_needed)
        query_size  = min(query_size,  len(valid_idx) // (query_size + gallery_size) * query_size)
        gallery_size = len(valid_idx) - query_size

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    query_df   = df_valid.iloc[:query_size]
    gallery_df = df_valid.iloc[query_size:query_size + gallery_size]
    log.info("Split: query=%d, gallery=%d", len(query_df), len(gallery_df))

    q_ds  = BigEarthParquetDataset(query_df,   bigearth_root)
    g_ds  = BigEarthParquetDataset(gallery_df, bigearth_root)
    return q_ds, g_ds


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_class_text_features(clip_model, clip_tok, class_list, text_map, device):
    texts = [f"A satellite image of {text_map.get(c,c)}." for c in class_list]
    with torch.no_grad():
        toks = clip_tok(texts).to(device)
        feats = F.normalize(clip_model.encode_text(toks).float(), dim=-1)
    return feats.cpu()  # (C, 512)


def encode_ours(loader, clip_model, clip_tok, class_list, text_map,
                device, micro_bs, multilabel=True):
    """Encode split with Ours pipeline. Returns dict of tensors."""
    pipeline = MultispectralRetrievalPipeline(**HP)
    class_feats = encode_class_text_features(clip_model, clip_tok, class_list, text_map, device)

    all_fused, all_band, all_query, all_labels, all_lnames = [], [], [], [], []

    for batch in tqdm(loader, desc="  Ours encode", leave=False):
        images = batch["image"]         # (N, B, H, W)
        N = images.shape[0]
        band_embs = encode_multispectral_batch(images, clip_model, device,
                                               micro_batch_size=micro_bs)  # (N,B,512)
        if multilabel:
            labels = batch["labels"]    # (N, C)
            lnames = batch["label_names"]
            q_embs = []
            for i in range(N):
                ln = lnames[i]
                primary = ln[0] if ln and ln[0] in class_list else class_list[0]
                q_embs.append(class_feats[class_list.index(primary)])
            q_embs_t = torch.stack(q_embs)
        else:
            labels = batch["label"]     # (N,)
            lnames = list(batch.get("label_name", [""] * N))
            q_embs_t = class_feats[labels.long()]  # (N, 512)

        fused = [pipeline.retrieve(band_embs[i], q_embs_t[i]).fused_embedding
                 for i in range(N)]

        all_fused.append(torch.stack(fused))
        all_band.append(band_embs)
        all_query.append(q_embs_t)
        all_labels.append(labels)
        all_lnames.extend(lnames)

    return {
        "fused":   torch.cat(all_fused),   # (N, 512)
        "band":    torch.cat(all_band),    # (N, B, 512)
        "query":   torch.cat(all_query),   # (N, 512)
        "labels":  torch.cat(all_labels),  # (N,) or (N,C)
        "lnames":  all_lnames,
    }


def encode_rgb(loader, clip_model, device, rgb_idx, micro_bs, multilabel=True):
    """RGB-CLIP baseline encoding."""
    from src.models.per_band_encoder import CLIP_MEAN, CLIP_STD
    all_embs, all_labels = [], []
    for batch in tqdm(loader, desc="  RGB encode", leave=False):
        images = batch["image"]
        N = images.shape[0]
        r, g, b = rgb_idx
        rgb = images[:, [r, g, b]].float().clamp(0, 1)
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        rgb = (rgb - CLIP_MEAN.view(1,3,1,1)) / CLIP_STD.view(1,3,1,1)
        embs = []
        for s in range(0, N, micro_bs):
            chunk = rgb[s:s+micro_bs].to(device)
            with torch.no_grad():
                f = F.normalize(clip_model.encode_image(chunk).float(), dim=-1)
            embs.append(f.cpu())
        all_embs.append(torch.cat(embs))
        all_labels.append(batch["labels"] if multilabel else batch["label"])
    return {"fused": torch.cat(all_embs), "labels": torch.cat(all_labels)}


# ── Per-class breakdown ───────────────────────────────────────────────────────

def perclass_breakdown(ranked_rel: torch.Tensor, lnames_list, class_list):
    rows = []
    for cls in class_list:
        mask = torch.tensor([cls in (ln if isinstance(ln, list) else [ln])
                             for ln in lnames_list], dtype=torch.bool)
        if mask.sum() == 0:
            continue
        rel = ranked_rel[mask]
        rows.append({
            "class":      cls,
            "n_queries":  int(mask.sum()),
            "R@1":  round(rel[:,:1].any(1).float().mean().item()*100, 2),
            "R@5":  round(rel[:,:5].any(1).float().mean().item()*100, 2),
            "R@10": round(rel[:,:10].any(1).float().mean().item()*100, 2),
        })
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_bar(rows, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods  = [r["method"] for r in rows]
    r1  = [r.get("R@1",  0) for r in rows]
    r10 = [r.get("R@10", 0) for r in rows]
    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(methods)*1.6), 6))
    b1 = ax.bar(x-w/2, r1,  w, label="R@1",  color=["#2563EB" if "Ours" in m else "#94A3B8" for m in methods], alpha=0.9)
    b2 = ax.bar(x+w/2, r10, w, label="R@10", color=["#1E40AF" if "Ours" in m else "#64748B" for m in methods], alpha=0.9)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Performance (%)"); ax.legend()
    ax.set_title("Cross-Dataset: BigEarthNet (zero-shot, EuroSAT hyperparams)", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir/"cross_dataset_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig); log.info("Saved cross_dataset_bar_chart.png")


def plot_resolution(rows, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted(set(r["dataset"] for r in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), sharey=False)
    if len(datasets) == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        dr = [r for r in rows if r["dataset"]==ds]
        res = [r["resolution"] for r in dr]
        x = np.arange(len(res))
        ax.plot(x, [r.get("R@1",0) for r in dr],  "o-", label="R@1",  color="#2563EB", lw=2, ms=8)
        ax.plot(x, [r.get("R@10",0) for r in dr], "s-", label="R@10", color="#DC2626", lw=2, ms=8)
        ax.set_xticks(x); ax.set_xticklabels(res)
        ax.set_title(f"{ds} — Resolution Robustness", fontweight="bold")
        ax.set_ylabel("Performance (%)"); ax.legend(); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir/"resolution_robustness_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig); log.info("Saved resolution_robustness_plot.png")


def plot_attribution_comparison(es_attr, be_attr, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle("Band Attribution: EuroSAT (13 bands) vs BigEarthNet (12 bands)", fontsize=14, fontweight="bold")
    cmap = "YlOrRd"
    for ax, attr, title, band_names in [
        (axes[0], es_attr, "EuroSAT (13 bands, single-label)",
         ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]),
        (axes[1], be_attr, "BigEarthNet (12 bands, multi-label)", list(BIGEARTH_BANDS)),
    ]:
        cls_names = attr.class_names
        n_cls = len(cls_names); n_bands = len(band_names)
        mat = np.zeros((n_cls, n_bands))
        for i, cls in enumerate(cls_names):
            a = attr.class_attributions[cls]
            mat[i, :len(a)] = a
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(n_bands)); ax.set_xticklabels(band_names, fontsize=8, fontweight="bold")
        ax.set_yticks(range(n_cls));   ax.set_yticklabels(cls_names, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Spectral Band")
        fig.colorbar(im, ax=ax, shrink=0.8)
        for i in range(n_cls):
            for j in range(n_bands):
                v = mat[i,j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "black")
    plt.tight_layout()
    fig.savefig(out_dir/"band_attribution_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig); log.info("Saved band_attribution_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s | Output: %s", device, out_dir)

    clip_model, clip_tok = load_openai_clip_model(Path(args.checkpoint), device)
    log.info("CLIP loaded.")
    t0_global = time.perf_counter()

    # ── BigEarthNet splits (parquet-based, ONE scan) ───────────────────────────
    ts = datetime.now(timezone.utc).isoformat()
    q_enc = None  # populated in Phase 1 unless skipped

    if not args.skip_bigearth_encoding:
        log.info("=== Phase 1: Zero-shot BigEarthNet evaluation ===")
        q_ds, g_ds = load_bigearth_splits(
            Path(args.metadata_parquet), Path(args.bigearth_root),
            args.query_size, args.gallery_size, args.seed,
        )
        q_loader = DataLoader(q_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, collate_fn=be_collate, pin_memory=False)
        g_loader = DataLoader(g_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, collate_fn=be_collate, pin_memory=False)

        be_rgb = (list(BIGEARTH_BANDS).index("B04"),
                  list(BIGEARTH_BANDS).index("B03"),
                  list(BIGEARTH_BANDS).index("B02"))

        log.info("Encoding query with Ours pipeline ...")
        q_enc = encode_ours(q_loader, clip_model, clip_tok, TOP10_CLASSES, BE_TEXT,
                            device, args.micro_batch, multilabel=True)
        log.info("Encoding gallery with Ours pipeline ...")
        g_enc = encode_ours(g_loader, clip_model, clip_tok, TOP10_CLASSES, BE_TEXT,
                            device, args.micro_batch, multilabel=True)

        metrics_ours, _, _, ranked_rel_ours, pq = evaluate_multilabel_image_retrieval(
            q_enc["fused"], q_enc["labels"], g_enc["fused"], g_enc["labels"])
        log.info("Ours → R@1=%.2f%%, R@10=%.2f%%", metrics_ours["R@1"], metrics_ours["R@10"])

        log.info("Encoding with RGB-CLIP ...")
        q_rgb = encode_rgb(q_loader, clip_model, device, be_rgb, args.micro_batch)
        g_rgb = encode_rgb(g_loader, clip_model, device, be_rgb, args.micro_batch)
        metrics_rgb, *_ = evaluate_multilabel_image_retrieval(
            q_rgb["fused"], q_rgb["labels"], g_rgb["fused"], g_rgb["labels"])
        log.info("RGB-CLIP → R@1=%.2f%%, R@10=%.2f%%", metrics_rgb["R@1"], metrics_rgb["R@10"])

        comparison_rows = [
            {"method":"Ours (EuroSAT HP)","dataset":"BigEarthNet-S2",
             "protocol":f"{len(q_ds)}q_{len(g_ds)}g","zero_shot":True,"timestamp":ts,
             **{k:round(v,4) for k,v in metrics_ours.items()}},
            {"method":"RGB-CLIP","dataset":"BigEarthNet-S2",
             "protocol":f"{len(q_ds)}q_{len(g_ds)}g","zero_shot":False,"timestamp":ts,
             **{k:round(v,4) for k,v in metrics_rgb.items()}},
        ]
        save_csv_rows(comparison_rows, out_dir/"cross_dataset_comparison.csv")
        log.info("Saved cross_dataset_comparison.csv")

        pc_rows = perclass_breakdown(ranked_rel_ours, q_enc["lnames"], TOP10_CLASSES)
        if pc_rows:
            save_csv_rows(pc_rows, out_dir/"per_class_breakdown.csv")
            log.info("Saved per_class_breakdown.csv (%d classes)", len(pc_rows))
        if pq:
            save_csv_rows(pq, out_dir/"bigearth_per_query_ours.csv")

        plot_bar(comparison_rows, out_dir)
    else:
        log.info("=== Phase 1: SKIPPED (--skip_bigearth_encoding) ===")
        csv_path = out_dir / "cross_dataset_comparison.csv"
        if csv_path.exists():
            import csv
            with open(csv_path) as f:
                comparison_rows = list(csv.DictReader(f))
            plot_bar(comparison_rows, out_dir)
            log.info("Reloaded existing cross_dataset_comparison.csv and re-plotted.")
        else:
            comparison_rows = []
            log.warning("cross_dataset_comparison.csv not found — bar chart skipped.")

    # ── Phase 2: EuroSAT Resolution Robustness ────────────────────────────────
    if not args.skip_eurosat_resolution:
        log.info("=== Phase 2: EuroSAT Resolution Robustness ===")
        res_rows = []
        pipeline = MultispectralRetrievalPipeline(**HP)
        es_class_feats = encode_class_text_features(clip_model, clip_tok, EUROSAT_CLASSES, ES_TEXT, device)
        _es = build_eurosat_subsets(root=Path(args.eurosat_root), seed=args.seed)
        es_splits = _es["subsets"]

        for sz in [64, 128]:
            log.info("  EuroSAT @ %dx%d ...", sz, sz)
            all_f, all_l = [], []
            for split_name in ["test"]:
                ldr = DataLoader(es_splits[split_name], batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
                for batch in tqdm(ldr, desc=f"  encode@{sz}", leave=False):
                    imgs = batch["image"]   # (N,13,64,64) native
                    N = imgs.shape[0]
                    # Pre-resize to target spatial resolution, then encode at 224 for CLIP
                    if sz != imgs.shape[-1]:
                        N_b, B, H, W = imgs.shape
                        imgs_r = F.interpolate(
                            imgs.float().view(N_b*B,1,H,W),
                            size=(sz,sz), mode="bilinear", align_corners=False
                        ).view(N_b,B,sz,sz)
                    else:
                        imgs_r = imgs.float()
                    be = encode_multispectral_batch(imgs_r, clip_model, device,
                                                   target_size=224, micro_batch_size=args.micro_batch)
                    lbs = batch["label"]
                    qe = es_class_feats[lbs.long()]
                    fused = [pipeline.retrieve(be[i], qe[i]).fused_embedding for i in range(N)]
                    all_f.append(torch.stack(fused)); all_l.append(lbs)
            # For gallery use train split
            all_gf, all_gl = [], []
            for batch in tqdm(DataLoader(es_splits["train"], batch_size=args.batch_size,
                                         shuffle=False, num_workers=0),
                              desc=f"  gallery@{sz}", leave=False):
                imgs = batch["image"]; N = imgs.shape[0]
                if sz != imgs.shape[-1]:
                    N_b, B, H, W = imgs.shape
                    imgs = F.interpolate(
                        imgs.float().view(N_b*B,1,H,W),
                        size=(sz,sz), mode="bilinear", align_corners=False
                    ).view(N_b,B,sz,sz)
                be = encode_multispectral_batch(imgs, clip_model, device,
                                               target_size=224, micro_batch_size=args.micro_batch)
                lbs = batch["label"]
                qe = es_class_feats[lbs.long()]
                fused = [pipeline.retrieve(be[i], qe[i]).fused_embedding for i in range(N)]
                all_gf.append(torch.stack(fused)); all_gl.append(lbs)
            mets, *_ = evaluate_text_to_image_retrieval(
                torch.cat(all_f), torch.cat(all_l),
                torch.cat(all_gf), torch.cat(all_gl))
            res_rows.append({"dataset":"EuroSAT","resolution":f"{sz}x{sz}",
                             "method":"Ours","timestamp":ts,
                             **{k:round(v,4) for k,v in mets.items()}})
            log.info("    R@1=%.2f%%, R@10=%.2f%%", mets["R@1"], mets["R@10"])

        if res_rows:
            save_csv_rows(res_rows, out_dir/"resolution_robustness.csv")
            plot_resolution(res_rows, out_dir)
    else:
        log.info("Skipping resolution (--skip_eurosat_resolution)")

    # ── Phase 3: Band Attribution ──────────────────────────────────────────────
    if not args.skip_attribution:
        log.info("=== Phase 3: Band Attribution Comparison ===")
        n = args.attr_per_class
        es_class_feats2 = encode_class_text_features(clip_model, clip_tok, EUROSAT_CLASSES, ES_TEXT, device)

        # EuroSAT attribution
        _es2 = build_eurosat_subsets(root=Path(args.eurosat_root), seed=args.seed)
        es_splits2 = _es2["subsets"]
        es_band_list, es_q_list, es_cls_list = [], [], []
        counts = {c:0 for c in EUROSAT_CLASSES}
        for batch in DataLoader(es_splits2["test"], batch_size=args.batch_size,
                                shuffle=False, num_workers=0):
            imgs = batch["image"]; lbs = batch["label"]
            be = encode_multispectral_batch(imgs, clip_model, device,
                                            micro_batch_size=args.micro_batch)
            for i in range(len(lbs)):
                c = EUROSAT_CLASSES[int(lbs[i])]
                if counts[c] < n:
                    es_band_list.append(be[i])
                    es_q_list.append(es_class_feats2[int(lbs[i])].cpu())
                    es_cls_list.append(c)
                    counts[c] += 1
            if all(v >= n for v in counts.values()): break
        es_attr = compute_class_band_attribution(es_band_list, es_q_list, es_cls_list, sigma=HP["sigma"])

        # BigEarthNet attribution
        # If q_enc is available (Phase 1 ran), reuse its band embeddings.
        # If not (--skip_bigearth_encoding), encode a small fresh subset (~n×10 patches).
        be_band_list, be_q_list, be_cls_list = [], [], []
        be_counts = {c: 0 for c in TOP10_CLASSES}

        if q_enc is not None:
            # Fast path: reuse already-encoded embeddings from Phase 1
            N_enc = q_enc["band"].shape[0]
            for i in range(N_enc):
                for lname in q_enc["lnames"][i]:
                    if lname in be_counts and be_counts[lname] < n:
                        be_band_list.append(q_enc["band"][i])
                        be_q_list.append(q_enc["query"][i])
                        be_cls_list.append(lname)
                        be_counts[lname] += 1
        else:
            # Resume path: encode a small fresh subset for attribution only
            log.info("  q_enc not available (resume mode): encoding small BigEarthNet subset for attribution ...")
            attr_q_ds, _ = load_bigearth_splits(
                Path(args.metadata_parquet), Path(args.bigearth_root),
                query_size=min(n * len(TOP10_CLASSES) * 3, 1000),
                gallery_size=1,
                seed=args.seed,
            )
            be_class_feats = encode_class_text_features(clip_model, clip_tok, TOP10_CLASSES, BE_TEXT, device)
            attr_loader = DataLoader(attr_q_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, collate_fn=be_collate)
            for batch in tqdm(attr_loader, desc="  BE attribution encode", leave=False):
                imgs = batch["image"]
                N_b = imgs.shape[0]
                band_embs = encode_multispectral_batch(imgs, clip_model, device,
                                                      target_size=224,
                                                      micro_batch_size=args.micro_batch)
                for i in range(N_b):
                    lnames = batch["label_names"][i]
                    for lname in lnames:
                        if lname in be_counts and be_counts[lname] < n:
                            idx = TOP10_CLASSES.index(lname) if lname in TOP10_CLASSES else 0
                            be_band_list.append(band_embs[i])
                            be_q_list.append(be_class_feats[idx].cpu())
                            be_cls_list.append(lname)
                            be_counts[lname] += 1
                if all(v >= n for v in be_counts.values()):
                    break

        be_attr = compute_class_band_attribution(be_band_list, be_q_list, be_cls_list, sigma=HP["sigma"])


        # Save CSVs
        def save_attr_csv(attr, path):
            rows = []
            for cls in attr.class_names:
                row = {"class": cls, "n_samples": attr.class_counts[cls]}
                for band, val in zip(attr.band_names, attr.class_attributions[cls]):
                    row[band] = round(float(val), 5)
                rows.append(row)
            if rows: save_csv_rows(rows, path)
        save_attr_csv(es_attr, out_dir/"band_attribution_eurosat.csv")
        save_attr_csv(be_attr, out_dir/"band_attribution_bigearth.csv")

        # Domain shift
        PAIRS = [("AnnualCrop","Arable land"),("Forest","Broad-leaved forest"),
                 ("Residential","Urban fabric"),("SeaLake","Inland waters"),
                 ("Pasture","Pastures"),("HerbaceousVegetation","Natural grassland and sparsely vegetated areas")]
        shift_rows = []
        for es_c, be_c in PAIRS:
            if es_c not in es_attr.class_attributions or be_c not in be_attr.class_attributions:
                continue
            ev = torch.tensor(es_attr.class_attributions[es_c], dtype=torch.float32)
            bv = torch.tensor(be_attr.class_attributions[be_c],  dtype=torch.float32)
            mn = min(len(ev), len(bv)); ev = ev[:mn]; bv = bv[:mn]
            cos = float(F.cosine_similarity(ev.unsqueeze(0), bv.unsqueeze(0)))
            l2  = float(torch.norm(ev-bv))
            diff = (bv-ev).abs(); top_b = diff.argsort(descending=True)[:3].tolist()
            bnames = list(BIGEARTH_BANDS)
            shift_rows.append({"eurosat_class":es_c,"bigearth_class":be_c,
                               "cosine_similarity":round(cos,4),"l2_distance":round(l2,4),
                               "shift_magnitude":round(1-cos,4),
                               "top_shifted_bands":", ".join(bnames[j] for j in top_b if j<len(bnames))})
        shift_rows.sort(key=lambda r: r["shift_magnitude"], reverse=True)
        if shift_rows:
            save_csv_rows(shift_rows, out_dir/"domain_shift_analysis.csv")
            log.info("Saved domain_shift_analysis.csv")
        plot_attribution_comparison(es_attr, be_attr, out_dir)
    else:
        log.info("Skipping attribution (--skip_attribution)")

    # ── Manifest ──────────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter()-t0_global, 1)
    manifest = {
        "experiment": "cross_dataset_generalization", "day":"48-49","week":7,"task":3,
        "timestamp": ts, "elapsed_total_s": elapsed,
        "zero_shot_definition": "Hyperparameters tuned on EuroSAT val; CLIP weights frozen.",
        "ours_hyperparameters": HP,
        "eurosat_root": args.eurosat_root, "bigearth_root": args.bigearth_root,
        "metadata_parquet": args.metadata_parquet,
        "query_size": args.query_size, "gallery_size": args.gallery_size,
        "seed": args.seed, "device": str(device),
    }
    (out_dir/"cross_dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("="*55)
    log.info("ALL DONE in %.1f s. Results: %s", elapsed, out_dir)
    log.info("="*55)


if __name__ == "__main__":
    main()
