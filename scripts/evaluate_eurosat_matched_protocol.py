#!/usr/bin/env python3
"""Evaluate the 13-band pipeline on the standard EuroSAT 80/10/10 protocol.

This script is intentionally standalone so the matched-protocol experiment can
be added without modifying the existing baseline sources or result artifacts.

Outputs are written under ``results/matched_protocol/``:
    - eurosat_matched_protocol_summary.csv
    - eurosat_matched_protocol_per_query.csv
    - eurosat_matched_protocol_split_manifest.csv
    - eurosat_matched_protocol_comparison.csv
    - eurosat_matched_protocol_manifest.json
    - eurosat_matched_protocol_query_band_embeddings.pt
    - eurosat_matched_protocol_gallery_band_embeddings.pt

The comparison table merges:
    - existing PCA EuroSAT baseline result
    - existing NDVI EuroSAT baseline result
    - existing RS-TransCLIP EuroSAT baseline result
    - newly-computed 13-band pipeline result on the same split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure `src` imports work when the script is run as `python3 scripts/...`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.eurosat import build_eurosat_dataloaders
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline
from src.utils.metrics import evaluate_text_to_image_retrieval
from src.utils.shared import get_device, load_openai_clip_model, save_csv_rows


CLIP_MEAN = torch.tensor([0.48145466, 0.45782750, 0.40821073], dtype=torch.float32)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)


def write_json(payload: Dict[str, Any], output_path: Path) -> None:
    """Save a JSON artifact with deterministic formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


@torch.no_grad()
def encode_class_text_features(
    clip_model,
    class_prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Encode EuroSAT class text prompts with CLIP's text encoder.

    Args:
        clip_model: Frozen CLIP model.
        class_prompts: List of C text prompts, one per class
            (e.g. from ``dataset.get_text_prompts()``).
        device: Target device.

    Returns:
        L2-normalized text features [C, D] on CPU.
    """
    import clip as _clip_pkg
    tokens = _clip_pkg.tokenize(class_prompts).to(device)
    features = clip_model.encode_text(tokens).float()
    return F.normalize(features, dim=-1).cpu()


def preprocess_band_batch(
    images: torch.Tensor,
    *,
    target_size: int = 224,
    reflectance_scale: float = 10_000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Convert ``(N, 13, H, W)`` multispectral tensors into CLIP-ready batches."""
    if images.ndim != 4:
        raise ValueError(f"Expected shape (N, B, H, W), got {tuple(images.shape)}")
    if images.shape[1] < 2:
        raise ValueError(f"Expected at least 2 bands, got {images.shape[1]}")

    x = images.to(torch.float32)
    if x.max() > 1.0:
        x = x / reflectance_scale
    x = x.clamp(clamp_range[0], clamp_range[1])

    n, b, h, w = x.shape
    x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1).view(n * b, 3, h, w)
    x = F.interpolate(
        x,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    mean = CLIP_MEAN.view(1, 3, 1, 1)
    std = CLIP_STD.view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def encode_multispectral_batch(
    images: torch.Tensor,
    clip_model,
    device: torch.device,
    *,
    target_size: int = 224,
    reflectance_scale: float = 10_000.0,
    clamp_range: tuple[float, float] = (0.0, 1.0),
    micro_batch_size: int = 64,
) -> torch.Tensor:
    """Encode a multispectral batch into per-band CLIP embeddings."""
    clip_model.eval()

    clip_inputs = preprocess_band_batch(
        images,
        target_size=target_size,
        reflectance_scale=reflectance_scale,
        clamp_range=clamp_range,
    )

    chunks: List[torch.Tensor] = []
    for start in range(0, clip_inputs.shape[0], micro_batch_size):
        stop = start + micro_batch_size
        chunk = clip_inputs[start:stop].to(device)
        features = clip_model.encode_image(chunk).float()
        chunks.append(F.normalize(features, dim=-1).cpu())

    stacked = torch.cat(chunks, dim=0)
    return stacked.view(images.shape[0], images.shape[1], -1)


@torch.inference_mode()
def encode_loader_with_band_embeddings(
    loader: DataLoader,
    clip_model,
    device: torch.device,
    *,
    micro_batch_size: int,
    desc: str,
    show_progress: bool,
) -> Dict[str, Any]:
    """Encode a dataloader into per-band embeddings while preserving metadata."""
    embedding_chunks: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    indices: List[torch.Tensor] = []
    label_names: List[str] = []
    paths: List[str] = []

    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader
    for batch in iterator:
        embeddings = encode_multispectral_batch(
            batch["image"],
            clip_model,
            device,
            micro_batch_size=micro_batch_size,
        )
        embedding_chunks.append(embeddings.cpu())
        labels.append(batch["label"].cpu())

        batch_indices = batch["index"]
        if torch.is_tensor(batch_indices):
            indices.append(batch_indices.cpu())
        else:
            indices.append(torch.tensor(batch_indices, dtype=torch.long))

        label_names.extend([str(name) for name in batch["label_name"]])
        paths.extend([str(path) for path in batch["path"]])

    return {
        "band_embeddings": torch.cat(embedding_chunks, dim=0),
        "label": torch.cat(labels, dim=0),
        "index": torch.cat(indices, dim=0),
        "label_name": label_names,
        "path": paths,
    }


def save_band_cache(payload: Dict[str, Any], cache_path: Path, *, seed: int, split_name: str) -> None:
    """Persist encoded query/gallery band embeddings for reuse."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "split_name": split_name,
        "seed": int(seed),
        "band_embeddings": payload["band_embeddings"],
        "label": payload["label"],
        "index": payload["index"],
        "label_name": list(payload["label_name"]),
        "path": list(payload["path"]),
    }
    torch.save(serializable, cache_path)


def load_band_cache(cache_path: Path, *, seed: int, split_name: str) -> Dict[str, Any]:
    """Load a previously-saved query/gallery band-embedding cache."""
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("seed") != seed:
        raise ValueError(
            f"Cache seed mismatch for {cache_path}: expected {seed}, got {payload.get('seed')}"
        )
    if payload.get("split_name") != split_name:
        raise ValueError(
            f"Cache split mismatch for {cache_path}: "
            f"expected {split_name}, got {payload.get('split_name')}"
        )
    return payload


def maybe_load_or_encode_split(
    *,
    cache_path: Path,
    split_name: str,
    seed: int,
    loader: DataLoader,
    clip_model,
    device: torch.device,
    micro_batch_size: int,
    show_progress: bool,
    force_reencode: bool,
) -> Dict[str, Any]:
    """Reuse cached band embeddings when possible, otherwise encode from raw images."""
    if cache_path.exists() and not force_reencode:
        print(f"Loading cached {split_name} band embeddings from {cache_path}")
        return load_band_cache(cache_path, seed=seed, split_name=split_name)

    payload = encode_loader_with_band_embeddings(
        loader,
        clip_model,
        device,
        micro_batch_size=micro_batch_size,
        desc=f"Encoding EuroSAT {split_name} per-band embeddings",
        show_progress=show_progress,
    )
    save_band_cache(payload, cache_path, seed=seed, split_name=split_name)
    print(f"Saved {split_name} band embeddings to {cache_path}")
    return payload


def fuse_band_embeddings(
    batch_band_embeddings: torch.Tensor,
    pipeline: MultispectralRetrievalPipeline,
    *,
    query_text_features: torch.Tensor,
    sample_labels: torch.Tensor,
    desc: str,
    show_progress: bool,
) -> Dict[str, Any]:
    """
    Fuse per-band embeddings one sample at a time using the current pipeline.

    The affinity graph (Eq. 1) is conditioned on the text embedding of each
    sample's class label, as specified in ACIVS_2026_Implementation_Plan (Day 21).
    This replaces the previous ``mean(band_embeddings)`` proxy.

    Args:
        batch_band_embeddings: [N, B, D] per-band embeddings.
        pipeline: MultispectralRetrievalPipeline instance.
        query_text_features: [C, D] L2-normalised class text embeddings from CLIP.
        sample_labels: [N] integer class indices for each sample.
        desc: tqdm description string.
        show_progress: Whether to show a progress bar.

    Returns:
        Dict with keys: features [N, D], elapsed_ms, optimization_ms.
    """
    fused_features: List[torch.Tensor] = []
    elapsed_ms: List[float] = []
    optimization_ms: List[float] = []

    query_text_features = F.normalize(query_text_features.float().cpu(), dim=-1)

    iterator: Iterable[int] = range(batch_band_embeddings.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc=desc)

    for idx in iterator:
        band_embeddings = batch_band_embeddings[idx].cpu().float()
        label_idx = int(sample_labels[idx].item())
        # Use class text embedding as query — text-conditioned affinity (Eq. 1)
        query_embedding = query_text_features[label_idx]
        with torch.enable_grad():
            result = pipeline.retrieve(
                band_embeddings=band_embeddings,
                query_embedding=query_embedding,
            )
        fused_features.append(result.fused_embedding.cpu())
        elapsed_ms.append(float(result.elapsed_ms))
        optimization_ms.append(float(result.optimization_ms))

    return {
        "features": torch.stack(fused_features, dim=0),
        "elapsed_ms": elapsed_ms,
        "optimization_ms": optimization_ms,
    }


def build_split_manifest(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten train/val/test membership into one CSV-friendly manifest."""
    dataset = bundle["dataset"]
    split_indices = bundle["indices"]

    rows: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        for position, dataset_index in enumerate(split_indices[split_name]):
            sample = dataset.samples[dataset_index]
            rows.append(
                {
                    "split": split_name,
                    "split_position": position,
                    "dataset_index": int(dataset_index),
                    "label": int(sample["label"]),
                    "label_name": str(sample["label_name"]),
                    "path": str(sample["path"]),
                }
            )
    return rows


def load_single_csv_row(csv_path: Path, *, dataset_name: str) -> Optional[Dict[str, str]]:
    """Return the first row matching the requested dataset, or ``None``."""
    if not csv_path.exists():
        return None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") == dataset_name:
                return row
    return None


def build_comparison_rows(
    *,
    ours_summary: Dict[str, Any],
    pca_csv: Path,
    ndvi_csv: Path,
    rs_csv: Path,
) -> List[Dict[str, Any]]:
    """Merge stored baseline rows with the newly-computed matched-protocol row."""
    rows: List[Dict[str, Any]] = []

    baseline_specs = [
        ("PCA", pca_csv, "existing baseline artifact"),
        ("NDVI", ndvi_csv, "existing baseline artifact"),
        ("RS-TransCLIP", rs_csv, "existing baseline artifact"),
    ]
    for method, csv_path, source_note in baseline_specs:
        raw_row = load_single_csv_row(csv_path, dataset_name="EuroSAT_MS")
        if raw_row is None:
            continue

        notes = source_note
        if method == "RS-TransCLIP" and raw_row.get("best_alpha"):
            notes = f"{source_note}; best_alpha={raw_row['best_alpha']}"

        rows.append(
            {
                "method": method,
                "dataset": raw_row["dataset"],
                "protocol": "EuroSAT 80/10/10 matched protocol",
                "num_queries": int(raw_row["num_queries"]),
                "num_gallery": int(raw_row["num_gallery"]),
                "R@1": float(raw_row["R@1"]),
                "R@5": float(raw_row["R@5"]),
                "R@10": float(raw_row["R@10"]),
                "mAP": float(raw_row["mAP"]),
                "source_artifact": str(csv_path),
                "notes": notes,
            }
        )

    rows.append(
        {
            "method": ours_summary["method"],
            "dataset": ours_summary["dataset"],
            "protocol": ours_summary["protocol"],
            "num_queries": int(ours_summary["num_queries"]),
            "num_gallery": int(ours_summary["num_gallery"]),
            "R@1": float(ours_summary["R@1"]),
            "R@5": float(ours_summary["R@5"]),
            "R@10": float(ours_summary["R@10"]),
            "mAP": float(ours_summary["mAP"]),
            "source_artifact": str(ours_summary["summary_csv"]),
            "notes": "new matched-protocol 13-band pipeline evaluation",
        }
    )

    ours_queries = float(ours_summary["num_queries"])
    ours_gallery = float(ours_summary["num_gallery"])
    for row in rows:
        row["queries_vs_ours"] = float(row["num_queries"]) / max(ours_queries, 1.0)
        row["gallery_vs_ours"] = float(row["num_gallery"]) / max(ours_gallery, 1.0)

    method_order = {
        "PCA": 0,
        "NDVI": 1,
        "RS-TransCLIP": 2,
        "Ours (13-band pipeline)": 3,
    }
    rows.sort(key=lambda row: method_order.get(str(row["method"]), 99))
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the 13-band EuroSAT pipeline on the matched 80/10/10 protocol."
    )
    parser.add_argument("--eurosat-root", type=Path, default=Path("data/EuroSAT_MS"))
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("checkpoints/ViT-B-16.pt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/matched_protocol"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=None,
                        help="Affinity softmax temperature τ (None = symmetric normalize)")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-m", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-tol", type=float, default=1e-6)
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Ignore cached query/gallery band embeddings and re-encode from raw images.",
    )
    parser.add_argument(
        "--hide-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)

    parsed.results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = parsed.results_dir / "eurosat_matched_protocol_summary.csv"
    per_query_csv = parsed.results_dir / "eurosat_matched_protocol_per_query.csv"
    split_manifest_csv = parsed.results_dir / "eurosat_matched_protocol_split_manifest.csv"
    comparison_csv = parsed.results_dir / "eurosat_matched_protocol_comparison.csv"
    manifest_json = parsed.results_dir / "eurosat_matched_protocol_manifest.json"
    query_cache = parsed.results_dir / "eurosat_matched_protocol_query_band_embeddings.pt"
    gallery_cache = parsed.results_dir / "eurosat_matched_protocol_gallery_band_embeddings.pt"

    pca_csv = Path("results/pca_baseline/pca_baseline_results.csv")
    ndvi_csv = Path("results/ndvi_baseline/ndvi_baseline_results.csv")
    rs_csv = Path("results/rs_transclip/rs_transclip_baseline_results_notebook.csv")

    device = get_device()
    print(f"Using device: {device}")

    clip_model, _ = load_openai_clip_model(parsed.clip_checkpoint, device)

    bundle = build_eurosat_dataloaders(
        root=parsed.eurosat_root,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=parsed.seed,
        normalize=True,
        reflectance_scale=10_000.0,
        clamp_range=(0.0, 1.0),
        transform=None,
    )

    # Encode class text features for query-conditioned affinity (Eq. 1)
    dataset = bundle["dataset"]
    print("Encoding class text embeddings...")
    class_text_features = encode_class_text_features(
        clip_model,
        class_prompts=dataset.get_text_prompts(),
        device=device,
    )  # [C, D]

    save_csv_rows(build_split_manifest(bundle), split_manifest_csv)

    query_payload = maybe_load_or_encode_split(
        cache_path=query_cache,
        split_name="val",
        seed=parsed.seed,
        loader=bundle["loaders"]["val"],
        clip_model=clip_model,
        device=device,
        micro_batch_size=parsed.micro_batch_size,
        show_progress=not parsed.hide_progress,
        force_reencode=parsed.force_reencode,
    )
    gallery_payload = maybe_load_or_encode_split(
        cache_path=gallery_cache,
        split_name="test",
        seed=parsed.seed,
        loader=bundle["loaders"]["test"],
        clip_model=clip_model,
        device=device,
        micro_batch_size=parsed.micro_batch_size,
        show_progress=not parsed.hide_progress,
        force_reencode=parsed.force_reencode,
    )

    pipeline = MultispectralRetrievalPipeline(
        sigma=parsed.sigma,
        tau=parsed.tau,
        num_steps=parsed.num_steps,
        lr=parsed.lr,
        lambda_m=parsed.lambda_m,
        k=parsed.k,
        grad_clip=parsed.grad_clip,
        early_stop_tol=parsed.early_stop_tol,
    )

    query_fused = fuse_band_embeddings(
        query_payload["band_embeddings"],
        pipeline,
        query_text_features=class_text_features,
        sample_labels=query_payload["label"],
        desc="Fusing matched-protocol queries",
        show_progress=not parsed.hide_progress,
    )
    gallery_fused = fuse_band_embeddings(
        gallery_payload["band_embeddings"],
        pipeline,
        query_text_features=class_text_features,
        sample_labels=gallery_payload["label"],
        desc="Fusing matched-protocol gallery",
        show_progress=not parsed.hide_progress,
    )

    metrics, _, _, _, per_query_rows = evaluate_text_to_image_retrieval(
        query_features=query_fused["features"],
        query_labels=query_payload["label"],
        gallery_features=gallery_fused["features"],
        gallery_labels=gallery_payload["label"],
        ks=(1, 5, 10),
    )

    for row, path, label_name, dataset_index in zip(
        per_query_rows,
        query_payload["path"],
        query_payload["label_name"],
        query_payload["index"].tolist(),
    ):
        row["path"] = path
        row["label_name"] = label_name
        row["dataset_index"] = int(dataset_index)

    summary_row: Dict[str, Any] = {
        "dataset": "EuroSAT_MS",
        "task": "image_to_image_single_label",
        "method": "Ours (13-band pipeline)",
        "protocol": "EuroSAT 80/10/10 matched protocol",
        "query_split": "val",
        "gallery_split": "test",
        "seed": int(parsed.seed),
        "num_train": int(len(bundle["indices"]["train"])),
        "num_queries": int(query_payload["label"].numel()),
        "num_gallery": int(gallery_payload["label"].numel()),
        "clip_checkpoint": str(parsed.clip_checkpoint),
        "summary_csv": str(summary_csv),
        "per_query_csv": str(per_query_csv),
        "split_manifest_csv": str(split_manifest_csv),
        "query_band_cache": str(query_cache),
        "gallery_band_cache": str(gallery_cache),
        "query_proxy": "class_text_embedding",
        "sigma": float(parsed.sigma),
        "tau": parsed.tau,
        "num_steps": int(parsed.num_steps),
        "lr": float(parsed.lr),
        "lambda_m": float(parsed.lambda_m),
        "k": int(parsed.k),
        "grad_clip": float(parsed.grad_clip),
        "early_stop_tol": float(parsed.early_stop_tol),
        "avg_query_fusion_ms": sum(query_fused["elapsed_ms"]) / max(len(query_fused["elapsed_ms"]), 1),
        "avg_query_optimization_ms": (
            sum(query_fused["optimization_ms"]) / max(len(query_fused["optimization_ms"]), 1)
        ),
        "avg_gallery_fusion_ms": (
            sum(gallery_fused["elapsed_ms"]) / max(len(gallery_fused["elapsed_ms"]), 1)
        ),
        "avg_gallery_optimization_ms": (
            sum(gallery_fused["optimization_ms"]) / max(len(gallery_fused["optimization_ms"]), 1)
        ),
        **{name: float(value) for name, value in metrics.items()},
    }

    comparison_rows = build_comparison_rows(
        ours_summary=summary_row,
        pca_csv=pca_csv,
        ndvi_csv=ndvi_csv,
        rs_csv=rs_csv,
    )

    save_csv_rows([summary_row], summary_csv)
    save_csv_rows(per_query_rows, per_query_csv)
    save_csv_rows(comparison_rows, comparison_csv)

    write_json(
        {
            "protocol": {
                "dataset": "EuroSAT_MS",
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "query_split": "val",
                "gallery_split": "test",
                "seed": int(parsed.seed),
            },
            "pipeline": {
                "sigma": float(parsed.sigma),
                "tau": float(parsed.tau) if parsed.tau is not None else None,
                "num_steps": int(parsed.num_steps),
                "lr": float(parsed.lr),
                "lambda_m": float(parsed.lambda_m),
                "k": int(parsed.k),
                "grad_clip": float(parsed.grad_clip),
                "early_stop_tol": float(parsed.early_stop_tol),
                "query_proxy": "class_text_embedding",
            },
            "artifacts": {
                "summary_csv": str(summary_csv),
                "per_query_csv": str(per_query_csv),
                "split_manifest_csv": str(split_manifest_csv),
                "comparison_csv": str(comparison_csv),
                "query_band_cache": str(query_cache),
                "gallery_band_cache": str(gallery_cache),
                "baseline_pca_csv": str(pca_csv),
                "baseline_ndvi_csv": str(ndvi_csv),
                "baseline_rs_csv": str(rs_csv),
            },
            "summary": summary_row,
        },
        manifest_json,
    )

    print("\nMatched-protocol summary:")
    for metric_name in ("R@1", "R@5", "R@10", "mAP"):
        print(f"  {metric_name}: {summary_row[metric_name]:.2f}%")

    print(f"\nSaved summary CSV to {summary_csv}")
    print(f"Saved per-query CSV to {per_query_csv}")
    print(f"Saved split manifest to {split_manifest_csv}")
    print(f"Saved comparison CSV to {comparison_csv}")
    print(f"Saved manifest JSON to {manifest_json}")


if __name__ == "__main__":
    main()
