#!/usr/bin/env python3
"""Run Week 7 component ablations on the EuroSAT matched protocol.

The script evaluates the incremental configurations requested in
``WEEKLY_TASKS.md``:

    1. RGB CLIP baseline
    2. Mean per-band fusion
    3. Query-alignment affinity weights
    4. Fiedler weights without manifold optimization
    5. Full Fiedler + manifold optimization pipeline

Outputs are written under ``results/ablation/`` by default:

    - component_ablation_table.csv
    - component_ablation_waterfall.png
    - component_contribution_analysis.txt
    - component_ablation_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


os.environ.setdefault("XDG_CACHE_HOME", "/tmp/acivs_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/acivs_matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.eurosat import build_eurosat_dataloaders
from src.models.affinity_graph import compute_affinity_graph, compute_query_weights
from src.models.clip_utils import preprocess_rgb_for_clip
from src.models.fiedler import compute_fiedler_magnitude_weights
from src.models.manifold import compute_fused_embedding
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline
from src.utils.metrics import evaluate_text_to_image_retrieval
from src.utils.shared import get_device, load_openai_clip_model, save_csv_rows


BAND_NAMES: Sequence[str] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)


def write_json(payload: Dict[str, Any], output_path: Path) -> None:
    """Write JSON with deterministic formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


@torch.no_grad()
def encode_class_text_features(
    clip_model,
    class_prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode EuroSAT class prompts with CLIP's text encoder."""
    import clip as _clip_pkg

    tokens = _clip_pkg.tokenize(list(class_prompts)).to(device)
    text_features = clip_model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).cpu()


def load_band_cache(cache_path: Path, *, seed: int, split_name: str) -> Dict[str, Any]:
    """Load and validate a matched-protocol per-band embedding cache."""
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

    required = ("band_embeddings", "label", "index", "label_name", "path")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {cache_path}: {missing}")

    embeddings = payload["band_embeddings"]
    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected band embeddings [N,B,D] in {cache_path}, got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[1] != len(BAND_NAMES):
        raise ValueError(
            f"Expected {len(BAND_NAMES)} bands in {cache_path}, got {embeddings.shape[1]}"
        )
    return payload


def limit_payload(payload: Dict[str, Any], max_items: Optional[int]) -> Dict[str, Any]:
    """Return a shallow payload copy restricted to the first ``max_items`` rows."""
    if max_items is None:
        return payload
    if max_items <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")

    n = min(max_items, int(payload["label"].shape[0]))
    limited: Dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.shape[:1] == payload["label"].shape[:1]:
            limited[key] = value[:n]
        elif isinstance(value, list) and len(value) == int(payload["label"].shape[0]):
            limited[key] = value[:n]
        else:
            limited[key] = value
    return limited


def validate_against_split_manifest(
    payload: Dict[str, Any],
    *,
    split_manifest_csv: Path,
    split_name: str,
) -> None:
    """Check that cache order matches the saved matched-protocol split manifest."""
    if not split_manifest_csv.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_csv}")

    rows: List[Dict[str, str]] = []
    with split_manifest_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == split_name:
                rows.append(row)

    n = int(payload["label"].shape[0])
    if len(rows) < n:
        raise ValueError(
            f"Split manifest has {len(rows)} rows for {split_name}, but cache has {n}"
        )

    manifest_indices = [int(row["dataset_index"]) for row in rows[:n]]
    manifest_labels = [int(row["label"]) for row in rows[:n]]
    cache_indices = payload["index"].cpu().tolist()
    cache_labels = payload["label"].cpu().tolist()

    if manifest_indices != cache_indices:
        raise ValueError(f"{split_name} cache indices do not match split manifest order")
    if manifest_labels != cache_labels:
        raise ValueError(f"{split_name} cache labels do not match split manifest labels")


def build_split_loader(
    bundle: Dict[str, Any],
    split_name: str,
    *,
    max_items: Optional[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Return a deterministic query/gallery loader, optionally limited for smoke tests."""
    subset = bundle["subsets"][split_name]
    if max_items is not None:
        subset = Subset(subset, list(range(min(max_items, len(subset)))))

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


@torch.inference_mode()
def encode_rgb_split(
    loader: DataLoader,
    clip_model,
    device: torch.device,
    *,
    image_size: int,
    rgb_indices: Sequence[int],
    desc: str,
    show_progress: bool,
) -> Dict[str, Any]:
    """Encode one EuroSAT split using the standard RGB CLIP baseline."""
    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    index_chunks: List[torch.Tensor] = []
    label_names: List[str] = []
    paths: List[str] = []

    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader
    clip_model.eval()

    for batch in iterator:
        image = batch["image"].to(torch.float32)
        rgb = image[:, list(rgb_indices), :, :]
        clip_inputs = preprocess_rgb_for_clip(rgb, image_size=image_size).to(device)
        features = clip_model.encode_image(clip_inputs).float()
        feature_chunks.append(F.normalize(features, dim=-1).cpu())

        label_chunks.append(batch["label"].cpu())
        batch_indices = batch["index"]
        if torch.is_tensor(batch_indices):
            index_chunks.append(batch_indices.cpu())
        else:
            index_chunks.append(torch.tensor(batch_indices, dtype=torch.long))
        label_names.extend([str(name) for name in batch["label_name"]])
        paths.extend([str(path) for path in batch["path"]])

    return {
        "features": torch.cat(feature_chunks, dim=0),
        "label": torch.cat(label_chunks, dim=0),
        "index": torch.cat(index_chunks, dim=0),
        "label_name": label_names,
        "path": paths,
    }


def load_rgb_cache(
    cache_path: Path,
    *,
    seed: int,
    split_name: str,
    image_size: int,
    rgb_indices: Sequence[int],
) -> Dict[str, Any]:
    """Load a cached RGB feature split."""
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("seed") != seed or payload.get("split_name") != split_name:
        raise ValueError(f"RGB cache metadata mismatch for {cache_path}")
    if int(payload.get("image_size")) != int(image_size):
        raise ValueError(f"RGB cache image_size mismatch for {cache_path}")
    if tuple(payload.get("rgb_indices")) != tuple(rgb_indices):
        raise ValueError(f"RGB cache rgb_indices mismatch for {cache_path}")
    return payload


def save_rgb_cache(
    payload: Dict[str, Any],
    cache_path: Path,
    *,
    seed: int,
    split_name: str,
    image_size: int,
    rgb_indices: Sequence[int],
) -> None:
    """Persist RGB features for faster reruns."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "split_name": split_name,
        "seed": int(seed),
        "image_size": int(image_size),
        "rgb_indices": tuple(int(idx) for idx in rgb_indices),
        "num_items": int(payload["label"].shape[0]),
        **payload,
    }
    torch.save(serializable, cache_path)


def maybe_load_or_encode_rgb_split(
    *,
    cache_path: Path,
    split_name: str,
    seed: int,
    loader: DataLoader,
    clip_model,
    device: torch.device,
    image_size: int,
    rgb_indices: Sequence[int],
    show_progress: bool,
    force_reencode: bool,
) -> Dict[str, Any]:
    """Reuse cached RGB features when possible, otherwise encode raw images."""
    requested_count = len(loader.dataset)
    if cache_path.exists() and not force_reencode:
        try:
            cached = load_rgb_cache(
                cache_path,
                seed=seed,
                split_name=split_name,
                image_size=image_size,
                rgb_indices=rgb_indices,
            )
            if int(cached["label"].shape[0]) >= requested_count:
                return limit_payload(cached, requested_count)
        except (KeyError, ValueError):
            pass

    payload = encode_rgb_split(
        loader,
        clip_model,
        device,
        image_size=image_size,
        rgb_indices=rgb_indices,
        desc=f"Encoding RGB-CLIP {split_name}",
        show_progress=show_progress,
    )
    save_rgb_cache(
        payload,
        cache_path,
        seed=seed,
        split_name=split_name,
        image_size=image_size,
        rgb_indices=rgb_indices,
    )
    return payload


def fuse_band_payload(
    payload: Dict[str, Any],
    *,
    mode: str,
    class_text_features: torch.Tensor,
    sigma: float,
    pipeline: Optional[MultispectralRetrievalPipeline],
    show_progress: bool,
    desc: str,
) -> Dict[str, Any]:
    """Fuse cached per-band embeddings for one ablation mode."""
    band_embeddings = F.normalize(payload["band_embeddings"].cpu().float(), dim=-1)
    labels = payload["label"].cpu().long()
    class_text_features = F.normalize(class_text_features.cpu().float(), dim=-1)

    fused_features: List[torch.Tensor] = []
    weight_rows: List[torch.Tensor] = []
    elapsed_ms: List[float] = []

    iterator: Iterable[int] = range(band_embeddings.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc=desc)

    for idx in iterator:
        bands = band_embeddings[idx]
        query_embedding = class_text_features[int(labels[idx].item())]
        start = time.perf_counter()

        if mode == "mean":
            weights = torch.full(
                (bands.shape[0],),
                1.0 / float(bands.shape[0]),
                dtype=bands.dtype,
            )
            fused = compute_fused_embedding(bands, weights, normalize_output=True)
        elif mode == "affinity":
            weights = compute_query_weights(
                band_embeddings=bands,
                query_embedding=query_embedding,
                sigma=sigma,
                normalize_inputs=True,
            )
            fused = compute_fused_embedding(bands, weights, normalize_output=True)
        elif mode == "fiedler":
            affinity = compute_affinity_graph(
                band_embeddings=bands,
                query_embedding=query_embedding,
                sigma=sigma,
                normalize_inputs=True,
                return_details=False,
            )
            weights = compute_fiedler_magnitude_weights(affinity, normalized=True)
            fused = compute_fused_embedding(bands, weights, normalize_output=True)
        elif mode == "full":
            if pipeline is None:
                raise ValueError("pipeline is required for mode='full'")
            result = pipeline.retrieve(
                band_embeddings=bands,
                query_embedding=query_embedding,
            )
            weights = result.weights
            fused = result.fused_embedding
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        fused_features.append(fused.detach().cpu())
        weight_rows.append(weights.detach().cpu())

    weights_stacked = torch.stack(weight_rows, dim=0)
    mean_weights = weights_stacked.mean(dim=0)
    entropy = -(weights_stacked.clamp_min(1e-12) * weights_stacked.clamp_min(1e-12).log())
    entropy = entropy.sum(dim=1).mean().item()

    return {
        "features": torch.stack(fused_features, dim=0),
        "mean_ms": sum(elapsed_ms) / max(len(elapsed_ms), 1),
        "mean_weights": mean_weights,
        "mean_weight_entropy": float(entropy),
    }


def evaluate_config(
    *,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate one ablation configuration with the canonical single-label metric."""
    metrics, _, _, _, _ = evaluate_text_to_image_retrieval(
        query_features=F.normalize(query_features.float(), dim=-1),
        query_labels=query_labels.long(),
        gallery_features=F.normalize(gallery_features.float(), dim=-1),
        gallery_labels=gallery_labels.long(),
        ks=(1, 5, 10),
    )
    return {name: float(value) for name, value in metrics.items()}


def add_delta_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add cumulative and incremental deltas relative to the RGB baseline."""
    baseline_r1 = float(rows[0]["R@1"])
    baseline_r10 = float(rows[0]["R@10"])
    previous_r1 = baseline_r1
    previous_r10 = baseline_r10

    for row in rows:
        r1 = float(row["R@1"])
        r10 = float(row["R@10"])
        row["delta_R@1_vs_rgb"] = r1 - baseline_r1
        row["delta_R@10_vs_rgb"] = r10 - baseline_r10
        row["incremental_delta_R@1"] = r1 - previous_r1
        row["incremental_delta_R@10"] = r10 - previous_r10
        row["redundant_by_R@1"] = bool(row["stage_order"] > 1 and (r1 - previous_r1) <= 0.0)
        previous_r1 = r1
        previous_r10 = r10
    return rows


def save_waterfall_plot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Save a step-by-step R@1 contribution waterfall plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [textwrap.fill(str(row["component"]), width=16) for row in rows]
    r1_values = [float(row["R@1"]) for row in rows]
    increments = [float(row["incremental_delta_R@1"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    x = list(range(len(rows)))

    ax.bar(x[0], r1_values[0], color="#5B677A", width=0.68, label="RGB baseline")
    for idx in range(1, len(rows)):
        bottom = r1_values[idx - 1] if increments[idx] >= 0 else r1_values[idx]
        height = abs(increments[idx])
        color = "#2A9D8F" if increments[idx] >= 0 else "#C44E52"
        ax.bar(x[idx], height, bottom=bottom, color=color, width=0.68)
        ax.plot([idx - 1, idx], [r1_values[idx - 1], r1_values[idx - 1]], color="#A0A7B2", lw=1)

    for idx, value in enumerate(r1_values):
        ax.text(idx, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        if idx > 0:
            sign = "+" if increments[idx] >= 0 else ""
            ax.text(
                idx,
                min(r1_values[idx], r1_values[idx - 1]) + abs(increments[idx]) / 2.0,
                f"{sign}{increments[idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    y_min = max(0.0, min(r1_values) - 5.0)
    y_max = min(100.0, max(r1_values) + 6.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Component Ablation: Incremental R@1 Contribution")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Create the requested markdown table for the text analysis artifact."""
    lines = [
        "| Component | R@1 | R@10 | Delta R@1 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {component} | {r1:.2f} | {r10:.2f} | {delta:+.2f} |".format(
                component=row["component"],
                r1=float(row["R@1"]),
                r10=float(row["R@10"]),
                delta=float(row["delta_R@1_vs_rgb"]),
            )
        )
    return "\n".join(lines)


def save_contribution_analysis(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write a concise component contribution analysis."""
    full_delta_r1 = float(rows[-1]["delta_R@1_vs_rgb"])
    full_delta_r10 = float(rows[-1]["delta_R@10_vs_rgb"])
    redundant = [row for row in rows if row.get("redundant_by_R@1")]
    monotonic = len(redundant) == 0

    lines = [
        "# Component Contribution Analysis",
        "",
        "Protocol: EuroSAT-MS 80/10/10 matched protocol, val queries vs test gallery.",
        "Conditioning: each non-RGB configuration uses the CLIP text embedding of the sample class for Eq. 1.",
        "The values below are measured from local artifacts, not injected from the planning targets.",
        "",
        format_markdown_table(rows),
        "",
        f"Cumulative R@1 delta: {full_delta_r1:+.2f} percentage points.",
        f"Cumulative R@10 delta: {full_delta_r10:+.2f} percentage points.",
        f"Monotonic R@1 increase: {'yes' if monotonic else 'no'}.",
        "",
        "Incremental R@1 contributions:",
    ]

    for row in rows[1:]:
        lines.append(
            "- {component}: {delta:+.2f} pp".format(
                component=row["component"],
                delta=float(row["incremental_delta_R@1"]),
            )
        )

    lines.append("")
    if redundant:
        lines.append("Components with non-positive incremental R@1 in this run:")
        for row in redundant:
            lines.append(
                "- {component}: {delta:+.2f} pp".format(
                    component=row["component"],
                    delta=float(row["incremental_delta_R@1"]),
                )
            )
    else:
        lines.append("No component is redundant by the monotonic R@1 criterion in this run.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Week 7 component ablations on the EuroSAT matched protocol."
    )
    parser.add_argument("--eurosat-root", type=Path, default=Path("data/EuroSAT_MS"))
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("checkpoints/ViT-B-16.pt"))
    parser.add_argument("--matched-dir", type=Path, default=Path("results/matched_protocol"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/ablation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rgb-indices", type=int, nargs=3, default=(3, 2, 1))
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-m", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-tol", type=float, default=1e-6)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-gallery", type=int, default=None)
    parser.add_argument("--force-rgb-reencode", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)

    parsed.results_dir.mkdir(parents=True, exist_ok=True)
    table_csv = parsed.results_dir / "component_ablation_table.csv"
    waterfall_png = parsed.results_dir / "component_ablation_waterfall.png"
    analysis_txt = parsed.results_dir / "component_contribution_analysis.txt"
    manifest_json = parsed.results_dir / "component_ablation_manifest.json"
    rgb_query_cache = parsed.results_dir / "component_ablation_rgb_query_features.pt"
    rgb_gallery_cache = parsed.results_dir / "component_ablation_rgb_gallery_features.pt"

    query_cache = parsed.matched_dir / "eurosat_matched_protocol_query_band_embeddings.pt"
    gallery_cache = parsed.matched_dir / "eurosat_matched_protocol_gallery_band_embeddings.pt"
    split_manifest_csv = parsed.matched_dir / "eurosat_matched_protocol_split_manifest.csv"

    query_payload = limit_payload(
        load_band_cache(query_cache, seed=parsed.seed, split_name="val"),
        parsed.max_queries,
    )
    gallery_payload = limit_payload(
        load_band_cache(gallery_cache, seed=parsed.seed, split_name="test"),
        parsed.max_gallery,
    )
    validate_against_split_manifest(
        query_payload,
        split_manifest_csv=split_manifest_csv,
        split_name="val",
    )
    validate_against_split_manifest(
        gallery_payload,
        split_manifest_csv=split_manifest_csv,
        split_name="test",
    )

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
    dataset = bundle["dataset"]
    class_text_features = encode_class_text_features(
        clip_model,
        dataset.get_text_prompts(),
        device,
    )

    query_loader = build_split_loader(
        bundle,
        "val",
        max_items=parsed.max_queries,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
    )
    gallery_loader = build_split_loader(
        bundle,
        "test",
        max_items=parsed.max_gallery,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
    )

    rgb_query = maybe_load_or_encode_rgb_split(
        cache_path=rgb_query_cache,
        split_name="val",
        seed=parsed.seed,
        loader=query_loader,
        clip_model=clip_model,
        device=device,
        image_size=parsed.image_size,
        rgb_indices=parsed.rgb_indices,
        show_progress=not parsed.hide_progress,
        force_reencode=parsed.force_rgb_reencode,
    )
    rgb_gallery = maybe_load_or_encode_rgb_split(
        cache_path=rgb_gallery_cache,
        split_name="test",
        seed=parsed.seed,
        loader=gallery_loader,
        clip_model=clip_model,
        device=device,
        image_size=parsed.image_size,
        rgb_indices=parsed.rgb_indices,
        show_progress=not parsed.hide_progress,
        force_reencode=parsed.force_rgb_reencode,
    )

    pipeline = MultispectralRetrievalPipeline(
        sigma=parsed.sigma,
        num_steps=parsed.num_steps,
        lr=parsed.lr,
        lambda_m=parsed.lambda_m,
        k=parsed.k,
        grad_clip=parsed.grad_clip,
        early_stop_tol=parsed.early_stop_tol,
    )

    rows: List[Dict[str, Any]] = []

    print("Evaluating RGB baseline...")
    rgb_metrics = evaluate_config(
        query_features=rgb_query["features"],
        query_labels=rgb_query["label"],
        gallery_features=rgb_gallery["features"],
        gallery_labels=rgb_gallery["label"],
    )
    rows.append(
        {
            "stage_order": 1,
            "component": "No per-band (RGB baseline)",
            "configuration": "RGB-CLIP image-to-image",
            "query_split": "val",
            "gallery_split": "test",
            "num_queries": int(rgb_query["label"].numel()),
            "num_gallery": int(rgb_gallery["label"].numel()),
            "fusion_mean_ms_query": 0.0,
            "fusion_mean_ms_gallery": 0.0,
            "mean_weight_entropy": "",
            **rgb_metrics,
        }
    )

    fusion_specs = [
        ("mean", "+ Per-band encode", "Uniform mean of 13 per-band CLIP embeddings"),
        ("affinity", "+ Affinity graph", "Query-alignment softmax band weights"),
        ("fiedler", "+ Fiedler", "Fiedler magnitude weights, no manifold optimization"),
        ("full", "+ Manifold optimization", "Full Fiedler + manifold optimization pipeline"),
    ]

    for stage_idx, (mode, component, configuration) in enumerate(fusion_specs, start=2):
        query_fused = fuse_band_payload(
            query_payload,
            mode=mode,
            class_text_features=class_text_features,
            sigma=parsed.sigma,
            pipeline=pipeline if mode == "full" else None,
            show_progress=not parsed.hide_progress,
            desc=f"Fusing queries: {component}",
        )
        gallery_fused = fuse_band_payload(
            gallery_payload,
            mode=mode,
            class_text_features=class_text_features,
            sigma=parsed.sigma,
            pipeline=pipeline if mode == "full" else None,
            show_progress=not parsed.hide_progress,
            desc=f"Fusing gallery: {component}",
        )
        metrics = evaluate_config(
            query_features=query_fused["features"],
            query_labels=query_payload["label"],
            gallery_features=gallery_fused["features"],
            gallery_labels=gallery_payload["label"],
        )
        rows.append(
            {
                "stage_order": stage_idx,
                "component": component,
                "configuration": configuration,
                "query_split": "val",
                "gallery_split": "test",
                "num_queries": int(query_payload["label"].numel()),
                "num_gallery": int(gallery_payload["label"].numel()),
                "fusion_mean_ms_query": float(query_fused["mean_ms"]),
                "fusion_mean_ms_gallery": float(gallery_fused["mean_ms"]),
                "mean_weight_entropy": float(
                    (query_fused["mean_weight_entropy"] + gallery_fused["mean_weight_entropy"]) / 2.0
                ),
                **metrics,
            }
        )

    rows = add_delta_columns(rows)
    save_csv_rows(rows, table_csv)
    save_waterfall_plot(rows, waterfall_png)
    save_contribution_analysis(rows, analysis_txt)

    write_json(
        {
            "protocol": {
                "dataset": "EuroSAT_MS",
                "split": "80/10/10 matched protocol",
                "query_split": "val",
                "gallery_split": "test",
                "seed": int(parsed.seed),
                "max_queries": parsed.max_queries,
                "max_gallery": parsed.max_gallery,
            },
            "pipeline": {
                "sigma": float(parsed.sigma),
                "num_steps": int(parsed.num_steps),
                "lr": float(parsed.lr),
                "lambda_m": float(parsed.lambda_m),
                "k": int(parsed.k),
                "grad_clip": float(parsed.grad_clip),
                "early_stop_tol": float(parsed.early_stop_tol),
                "query_conditioning": "class_text_embedding",
            },
            "artifacts": {
                "table_csv": str(table_csv),
                "waterfall_png": str(waterfall_png),
                "analysis_txt": str(analysis_txt),
                "query_band_cache": str(query_cache),
                "gallery_band_cache": str(gallery_cache),
                "rgb_query_cache": str(rgb_query_cache),
                "rgb_gallery_cache": str(rgb_gallery_cache),
            },
            "rows": rows,
        },
        manifest_json,
    )

    print("\nComponent ablation results:")
    for row in rows:
        print(
            "  {component}: R@1={r1:.2f} R@10={r10:.2f} delta_R@1={delta:+.2f}".format(
                component=row["component"],
                r1=float(row["R@1"]),
                r10=float(row["R@10"]),
                delta=float(row["delta_R@1_vs_rgb"]),
            )
        )
    print(f"\nSaved table to {table_csv}")
    print(f"Saved plot to {waterfall_png}")
    print(f"Saved analysis to {analysis_txt}")
    print(f"Saved manifest to {manifest_json}")


if __name__ == "__main__":
    main()
