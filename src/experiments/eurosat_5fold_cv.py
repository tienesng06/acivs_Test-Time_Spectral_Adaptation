from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.baselines.ndvi_baseline import encode_loader_with_spectral_indices
from src.baselines.pca_baseline import (
    encode_loader_with_pca,
    fit_global_pca_from_loader,
    load_openai_clip_model,
)
from src.baselines.rs_transclip_baseline import (
    build_gallery_patch_affinity_knn,
    encode_loader_with_rgb_patches,
    evaluate_single_label_retrieval_from_similarity,
    refine_similarity_matrix,
)
from src.datasets.eurosat import EuroSATMSDataset, get_band_index, get_default_text_queries
from src.models.clip_utils import encode_test_gallery_rgb
from src.models.per_band_encoder import encode_multispectral_batch, get_device
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline


DEFAULT_LOCAL_METHODS: tuple[str, ...] = (
    "RGB-CLIP",
    "PCA",
    "NDVI",
    "Tip-Adapter",
    "RS-TransCLIP",
    "Ours",
)
DEFAULT_EXTERNAL_METHODS: tuple[str, ...] = ("DOFA",)
DEFAULT_ALL_METHODS: tuple[str, ...] = DEFAULT_LOCAL_METHODS + DEFAULT_EXTERNAL_METHODS
DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


def save_csv_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _finalize_metadata(metadata: Dict[str, List[Any]]) -> Dict[str, Any]:
    finalized: Dict[str, Any] = {}
    for key, values in metadata.items():
        if not values:
            finalized[key] = []
        elif torch.is_tensor(values[0]):
            finalized[key] = torch.cat(values, dim=0)
        else:
            flat: List[Any] = []
            for value in values:
                if isinstance(value, (list, tuple)):
                    flat.extend(list(value))
                else:
                    flat.append(value)
            finalized[key] = flat
    return finalized


def _build_loader(
    dataset: EuroSATMSDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )


def cap_indices_per_class(
    dataset: EuroSATMSDataset,
    indices: Sequence[int],
    *,
    max_per_class: Optional[int],
    seed: int,
) -> List[int]:
    if max_per_class is None or max_per_class <= 0:
        return list(indices)

    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[int]] = {class_idx: [] for class_idx in range(len(dataset.class_names))}
    for index in indices:
        label = int(dataset.samples[int(index)]["label"])
        buckets[label].append(int(index))

    capped: List[int] = []
    for class_idx in range(len(dataset.class_names)):
        class_indices = buckets[class_idx]
        if len(class_indices) <= max_per_class:
            capped.extend(class_indices)
            continue
        capped.extend(rng.permutation(class_indices)[:max_per_class].tolist())

    return sorted(capped)


def make_stratified_kfold_indices(
    labels: Sequence[int],
    *,
    num_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    unique_labels = sorted(np.unique(labels_np).tolist())

    fold_buckets: List[List[int]] = [[] for _ in range(num_folds)]
    all_indices = np.arange(labels_np.shape[0])

    for label in unique_labels:
        class_indices = all_indices[labels_np == label]
        class_indices = rng.permutation(class_indices)
        class_splits = np.array_split(class_indices, num_folds)
        for fold_id, chunk in enumerate(class_splits):
            fold_buckets[fold_id].extend(chunk.tolist())

    fold_splits: List[Dict[str, List[int]]] = []
    full_index_set = set(all_indices.tolist())

    for fold_id, gallery_indices in enumerate(fold_buckets):
        gallery_indices = rng.permutation(gallery_indices).tolist()
        train_indices = sorted(full_index_set.difference(gallery_indices))
        fold_splits.append(
            {
                "fold_id": fold_id,
                "train": train_indices,
                "gallery": gallery_indices,
            }
        )

    return fold_splits


def build_fold_manifest_rows(
    dataset: EuroSATMSDataset,
    fold_splits: Sequence[Mapping[str, Sequence[int]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for fold in fold_splits:
        fold_id = int(fold["fold_id"])
        train_labels = torch.tensor(
            [dataset.samples[idx]["label"] for idx in fold["train"]],
            dtype=torch.long,
        )
        gallery_labels = torch.tensor(
            [dataset.samples[idx]["label"] for idx in fold["gallery"]],
            dtype=torch.long,
        )

        for class_idx, class_name in enumerate(dataset.class_names):
            rows.append(
                {
                    "fold_id": fold_id,
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "train_count": int((train_labels == class_idx).sum().item()),
                    "gallery_count": int((gallery_labels == class_idx).sum().item()),
                }
            )

    return rows


@torch.no_grad()
def encode_class_text_features(
    clip_model,
    clip_tokenize,
    *,
    class_prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    tokens = clip_tokenize(list(class_prompts)).to(device)
    text_features = clip_model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).cpu()


@torch.inference_mode()
def encode_loader_with_band_embeddings(
    loader: DataLoader,
    clip_model,
    device: torch.device,
    *,
    micro_batch_size: int = 64,
    desc: str,
    show_progress: bool,
) -> Dict[str, Any]:
    clip_model.eval()

    band_embedding_chunks: List[torch.Tensor] = []
    metadata_store: Dict[str, List[Any]] = {
        "label": [],
        "label_name": [],
        "path": [],
        "index": [],
    }

    iterator: Iterable[Any] = tqdm(loader, desc=desc) if show_progress else loader
    for batch in iterator:
        band_embeddings = encode_multispectral_batch(
            batch["image"],
            clip_model,
            device=device,
            micro_batch_size=micro_batch_size,
        )
        band_embedding_chunks.append(band_embeddings.cpu())

        for key in metadata_store:
            value = batch[key]
            if torch.is_tensor(value):
                metadata_store[key].append(value.cpu())
            else:
                metadata_store[key].append(value)

    result = {
        "band_embeddings": torch.cat(band_embedding_chunks, dim=0),
    }
    result.update(_finalize_metadata(metadata_store))
    return result


def _decorate_per_query_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    fold_id: int,
    method: str,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    decorated: List[Dict[str, Any]] = []
    for row in rows:
        q_label = int(row["query_label"])
        decorated.append(
            {
                "method": method,
                "fold_id": fold_id,
                "query_label": q_label,
                "class_name": class_names[q_label],
                **row,
            }
        )
    return decorated


def evaluate_class_query_similarity(
    similarity_matrix: torch.Tensor,
    *,
    gallery_labels: torch.Tensor,
    class_names: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Any]:
    query_labels = torch.arange(len(class_names), dtype=torch.long)
    metrics, ranked_indices, ranked_relevance, per_query_rows = (
        evaluate_single_label_retrieval_from_similarity(
            similarity_matrix,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            ks=ks,
        )
    )

    return {
        "metrics": metrics,
        "ranked_indices": ranked_indices,
        "ranked_relevance": ranked_relevance,
        "per_query_rows": per_query_rows,
    }


def build_tip_adapter_similarity_matrix(
    *,
    text_features: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    num_classes: int,
    alpha: float,
    beta: float,
    chunk_size: int = 256,
    compute_device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    device = compute_device or torch.device("cpu")
    text_features = F.normalize(text_features.float(), dim=-1).to(device)
    train_features = F.normalize(train_features.float(), dim=-1).to(device)
    gallery_features = F.normalize(gallery_features.float(), dim=-1).to(device)
    train_labels = train_labels.long().to(device)

    cache_values = F.one_hot(train_labels, num_classes=num_classes).float()
    logits_chunks: List[torch.Tensor] = []

    iterator = range(0, gallery_features.shape[0], chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Tip-Adapter scoring")

    for start in iterator:
        stop = min(start + chunk_size, gallery_features.shape[0])
        gallery_chunk = gallery_features[start:stop]

        clip_logits = gallery_chunk @ text_features.T
        affinity = gallery_chunk @ train_features.T
        cache_logits = torch.exp(-beta * (1.0 - affinity)) @ cache_values
        final_logits = clip_logits + alpha * cache_logits
        logits_chunks.append(final_logits.cpu())

    gallery_logits = torch.cat(logits_chunks, dim=0)
    return gallery_logits.T.contiguous()


def build_ours_similarity_matrix(
    *,
    text_features: torch.Tensor,
    gallery_band_embeddings: torch.Tensor,
    pipeline: MultispectralRetrievalPipeline,
    compute_device: Optional[torch.device],
    show_progress: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    device = compute_device or torch.device("cpu")
    band_embeddings = gallery_band_embeddings.to(device)
    normalized_text = F.normalize(text_features.float(), dim=-1).to(device)

    similarity_rows: List[torch.Tensor] = []
    mean_ms_rows: List[float] = []

    iterator: Iterable[Any] = enumerate(normalized_text)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=normalized_text.shape[0],
            desc="Ours query-conditioned fusion",
        )

    for _, query_embedding in iterator:
        batch_result = pipeline.retrieve_batch(
            band_embeddings,
            query_embedding,
        )
        fused_embeddings = F.normalize(batch_result.fused_embeddings.float(), dim=-1)
        similarity = query_embedding.view(1, -1) @ fused_embeddings.T
        similarity_rows.append(similarity.cpu())
        mean_ms_rows.append(float(batch_result.mean_ms))

    stats = {
        "avg_gallery_fusion_ms_per_query": float(np.mean(mean_ms_rows)) if mean_ms_rows else math.nan,
        "std_gallery_fusion_ms_per_query": float(np.std(mean_ms_rows, ddof=0)) if mean_ms_rows else math.nan,
    }
    return torch.cat(similarity_rows, dim=0), stats


def safe_paired_ttest(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if x_arr.size < 2:
        return {
            "n_pairs": float(x_arr.size),
            "t_stat": math.nan,
            "p_value": math.nan,
        }

    if np.allclose(x_arr, y_arr):
        return {
            "n_pairs": float(x_arr.size),
            "t_stat": 0.0,
            "p_value": 1.0,
        }

    t_stat, p_value = ttest_rel(x_arr, y_arr, nan_policy="omit")
    return {
        "n_pairs": float(x_arr.size),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def format_mean_std(mean_value: float, std_value: float) -> str:
    if not np.isfinite(mean_value):
        return ""
    if not np.isfinite(std_value):
        return f"{mean_value:.2f}"
    return f"{mean_value:.2f}±{std_value:.2f}"


def apply_ordered_category(
    df: pd.DataFrame,
    *,
    column: str,
    categories: Sequence[str],
) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df

    result = df.copy()
    result[column] = pd.Categorical(result[column], categories=categories, ordered=True)
    return result


def sort_by_available_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    sort_columns = [column for column in columns if column in df.columns]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns).reset_index(drop=True)


def build_comparison_table(
    fold_metrics_df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("R@1", "R@5", "R@10", "mAP"),
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for method, method_df in fold_metrics_df.groupby("method", sort=False):
        row: Dict[str, Any] = {
            "method": method,
            "num_folds": int(method_df["fold_id"].nunique()),
        }
        for metric in metrics:
            values = method_df[metric].astype(float)
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = format_mean_std(mean_value, std_value)
        rows.append(row)

    return pd.DataFrame(rows)


def build_paired_ttest_table(
    fold_metrics_df: pd.DataFrame,
    *,
    reference_method: str = "Ours",
    metrics: Sequence[str] = ("R@1", "R@10", "mAP"),
) -> pd.DataFrame:
    if fold_metrics_df.empty or reference_method not in set(fold_metrics_df["method"]):
        return pd.DataFrame()

    ref_df = (
        fold_metrics_df[fold_metrics_df["method"] == reference_method]
        .sort_values("fold_id")
        .set_index("fold_id")
    )

    rows: List[Dict[str, Any]] = []
    for method, method_df in fold_metrics_df.groupby("method", sort=False):
        if method == reference_method:
            continue

        method_df = method_df.sort_values("fold_id").set_index("fold_id")
        shared_folds = sorted(set(ref_df.index).intersection(method_df.index))
        if not shared_folds:
            continue

        for metric in metrics:
            test_result = safe_paired_ttest(
                ref_df.loc[shared_folds, metric].tolist(),
                method_df.loc[shared_folds, metric].tolist(),
            )
            rows.append(
                {
                    "reference_method": reference_method,
                    "compared_method": method,
                    "metric": metric,
                    "n_pairs": int(test_result["n_pairs"]),
                    "t_stat": test_result["t_stat"],
                    "p_value": test_result["p_value"],
                    "significant_p_lt_0_05": bool(
                        np.isfinite(test_result["p_value"]) and test_result["p_value"] < 0.05
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_per_class_table(per_query_df: pd.DataFrame) -> pd.DataFrame:
    if per_query_df.empty:
        return pd.DataFrame()

    metric_cols = ["AP_percent", "hit@1", "hit@5", "hit@10"]
    working = per_query_df.copy()
    for column in ("hit@1", "hit@5", "hit@10"):
        working[column] = working[column].astype(float) * 100.0

    rows: List[Dict[str, Any]] = []
    grouped = working.groupby(["method", "query_label", "class_name"], sort=False)
    for (method, query_label, class_name), class_df in grouped:
        row: Dict[str, Any] = {
            "method": method,
            "query_label": int(query_label),
            "class_name": class_name,
            "num_folds": int(class_df["fold_id"].nunique()),
        }
        for column in metric_cols:
            mean_value = float(class_df[column].mean())
            std_value = float(class_df[column].std(ddof=1)) if len(class_df) > 1 else 0.0
            pretty_name = "mAP" if column == "AP_percent" else column.replace("hit@", "R@")
            row[f"{pretty_name}_mean"] = mean_value
            row[f"{pretty_name}_std"] = std_value
            row[f"{pretty_name}_mean_std"] = format_mean_std(mean_value, std_value)
        rows.append(row)

    return pd.DataFrame(rows)


def write_external_method_templates(
    results_dir: Path,
    *,
    external_methods: Sequence[str] = DEFAULT_EXTERNAL_METHODS,
) -> Dict[str, Dict[str, Path]]:
    template_dir = results_dir / "external_templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Dict[str, Path]] = {}
    for method in external_methods:
        summary_path = template_dir / f"{method.lower().replace(' ', '_')}_fold_metrics_template.csv"
        per_query_path = template_dir / f"{method.lower().replace(' ', '_')}_per_query_template.csv"

        if not summary_path.exists():
            save_csv_rows(
                [
                    {
                        "method": method,
                        "fold_id": 0,
                        "R@1": "",
                        "R@5": "",
                        "R@10": "",
                        "mAP": "",
                        "num_train": "",
                        "num_gallery": "",
                        "notes": "Fill one row per fold for external methods.",
                    }
                ],
                summary_path,
            )

        if not per_query_path.exists():
            save_csv_rows(
                [
                    {
                        "method": method,
                        "fold_id": 0,
                        "query_label": 0,
                        "class_name": "AnnualCrop",
                        "AP_percent": "",
                        "first_hit_rank": "",
                        "hit@1": "",
                        "hit@5": "",
                        "hit@10": "",
                        "notes": "Optional but needed for the aggregated per-class table.",
                    }
                ],
                per_query_path,
            )

        paths[method] = {
            "summary_csv": summary_path,
            "per_query_csv": per_query_path,
        }

    return paths


def load_external_method_results(
    external_inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []

    for method, payload in external_inputs.items():
        summary_csv = payload.get("summary_csv")
        per_query_csv = payload.get("per_query_csv")

        if summary_csv:
            df_summary = pd.read_csv(summary_csv)
            if "method" not in df_summary.columns:
                df_summary["method"] = method
            summary_rows.extend(df_summary.to_dict(orient="records"))

        if per_query_csv:
            df_per_query = pd.read_csv(per_query_csv)
            if "method" not in df_per_query.columns:
                df_per_query["method"] = method
            per_query_rows.extend(df_per_query.to_dict(orient="records"))

    return summary_rows, per_query_rows


def run_eurosat_5fold_cv(
    *,
    root: Path | str,
    clip_checkpoint: Path | str,
    results_dir: Path | str = Path("results/eurosat_5fold_cv"),
    methods: Sequence[str] = DEFAULT_ALL_METHODS,
    external_method_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    num_folds: int = 5,
    fold_ids: Optional[Sequence[int]] = None,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    image_size: int = 224,
    micro_batch_size: int = 64,
    show_progress: bool = True,
    max_train_per_class: Optional[int] = None,
    max_gallery_per_class: Optional[int] = None,
    tip_adapter_alpha: float = 20.0,
    tip_adapter_beta: float = 5.5,
    tip_adapter_chunk_size: int = 256,
    rs_transclip_alpha: float = 0.3,
    rs_transclip_patch_pool_size: int = 4,
    rs_transclip_affinity_topk: int = 20,
    rs_transclip_affinity_chunk_size: int = 256,
    ours_sigma: float = 0.5,
    ours_num_steps: int = 5,
    ours_lr: float = 0.01,
    ours_lambda_m: float = 0.1,
    ours_k: int = 5,
    ours_grad_clip: float = 1.0,
    ours_early_stop_tol: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    root = Path(root)
    clip_checkpoint = Path(clip_checkpoint)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = list(methods)
    unknown_methods = sorted(set(methods).difference(DEFAULT_ALL_METHODS))
    if unknown_methods:
        raise ValueError(f"Unsupported methods requested: {unknown_methods}")

    external_templates = write_external_method_templates(results_dir)

    dataset = EuroSATMSDataset(
        root,
        normalize=True,
        reflectance_scale=10_000.0,
        clamp_range=(0.0, 1.0),
    )
    fold_splits = make_stratified_kfold_indices(dataset.get_labels(), num_folds=num_folds, seed=seed)
    if fold_ids is not None:
        requested_fold_ids = {int(fold_id) for fold_id in fold_ids}
        fold_splits = [fold for fold in fold_splits if int(fold["fold_id"]) in requested_fold_ids]
        if not fold_splits:
            raise ValueError(f"No folds left after applying fold_ids={sorted(requested_fold_ids)}")

    capped_fold_splits: List[Dict[str, List[int]]] = []
    for fold in fold_splits:
        fold_id = int(fold["fold_id"])
        capped_fold_splits.append(
            {
                "fold_id": fold_id,
                "train": cap_indices_per_class(
                    dataset,
                    fold["train"],
                    max_per_class=max_train_per_class,
                    seed=seed + fold_id * 17 + 1,
                ),
                "gallery": cap_indices_per_class(
                    dataset,
                    fold["gallery"],
                    max_per_class=max_gallery_per_class,
                    seed=seed + fold_id * 17 + 2,
                ),
            }
        )
    fold_splits = capped_fold_splits
    fold_manifest_rows = build_fold_manifest_rows(dataset, fold_splits)

    active_device = device or get_device()
    local_methods = [method for method in methods if method in DEFAULT_LOCAL_METHODS]
    class_prompts = get_default_text_queries()
    text_features = None
    clip_model = None

    if local_methods:
        clip_model, clip_tokenize = load_openai_clip_model(clip_checkpoint, active_device)
        text_features = encode_class_text_features(
            clip_model,
            clip_tokenize,
            class_prompts=class_prompts,
            device=active_device,
        )
    else:
        clip_tokenize = None

    fold_metric_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []

    for fold in fold_splits:
        fold_id = int(fold["fold_id"])
        train_indices = fold["train"]
        gallery_indices = fold["gallery"]

        train_loader = _build_loader(
            dataset,
            train_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        gallery_loader = _build_loader(
            dataset,
            gallery_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        gallery_rgb = None
        train_rgb = None
        gallery_band = None

        if any(method in methods for method in ("RGB-CLIP", "RS-TransCLIP")):
            gallery_rgb = encode_loader_with_rgb_patches(
                gallery_loader,
                clip_model,
                active_device,
                rgb_indices=(3, 2, 1),
                image_size=image_size,
                patch_pool_size=rs_transclip_patch_pool_size,
                metadata_keys=("label", "label_name", "path", "index"),
                desc=f"Fold {fold_id} RGB+patch gallery",
                show_progress=show_progress,
            )

        if "Tip-Adapter" in methods:
            train_rgb = encode_test_gallery_rgb(
                train_loader,
                clip_model,
                device=active_device,
                image_size=image_size,
                rgb_indices=(3, 2, 1),
                show_progress=show_progress,
            )
            if gallery_rgb is None:
                gallery_rgb = encode_loader_with_rgb_patches(
                    gallery_loader,
                    clip_model,
                    active_device,
                    rgb_indices=(3, 2, 1),
                    image_size=image_size,
                    patch_pool_size=rs_transclip_patch_pool_size,
                    metadata_keys=("label", "label_name", "path", "index"),
                    desc=f"Fold {fold_id} RGB+patch gallery",
                    show_progress=show_progress,
                )

        if "RGB-CLIP" in methods and gallery_rgb is not None:
            similarity = text_features @ gallery_rgb["features"].T
            evaluation = evaluate_class_query_similarity(
                similarity,
                gallery_labels=gallery_rgb["label"],
                class_names=dataset.class_names,
            )
            fold_metric_rows.append(
                {
                    "method": "RGB-CLIP",
                    "fold_id": fold_id,
                    "seed": seed,
                    "num_train": len(train_indices),
                    "num_gallery": len(gallery_indices),
                    **{metric: float(value) for metric, value in evaluation["metrics"].items()},
                }
            )
            per_query_rows.extend(
                _decorate_per_query_rows(
                    evaluation["per_query_rows"],
                    fold_id=fold_id,
                    method="RGB-CLIP",
                    class_names=dataset.class_names,
                )
            )

        if "PCA" in methods:
            pca_state = fit_global_pca_from_loader(
                train_loader,
                desc=f"Fold {fold_id} PCA fit",
                show_progress=show_progress,
            )
            gallery_pca = encode_loader_with_pca(
                gallery_loader,
                clip_model,
                active_device,
                image_size=image_size,
                pca_state=pca_state,
                metadata_keys=("label", "label_name", "path", "index"),
                desc=f"Fold {fold_id} PCA gallery",
                show_progress=show_progress,
            )
            similarity = text_features @ gallery_pca["features"].T
            evaluation = evaluate_class_query_similarity(
                similarity,
                gallery_labels=gallery_pca["label"],
                class_names=dataset.class_names,
            )
            fold_metric_rows.append(
                {
                    "method": "PCA",
                    "fold_id": fold_id,
                    "seed": seed,
                    "num_train": len(train_indices),
                    "num_gallery": len(gallery_indices),
                    **{metric: float(value) for metric, value in evaluation["metrics"].items()},
                }
            )
            per_query_rows.extend(
                _decorate_per_query_rows(
                    evaluation["per_query_rows"],
                    fold_id=fold_id,
                    method="PCA",
                    class_names=dataset.class_names,
                )
            )

        if "NDVI" in methods:
            gallery_ndvi = encode_loader_with_spectral_indices(
                gallery_loader,
                clip_model,
                active_device,
                nir_idx=get_band_index("B08"),
                red_idx=get_band_index("B04"),
                green_idx=get_band_index("B03"),
                image_size=image_size,
                metadata_keys=("label", "label_name", "path", "index"),
                desc=f"Fold {fold_id} NDVI gallery",
                show_progress=show_progress,
            )
            similarity = text_features @ gallery_ndvi["features"].T
            evaluation = evaluate_class_query_similarity(
                similarity,
                gallery_labels=gallery_ndvi["label"],
                class_names=dataset.class_names,
            )
            fold_metric_rows.append(
                {
                    "method": "NDVI",
                    "fold_id": fold_id,
                    "seed": seed,
                    "num_train": len(train_indices),
                    "num_gallery": len(gallery_indices),
                    **{metric: float(value) for metric, value in evaluation["metrics"].items()},
                }
            )
            per_query_rows.extend(
                _decorate_per_query_rows(
                    evaluation["per_query_rows"],
                    fold_id=fold_id,
                    method="NDVI",
                    class_names=dataset.class_names,
                )
            )

        if "Tip-Adapter" in methods and train_rgb is not None and gallery_rgb is not None:
            similarity = build_tip_adapter_similarity_matrix(
                text_features=text_features,
                train_features=train_rgb["features"],
                train_labels=train_rgb["labels"],
                gallery_features=gallery_rgb["features"],
                num_classes=len(dataset.class_names),
                alpha=tip_adapter_alpha,
                beta=tip_adapter_beta,
                chunk_size=tip_adapter_chunk_size,
                compute_device=active_device,
                show_progress=show_progress,
            )
            evaluation = evaluate_class_query_similarity(
                similarity,
                gallery_labels=gallery_rgb["label"],
                class_names=dataset.class_names,
            )
            fold_metric_rows.append(
                {
                    "method": "Tip-Adapter",
                    "fold_id": fold_id,
                    "seed": seed,
                    "num_train": len(train_indices),
                    "num_gallery": len(gallery_indices),
                    "tip_adapter_alpha": tip_adapter_alpha,
                    "tip_adapter_beta": tip_adapter_beta,
                    **{metric: float(value) for metric, value in evaluation["metrics"].items()},
                }
            )
            per_query_rows.extend(
                _decorate_per_query_rows(
                    evaluation["per_query_rows"],
                    fold_id=fold_id,
                    method="Tip-Adapter",
                    class_names=dataset.class_names,
                )
            )

        if "RS-TransCLIP" in methods and gallery_rgb is not None:
            patch_affinity = build_gallery_patch_affinity_knn(
                gallery_rgb["patch_descriptors"],
                topk=rs_transclip_affinity_topk,
                chunk_size=rs_transclip_affinity_chunk_size,
                compute_device=active_device,
                show_progress=show_progress,
            )
            base_similarity = text_features @ gallery_rgb["features"].T
            refined_similarity = refine_similarity_matrix(
                base_similarity,
                patch_affinity=patch_affinity,
                alpha=rs_transclip_alpha,
            )
            evaluation = evaluate_class_query_similarity(
                refined_similarity,
                gallery_labels=gallery_rgb["label"],
                class_names=dataset.class_names,
            )
            fold_metric_rows.append(
                {
                    "method": "RS-TransCLIP",
                    "fold_id": fold_id,
                    "seed": seed,
                    "num_train": len(train_indices),
                    "num_gallery": len(gallery_indices),
                    "rs_transclip_alpha": rs_transclip_alpha,
                    **{metric: float(value) for metric, value in evaluation["metrics"].items()},
                }
            )
            per_query_rows.extend(
                _decorate_per_query_rows(
                    evaluation["per_query_rows"],
                    fold_id=fold_id,
                    method="RS-TransCLIP",
                    class_names=dataset.class_names,
                )
            )

        if "Ours" in methods:
            gallery_band = encode_loader_with_band_embeddings(
                gallery_loader,
                clip_model,
                active_device,
                micro_batch_size=micro_batch_size,
                desc=f"Fold {fold_id} band embeddings",
                show_progress=show_progress,
            )
            pipeline = MultispectralRetrievalPipeline(
                sigma=ours_sigma,
                num_steps=ours_num_steps,
                lr=ours_lr,
                lambda_m=ours_lambda_m,
                k=ours_k,
                grad_clip=ours_grad_clip,
                early_stop_tol=ours_early_stop_tol,
            )
            ours_similarity, ours_stats = build_ours_similarity_matrix(
                text_features=text_features,
                gallery_band_embeddings=gallery_band["band_embeddings"],
                pipeline=pipeline,
                compute_device=active_device,
                show_progress=show_progress,
            )
            evaluation = evaluate_class_query_similarity(
                ours_similarity,
                gallery_labels=gallery_band["label"],
                class_names=dataset.class_names,
            )
            fold_metric_rows.append(
                {
                    "method": "Ours",
                    "fold_id": fold_id,
                    "seed": seed,
                    "num_train": len(train_indices),
                    "num_gallery": len(gallery_indices),
                    "ours_sigma": ours_sigma,
                    "ours_num_steps": ours_num_steps,
                    "ours_lr": ours_lr,
                    "ours_lambda_m": ours_lambda_m,
                    "ours_k": ours_k,
                    **ours_stats,
                    **{metric: float(value) for metric, value in evaluation["metrics"].items()},
                }
            )
            per_query_rows.extend(
                _decorate_per_query_rows(
                    evaluation["per_query_rows"],
                    fold_id=fold_id,
                    method="Ours",
                    class_names=dataset.class_names,
                )
            )

    external_summary_rows: List[Dict[str, Any]] = []
    external_per_query_rows: List[Dict[str, Any]] = []
    if external_method_inputs:
        external_summary_rows, external_per_query_rows = load_external_method_results(
            external_method_inputs
        )

    fold_metric_rows.extend(external_summary_rows)
    per_query_rows.extend(external_per_query_rows)

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    if not fold_metrics_df.empty:
        fold_metrics_df = fold_metrics_df.sort_values(["method", "fold_id"]).reset_index(drop=True)

    per_query_df = pd.DataFrame(per_query_rows)
    if not per_query_df.empty:
        per_query_df = per_query_df.sort_values(["method", "fold_id", "query_label"]).reset_index(drop=True)

    comparison_df = build_comparison_table(fold_metrics_df)
    paired_ttest_df = build_paired_ttest_table(fold_metrics_df, reference_method="Ours")
    per_class_df = build_per_class_table(per_query_df)
    split_manifest_df = pd.DataFrame(fold_manifest_rows)
    if not split_manifest_df.empty:
        split_manifest_df = split_manifest_df.sort_values(["fold_id", "class_idx"]).reset_index(drop=True)

    method_order = []
    if not fold_metrics_df.empty and "method" in fold_metrics_df.columns:
        seen_methods = set(fold_metrics_df["method"].astype(str).tolist())
        method_order = [method for method in methods if method in seen_methods]

    if not fold_metrics_df.empty and method_order:
        fold_metrics_df = apply_ordered_category(
            fold_metrics_df,
            column="method",
            categories=method_order,
        )
        fold_metrics_df = sort_by_available_columns(fold_metrics_df, ["method", "fold_id"])
        comparison_df = apply_ordered_category(
            comparison_df,
            column="method",
            categories=method_order,
        )
        comparison_df = sort_by_available_columns(comparison_df, ["method"])
        per_class_df = apply_ordered_category(
            per_class_df,
            column="method",
            categories=method_order,
        )
        per_class_df = sort_by_available_columns(per_class_df, ["method", "query_label"])
        per_query_df = apply_ordered_category(
            per_query_df,
            column="method",
            categories=method_order,
        )
        per_query_df = sort_by_available_columns(per_query_df, ["method", "fold_id", "query_label"])

        if not paired_ttest_df.empty and "compared_method" in paired_ttest_df.columns:
            paired_ttest_df = apply_ordered_category(
                paired_ttest_df,
                column="compared_method",
                categories=[method for method in method_order if method != "Ours"],
            )
            paired_ttest_df = sort_by_available_columns(
                paired_ttest_df,
                ["compared_method", "metric"],
            )

    fold_metrics_csv = results_dir / "eurosat_5fold_fold_metrics.csv"
    per_query_csv = results_dir / "eurosat_5fold_per_query.csv"
    comparison_csv = results_dir / "eurosat_5fold_comparison.csv"
    paired_ttest_csv = results_dir / "eurosat_5fold_paired_ttest.csv"
    per_class_csv = results_dir / "eurosat_5fold_per_class.csv"
    split_manifest_csv = results_dir / "eurosat_5fold_split_manifest.csv"
    manifest_json = results_dir / "eurosat_5fold_manifest.json"

    fold_metrics_df.to_csv(fold_metrics_csv, index=False)
    per_query_df.to_csv(per_query_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    paired_ttest_df.to_csv(paired_ttest_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)
    split_manifest_df.to_csv(split_manifest_csv, index=False)

    manifest_payload = {
        "dataset_root": str(root),
        "clip_checkpoint": str(clip_checkpoint),
        "results_dir": str(results_dir),
        "methods_requested": list(methods),
        "local_methods_run": list(local_methods),
        "external_templates": {
            method: {name: str(path) for name, path in paths.items()}
            for method, paths in external_templates.items()
        },
        "external_method_inputs": {
            method: {name: str(value) for name, value in payload.items()}
            for method, payload in (external_method_inputs or {}).items()
        },
        "num_folds": num_folds,
        "fold_ids": sorted(int(fold["fold_id"]) for fold in fold_splits),
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "micro_batch_size": micro_batch_size,
        "debug_caps": {
            "max_train_per_class": max_train_per_class,
            "max_gallery_per_class": max_gallery_per_class,
        },
        "device": str(active_device),
        "outputs": {
            "fold_metrics_csv": str(fold_metrics_csv),
            "per_query_csv": str(per_query_csv),
            "comparison_csv": str(comparison_csv),
            "paired_ttest_csv": str(paired_ttest_csv),
            "per_class_csv": str(per_class_csv),
            "split_manifest_csv": str(split_manifest_csv),
        },
    }
    write_json(manifest_payload, manifest_json)

    return {
        "fold_metrics_df": fold_metrics_df,
        "per_query_df": per_query_df,
        "comparison_df": comparison_df,
        "paired_ttest_df": paired_ttest_df,
        "per_class_df": per_class_df,
        "split_manifest_df": split_manifest_df,
        "manifest_path": manifest_json,
        "external_templates": external_templates,
    }
