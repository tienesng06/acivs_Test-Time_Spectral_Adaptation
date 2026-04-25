"""
Shared utility functions used across baselines, experiments, and scripts.

This module centralizes functions that were previously duplicated across:
    - src/baselines/pca_baseline.py
    - src/baselines/ndvi_baseline.py
    - src/baselines/rs_transclip_baseline.py
    - src/experiments/eurosat_5fold_cv.py
    - scripts/evaluate_eurosat_matched_protocol.py

Public API:
    - get_device()
    - load_openai_clip_model()
    - save_csv_rows()
    - finalize_metadata()
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


__all__ = [
    "get_device",
    "load_openai_clip_model",
    "save_csv_rows",
    "finalize_metadata",
]


def get_device() -> torch.device:
    """
    Select the best available accelerator.

    Priority: MPS (Apple Silicon) > CUDA > CPU.

    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_openai_clip_model(
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Load a frozen OpenAI CLIP model from a local checkpoint.

    Args:
        checkpoint_path: Path to the local .pt checkpoint file.
            Typically ``checkpoints/ViT-B-16.pt`` (symlink to ~/.cache/clip/).
        device: Target device for model weights.

    Returns:
        (model, clip.tokenize) — model is frozen (all params require_grad=False).

    Raises:
        ImportError: if the ``clip`` package is not installed.
        FileNotFoundError: if the checkpoint file does not exist.
    """
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "clip package is required. Install with: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")

    model, _ = clip.load(str(checkpoint_path), device=device, jit=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, clip.tokenize


def save_csv_rows(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write a sequence of dicts to a CSV file.

    Column names are inferred from dict keys in insertion order.
    The parent directory is created automatically if it does not exist.

    Args:
        rows: Non-empty sequence of dicts with consistent keys.
        output_path: Destination CSV path.

    Raises:
        ValueError: if ``rows`` is empty.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to save for {output_path}")

    fieldnames: List[str] = []
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_metadata(
    metadata: Dict[str, List[Any]],
) -> Dict[str, Any]:
    """
    Collapse per-batch metadata lists into final tensors or flat Python lists.

    Used by all encode_loader_* functions to aggregate per-batch outputs.

    Args:
        metadata: Dict mapping string keys to lists of per-batch values.
            Each value is either a Tensor (to be concatenated) or a scalar /
            list / tuple (to be flattened).

    Returns:
        Dict with the same keys, but values collapsed:
            - List[Tensor] → torch.cat(values, dim=0)
            - List[list|tuple|scalar] → flattened Python list
    """
    finalized: Dict[str, Any] = {}
    for key, values in metadata.items():
        if not values:
            finalized[key] = []
        elif torch.is_tensor(values[0]):
            finalized[key] = torch.cat(values, dim=0)
        else:
            flattened: List[Any] = []
            for value in values:
                if isinstance(value, (list, tuple)):
                    flattened.extend(list(value))
                else:
                    flattened.append(value)
            finalized[key] = flattened
    return finalized
