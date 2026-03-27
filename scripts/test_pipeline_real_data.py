"""
Test End-to-End Multispectral Retrieval Pipeline on Real BigEarthNet-S2 Data.
"""

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to load clip
try:
    import clip
except ImportError:
    print("WARNING: clip not found. Please install via: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

from src.datasets.bigearth_loader import BigEarthNetDataset
from src.models.retrieval_pipeline import MultispectralRetrievalPipeline

def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model (ViT-B/16)... using {device}")
    model, _ = clip.load("ViT-B/16", device=device)
    
    print("\nInitializing BigEarthNetDataset...")
    root_dir = PROJECT_ROOT / "data" / "BigEarthNetS2"
    
    if not root_dir.exists():
        print(f"Dataset root not found: {root_dir}")
        sys.exit(1)
        
    ds = BigEarthNetDataset(
        root=root_dir,
        split="all",
        max_samples=100,
        use_cache=False,
    )
    
    if len(ds) == 0:
        print("No samples found. Please check dataset path.")
        sys.exit(1)
        
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    csv_path = PROJECT_ROOT / "results" / "real_data_pipeline_results.csv"
    csv_path.parent.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(ds)} samples and saving to {csv_path}...")
    
    pipeline = MultispectralRetrievalPipeline(num_steps=7, lr=0.02)
    query_text = "A satellite image of complex cultivation patterns"
    
    results_list = []
    
    for i, batch in enumerate(loader):
        image_bands = batch["image"][0]
        patch_name = batch["patch_name"][0]
        
        t0 = time.time()
        result = pipeline.retrieve_from_raw(
            image_13band=image_bands,
            query_text=query_text,
            clip_model=model,
            clip_tokenize_fn=clip.tokenize,
            device=device
        )
        t1 = time.time()
        
        row = {
            "patch_name": patch_name,
            "query": query_text,
            "total_ms": round((t1 - t0) * 1000, 2),
            "opt_ms": round(result.optimization_ms, 2),
            "initial_loss": round(result.loss_history[0], 4) if result.loss_history else None,
            "final_loss": round(result.loss_history[-1], 4) if result.loss_history else None,
            "weights_sum": round(result.weights.sum().item(), 4)
        }
        results_list.append(row)
        print(f"[{i+1}/{len(ds)}] Processed {patch_name} - Loss: {row['initial_loss']}->{row['final_loss']} in {row['total_ms']}ms")
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
        
    print(f"\nDone! CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
