# ACIVS 2026 Paper Implementation Plan: 2-Month Roadmap
## Project: Test-Time Spectral-Modality Bridging for Multispectral Image Retrieval

**Timeline:** 8 Weeks (April 6 - June 5, 2026)  
**Submission Deadline:** June 1, 2026  
**Target:** Camera-ready implementation with reproducible results

---

## Week 1 (April 6-12): Environment Setup & Data Pipeline

### Day 1-2: Development Environment
**Tasks:**
- [x] Set up Python 3.9+ virtual environment
- [x] Install core dependencies:
  ```bash
  pip install torch==2.0.1 torchvision==0.15.2
  pip install transformers==4.30.0  # For CLIP
  pip install numpy scipy scikit-learn
  pip install matplotlib seaborn plotly  # Visualization
  pip install pandas tqdm h5py
  pip install rasterio geopandas  # Multispectral data handling
  ```
- [x] Install development tools: pytest, black, flake8, jupyter
- [x] Set up Git repository with .gitignore for datasets/checkpoints
- [x] Create project structure:
  ```
  project/
  ├── data/              # Dataset storage
  ├── src/
  │   ├── models/        # Core model implementations
  │   ├── utils/         # Helper functions
  │   ├── datasets/      # Dataset loaders
  │   └── experiments/   # Experiment scripts
  ├── configs/           # YAML configuration files
  ├── notebooks/         # Jupyter notebooks for analysis
  ├── results/           # Experiment outputs
  └── tests/             # Unit tests
  ```

**Deliverables:**
- Working Python environment with all dependencies
- GitHub repository initialized
- Project structure created

---

### Day 3-5: Dataset Acquisition & Preprocessing

**EuroSAT Dataset (Primary):**
- [x] Download EuroSAT-MS (Sentinel-2, 13 bands, 27,000 images)
  - Source: https://github.com/phelber/EuroSAT
  - Size: ~2.8 GB
- [x] Implement EuroSAT dataloader:
  ```python
  class EuroSATDataset(torch.utils.data.Dataset):
      def __init__(self, root, split='train', bands='all'):
          # Load 13-band multispectral images
          # Bands: B01-B12, B8A (10m-60m resolution)
          pass
      
      def __getitem__(self, idx):
          # Return: (13, H, W) tensor, class_label, text_description
          pass
  ```
- [x] Create train/val/test splits (60/20/20)
- [x] Generate text descriptions for 10 classes:
  - "Annual Crop", "Forest", "Herbaceous Vegetation", etc.
  - Use templates: "A satellite image of {class}"
- [x] Compute dataset statistics (mean, std per band)

**BigEarthNet Dataset (Secondary):**
- [x] Download BigEarthNet-S2 (Sentinel-2, 12 bands, 590k patches)
  - Source: https://bigearth.net/
  - Size: ~65 GB (download subset if storage limited)
- [x] Implement BigEarthNet dataloader with multi-label support
- [x] Filter to top 10 most frequent classes for retrieval evaluation
- [x] Create retrieval query set (1,000 queries) and gallery (10,000 images)

**Data Validation:**
- [x] Verify band order matches Sentinel-2 specification
- [x] Check for missing/corrupted files
- [x] Visualize sample images (RGB composite + individual bands)
- [x] Compute band correlation matrix

**Deliverables:**
- EuroSAT dataloader with 27k images ready
- BigEarthNet dataloader (subset) ready
- Data statistics report (mean, std, class distribution)
- Visualization notebook showing sample multispectral images

---

### Day 6-7: Baseline CLIP Setup

**CLIP Model Integration:**
- [x] Load pretrained CLIP model (ViT-B/16 recommended):
  ```python
  import clip
  model, preprocess = clip.load("ViT-B/16", device="cuda")
  model.eval()  # Freeze weights
  ```
- [x] Implement RGB baseline:
  - Extract B04 (Red), B03 (Green), B02 (Blue) from 13-band data
  - Normalize to [0, 1] and resize to 224×224
  - Pass through CLIP image encoder
- [x] Implement text encoding:
  ```python
  text_inputs = clip.tokenize(["A satellite image of forest", ...])
  text_features = model.encode_text(text_inputs)
  ```
- [x] Build retrieval evaluation pipeline:
  - Compute image-text similarity: `sim = image_features @ text_features.T`
  - Metrics: Recall@1, Recall@5, Recall@10, mAP

**Baseline Experiments:**
- [x] Run RGB-CLIP on EuroSAT test set
- [x] Expected results: ~68% R@1, ~84% R@10 (from paper)
- [x] Log results to CSV with timestamps

**Deliverables:**
- Working CLIP inference pipeline
- Baseline RGB results on EuroSAT
- Evaluation script with standard metrics

---

## Week 2 (April 13-19): Core Component 1 - Query-Conditioned Spectral Weighting

### Day 8-10: Per-Band CLIP Encoding

**Implementation:**
- [ ] Create per-band encoding function:
  ```python
  def encode_multispectral_bands(image_13band, clip_model):
      """
      Args:
          image_13band: (13, H, W) tensor
      Returns:
          band_embeddings: (13, 512) tensor
      """
      band_embeddings = []
      for b in range(13):
          # Replicate single band to 3 channels for CLIP
          band_3ch = image_13band[b:b+1].repeat(3, 1, 1)
          # Resize to 224x224 and normalize
          band_3ch = preprocess(band_3ch)
          # Encode with frozen CLIP
          with torch.no_grad():
              emb = clip_model.encode_image(band_3ch.unsqueeze(0))
          band_embeddings.append(emb.squeeze(0))
      return torch.stack(band_embeddings)  # (13, 512)
  ```
- [x] Optimize batch processing for efficiency
- [x] Cache per-band embeddings to disk (HDF5 format)

**Validation:**
- [x] Verify embedding dimensions: (13, 512) for ViT-B/16
- [x] Check embedding magnitudes (should be L2-normalized)
- [ ] Visualize t-SNE of per-band embeddings across classes

**Deliverables:**
- Per-band encoding function
- Cached embeddings for EuroSAT (27k × 13 × 512)
- Validation notebook

---

### Day 11-13: Affinity Graph Construction

**Implementation (Equation 1 from paper):**
- [x] Compute query-conditioned affinity matrix:
  ```python
  def compute_affinity_graph(band_embeddings, query_embedding, sigma=0.5):
      """
      Args:
          band_embeddings: (B, D) - B bands, D=512
          query_embedding: (D,) - text query embedding
          sigma: temperature for query alignment
      Returns:
          A: (B, B) affinity matrix
      """
      B, D = band_embeddings.shape
      
      # Query alignment scores
      query_scores = (band_embeddings @ query_embedding) / sigma
      query_weights = torch.softmax(query_scores, dim=0)  # (B,)
      
      # Pairwise band similarities
      S = band_embeddings @ band_embeddings.T  # (B, B)
      
      # Query-conditioned affinity (Eq. 1)
      A = S * (query_weights.unsqueeze(1) * query_weights.unsqueeze(0))
      
      return A
  ```
- [x] Implement symmetric normalization:
  ```python
  D = torch.diag(A.sum(dim=1) ** -0.5)
  A_norm = D @ A @ D
  ```

**Hyperparameter Tuning:**
- [ ] Grid search for sigma: [0.1, 0.3, 0.5, 0.7, 1.0]
- [ ] Evaluate impact on R@10 (expect <1.5% variance per paper)

**Deliverables:**
- Affinity graph construction function
- Hyperparameter sensitivity analysis
- Visualization of affinity matrices for sample queries

---

### Day 14: Fiedler Vector Computation

**Implementation (Equation 2 from paper):**
- [x] Compute graph Laplacian:
  ```python
  def compute_fiedler_vector(A):
      """
      Args:
          A: (B, B) affinity matrix
      Returns:
          fiedler_vec: (B,) - second smallest eigenvector of Laplacian
      """
      # Graph Laplacian: L = D - A
      D = torch.diag(A.sum(dim=1))
      L = D - A
      
      # Eigendecomposition (use scipy for stability)
      L_np = L.cpu().numpy()
      eigenvalues, eigenvectors = np.linalg.eigh(L_np)
      
      # Fiedler vector is 2nd smallest eigenvalue's eigenvector
      fiedler_vec = torch.from_numpy(eigenvectors[:, 1])
      
      return fiedler_vec
  ```
- [x] Implement magnitude-based weighting:
  ```python
  w_fiedler = torch.abs(fiedler_vec)  # (B,)
  w_fiedler = w_fiedler / w_fiedler.sum()  # Normalize
  ```

**Theoretical Validation:**
- [x] Verify Proposition 3.1 from paper:
  - Fiedler vectors identify coherent spectral partitions
  - Test on synthetic 2-cluster band embeddings
- [x] Compare with alternative graph metrics (PageRank, betweenness)

**Deliverables:**
- Fiedler vector computation function
- Theoretical validation notebook
- Comparison with alternative graph metrics

---

## Week 3 (April 20-26): Core Component 2 - Test-Time Manifold Consistency

### Day 15-17: Manifold Consistency Loss

**Implementation (Equation 6 from paper):**
- [x] Implement k-NN graph construction:
  ```python
  def build_knn_graph(band_embeddings, k=5):
      """
      Args:
          band_embeddings: (B, D)
          k: number of nearest neighbors
      Returns:
          knn_indices: (B, k) - neighbor indices for each band
      """
      # Compute pairwise distances
      dist_matrix = torch.cdist(band_embeddings, band_embeddings)
      
      # Find k nearest neighbors (excluding self)
      knn_indices = torch.topk(dist_matrix, k+1, largest=False).indices[:, 1:]
      
      return knn_indices
  ```
- [x] Implement manifold consistency loss:
  ```python
  def manifold_consistency_loss(fused_embedding, band_embeddings, knn_indices, lambda_m=0.1):
      """
      Args:
          fused_embedding: (D,) - weighted fusion of band embeddings
          band_embeddings: (B, D)
          knn_indices: (B, k)
          lambda_m: loss weight
      Returns:
          loss: scalar
      """
      B, k = knn_indices.shape
      loss = 0.0
      
      for i in range(B):
          # Neighbors of band i in original space
          neighbors_orig = band_embeddings[knn_indices[i]]  # (k, D)
          
          # Distance from fused embedding to neighbors
          dist_fused = torch.norm(fused_embedding - neighbors_orig, dim=1)
          
          # Penalize large distances (preserve local structure)
          loss += dist_fused.mean()
      
      return lambda_m * loss / B
  ```

**Validation:**
- [x] Verify loss decreases during optimization
- [x] Check k-NN preservation: % of neighbors preserved after fusion
- [x] Visualize embedding space before/after manifold consistency

**Deliverables:**
- Manifold consistency loss implementation
- k-NN graph construction
- Validation metrics

---

### Day 18-20: Test-Time Optimization

**Implementation:**
- [x] Implement 5-step gradient descent:
  ```python
  def optimize_fusion(band_embeddings, query_embedding, w_fiedler, num_steps=5, lr=0.01):
      """
      Args:
          band_embeddings: (B, D)
          query_embedding: (D,)
          w_fiedler: (B,) - initial Fiedler weights
          num_steps: optimization steps
          lr: learning rate
      Returns:
          optimized_weights: (B,)
      """
      # Initialize learnable weights
      w = w_fiedler.clone().requires_grad_(True)
      optimizer = torch.optim.Adam([w], lr=lr)
      
      # Build k-NN graph once
      knn_indices = build_knn_graph(band_embeddings, k=5)
      
      for step in range(num_steps):
          optimizer.zero_grad()
          
          # Weighted fusion
          w_norm = torch.softmax(w, dim=0)
          fused_emb = (w_norm.unsqueeze(1) * band_embeddings).sum(dim=0)
          
          # Manifold consistency loss
          loss = manifold_consistency_loss(fused_emb, band_embeddings, knn_indices)
          
          loss.backward()
          optimizer.step()
      
      return torch.softmax(w, dim=0).detach()
  ```
- [x] Implement early stopping based on loss convergence
- [x] Add gradient clipping for stability

**Hyperparameter Tuning:**
- [x] Grid search: num_steps [3, 5, 7], lr [0.005, 0.01, 0.02]
- [x] Evaluate computational overhead: target <200ms per query

**Deliverables:**
- Test-time optimization function
- Convergence analysis (loss curves)
- Computational profiling report

---

### Day 21: Integration & End-to-End Pipeline

**Full Pipeline:**
- [x] Integrate all components:
  ```python
  def retrieve_multispectral(image_13band, query_text, clip_model, sigma=0.5, lambda_m=0.1):
      # 1. Encode per-band embeddings
      band_embeddings = encode_multispectral_bands(image_13band, clip_model)
      
      # 2. Encode query text
      query_embedding = clip_model.encode_text(clip.tokenize([query_text]))
      
      # 3. Compute affinity graph
      A = compute_affinity_graph(band_embeddings, query_embedding, sigma)
      
      # 4. Compute Fiedler weights
      fiedler_vec = compute_fiedler_vector(A)
      w_fiedler = torch.abs(fiedler_vec)
      w_fiedler = w_fiedler / w_fiedler.sum()
      
      # 5. Test-time optimization with manifold consistency
      w_optimized = optimize_fusion(band_embeddings, query_embedding, w_fiedler)
      
      # 6. Final weighted fusion
      fused_embedding = (w_optimized.unsqueeze(1) * band_embeddings).sum(dim=0)
      
      return fused_embedding, w_optimized
  ```
- [x] Add error handling and input validation
- [x] Optimize for batch processing (process multiple queries in parallel)

**End-to-End Testing:**
- [x] Run on 100 sample images from EuroSAT
- [x] Verify outputs: embedding shape (512,), weights sum to 1
- [x] Profile total inference time: target <200ms per image

**Deliverables:**
- Complete end-to-end retrieval function
- Integration tests
- Performance profiling report

---

## Week 4 (April 27 - May 3): Core Component 3 - Interpretable Band Attribution

### Day 22-24: Band Attribution Analysis

**Implementation:**
- [x] Compute per-band contribution scores:
  ```python
  def compute_band_attribution(band_embeddings, query_embedding, w_optimized):
      """
      Args:
          band_embeddings: (B, D)
          query_embedding: (D,)
          w_optimized: (B,) - optimized weights
      Returns:
          attribution_scores: (B,) - contribution of each band
      """
      # Query alignment per band
      alignment = (band_embeddings @ query_embedding).squeeze()
      
      # Attribution = weight × alignment
      attribution = w_optimized * alignment
      
      # Normalize to [0, 1]
      attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
      
      return attribution
  ```
- [x] Implement visualization:
  ```python
  def visualize_band_attribution(image_13band, attribution_scores, band_names):
      """
      Create bar plot showing contribution of each spectral band
      """
      import matplotlib.pyplot as plt
      
      fig, ax = plt.subplots(1, 1, figsize=(12, 4))
      ax.bar(band_names, attribution_scores.cpu().numpy())
      ax.set_xlabel('Spectral Band')
      ax.set_ylabel('Attribution Score')
      ax.set_title('Per-Band Contribution to Retrieval')
      plt.xticks(rotation=45)
      plt.tight_layout()
      return fig
  ```

**Sentinel-2 Band Names:**
- B01 (Coastal Aerosol), B02 (Blue), B03 (Green), B04 (Red)
- B05-B07 (Red Edge), B08 (NIR), B8A (Narrow NIR)
- B09 (Water Vapor), B10 (SWIR-Cirrus), B11-B12 (SWIR)

**Analysis Tasks:**
- [x] Compute attribution for all 10 EuroSAT classes
- [x] Identify class-specific band preferences:
  - Forest: Expect high NIR (B08), Red Edge (B05-B07)
  - Water: Expect high Blue (B02), Green (B03)
  - Urban: Expect high SWIR (B11-B12)
- [x] Statistical analysis: mean ± std attribution per class

**Deliverables:**
- Band attribution function
- Visualization script
- Attribution analysis report for 10 classes

---

### Day 25-26: Failure Case Analysis

**Implementation:**
- [ ] Identify failure cases (low retrieval accuracy):
  ```python
  def identify_failure_cases(results_df, threshold=0.5):
      """
      Args:
          results_df: DataFrame with columns [image_id, true_class, predicted_class, similarity_score]
          threshold: minimum similarity for success
      Returns:
          failure_df: DataFrame of failure cases
      """
      failures = results_df[results_df['similarity_score'] < threshold]
      return failures
  ```
- [x] Categorize failures:
  - Cloud contamination
  - Mixed urban/vegetation scenes
  - Rare/underrepresented classes
  - Seasonal variations

**Quantitative Analysis:**
- [x] Measure performance drop for each failure category:
  - Cloud cover: Expected ~17% R@1 drop (from paper)
  - Mixed scenes: Expected ~13% R@1 drop
  - Rare classes: Expected ~10% R@1 drop
- [x] Compute confusion matrix for failure cases

**Visualization:**
- [x] Create failure case gallery with:
  - Original multispectral image (RGB composite)
  - Attribution scores
  - True vs predicted class
  - Failure reason annotation

**Deliverables:**
- Failure case detection script
- Categorized failure analysis
- Visualization gallery

---

### Day 27-28: Interpretability Experiments

**Ablation Studies:**
- [x] Ablate individual bands and measure impact:
  ```python
  for band_idx in range(13):
      # Mask out band_idx
      masked_embeddings = band_embeddings.clone()
      masked_embeddings[band_idx] = 0
      
      # Recompute retrieval
      fused_emb = retrieve_multispectral(masked_embeddings, query_text, ...)
      
      # Measure R@10 drop
      print(f"Removing Band {band_idx}: ΔR@10 = {baseline_r10 - masked_r10:.2f}%")
  ```
- [x] Expected findings:
  - NIR (B08) most critical for vegetation classes
  - SWIR (B11-B12) most critical for urban/bare soil
  - Visible bands (B02-B04) important for water

**Human Interpretability Study (Optional):**
- [x] Generate 50 sample attributions
- [x] Ask domain expert to validate if attributions align with spectral physics
- [x] Compute agreement rate

**Deliverables:**
- Band ablation results
- Interpretability validation report

---

## Week 5 (May 4-10): Baseline Implementations & Comparisons

### Day 29-31: PCA-to-3-Channels Baseline

**Implementation:**
- [ ] PCA dimensionality reduction:
  ```python
  from sklearn.decomposition import PCA
  
  def pca_to_3_channels(image_13band):
      """
      Args:
          image_13band: (13, H, W)
      Returns:
          image_3ch: (3, H, W) - top 3 PCA components
      """
      # Reshape to (H*W, 13)
      H, W = image_13band.shape[1:]
      pixels = image_13band.reshape(13, -1).T  # (H*W, 13)
      
      # Fit PCA
      pca = PCA(n_components=3)
      pixels_pca = pca.fit_transform(pixels)  # (H*W, 3)
      
      # Reshape back to (3, H, W)
      image_3ch = pixels_pca.T.reshape(3, H, W)
      
      # Normalize to [0, 1]
      image_3ch = (image_3ch - image_3ch.min()) / (image_3ch.max() - image_3ch.min())
      
      return torch.from_numpy(image_3ch).float()
  ```
- [x] Run PCA+CLIP on EuroSAT and BigEarthNet
- [x] Expected results: ~72% R@1 on EuroSAT (from paper insights)

**Deliverables:**
- PCA baseline implementation
- Results on both datasets

---

### Day 32-33: NDVI-Based Band Selection Baseline

**Implementation:**
- [x] Compute NDVI index:
  ```python
  def compute_ndvi(image_13band):
      """
      NDVI = (NIR - Red) / (NIR + Red)
      """
      nir = image_13band[7]   # B08 (NIR)
      red = image_13band[3]   # B04 (Red)
      
      ndvi = (nir - red) / (nir + red + 1e-8)
      return ndvi
  ```
- [x] Select top-3 indices: NDVI, NDWI (water), SAVI (soil)
- [x] Create 3-channel composite from indices
- [x] Run through CLIP

**Expected Performance:**
- R² = 0.70-0.78 correlation with ground truth (from literature)
- Lower than PCA due to fixed index selection

**Deliverables:**
- NDVI baseline implementation
- Performance comparison with PCA

---

### Day 34-35: RS-TransCLIP Baseline

**Implementation:**
- [x] Implement transductive patch-affinity refinement:
  ```python
  def rs_transclip_refinement(image_features, text_features, alpha=0.5):
      """
      Refine predictions using patch-level affinities
      Args:
          image_features: (N, D) - N images
          text_features: (C, D) - C classes
          alpha: refinement weight
      Returns:
          refined_logits: (N, C)
      """
      # Initial predictions
      logits = image_features @ text_features.T  # (N, C)
      
      # Compute image-image affinity
      affinity = image_features @ image_features.T  # (N, N)
      affinity = torch.softmax(affinity / 0.1, dim=1)
      
      # Transductive refinement
      refined_logits = (1 - alpha) * logits + alpha * (affinity @ logits)
      
      return refined_logits
  ```
- [x] Tune alpha hyperparameter: [0.3, 0.5, 0.7]
- [x] Run on EuroSAT and BigEarthNet

**Expected Performance:**
- ~75% R@1 on EuroSAT (4.4% below our method per paper)
- ~63% R@1 on BigEarthNet (4.0% below our method)

**Deliverables:**
- RS-TransCLIP implementation
- Hyperparameter tuning results

---

## Week 6 (May 11-17): Full Experimental Evaluation

### Day 36-38: Main Results on EuroSAT

**Experimental Protocol:**
- [ ] Use 5-fold cross-validation for robustness
- [ ] For each fold:
  - Train: 60% (16,200 images)
  - Val: 20% (5,400 images)
  - Test: 20% (5,400 images)
- [ ] Run all methods:
  1. RGB-CLIP (baseline)
  2. PCA-to-3 + CLIP
  3. NDVI-based + CLIP
  4. Tip-Adapter
  5. RS-TransCLIP
  6. **Ours (full method)**
  7. DOFA-CLIP* (supervised upper bound, if available)

**Metrics:**
- [ ] Recall@1, Recall@5, Recall@10
- [ ] Mean Average Precision (mAP)
- [ ] Mean Reciprocal Rank (MRR)
- [ ] Per-class breakdown

**Statistical Testing:**
- [ ] Compute mean ± std across 5 folds
- [ ] Paired t-test: Ours vs each baseline
- [ ] Report p-values (expect p<0.001 vs CLIP, p<0.01 vs Tip-Adapter)

**Expected Results (from paper):**
| Method | R@1 | R@10 |
|--------|-----|------|
| RGB-CLIP | 68.1±1.2% | 84.1±1.0% |
| PCA+CLIP | 72.3±1.5% | 87.8±1.1% |
| Tip-Adapter | 73.3±1.6% | 89.3±1.3% |
| RS-TransCLIP | 75.1±1.3% | 91.1±1.1% |
| **Ours** | **79.5±1.4%** | **95.5±1.2%** |

**Deliverables:**
- Results table with mean ± std
- Statistical significance tests
- Per-class performance breakdown

---

### Day 39-41: Main Results on BigEarthNet

**Experimental Protocol:**
- [ ] Use subset of 100k images (full 590k too large for 2-month timeline)
- [ ] Filter to top 10 most frequent classes
- [ ] Create retrieval benchmark:
  - Query set: 1,000 images
  - Gallery set: 10,000 images
- [ ] Run all methods (same as EuroSAT)

**Multi-Label Retrieval:**
- [ ] Adapt metrics for multi-label:
  - Precision@K: % of top-K results with overlapping labels
  - Recall@K: % of relevant images in top-K
  - F1@K: harmonic mean of Precision@K and Recall@K

**Expected Results (from paper):**
| Method | R@1 | R@10 |
|--------|-----|------|
| RGB-CLIP | 53.6±1.9% | 75.3±1.6% |
| Tip-Adapter | 59.8±2.1% | 81.5±1.7% |
| RS-TransCLIP | 62.6±1.8% | 84.3±1.5% |
| **Ours** | **66.6±1.8%** | **88.3±1.4%** |

**Deliverables:**
- BigEarthNet results table
- Multi-label metrics
- Comparison with EuroSAT (discuss domain differences)

---

### Day 42: Computational Complexity Analysis

**Profiling:**
- [ ] Measure time for each component:
  ```python
  import time
  
  # Per-band encoding
  start = time.time()
  band_embeddings = encode_multispectral_bands(image, clip_model)
  time_encoding = time.time() - start
  
  # Affinity graph
  start = time.time()
  A = compute_affinity_graph(band_embeddings, query_embedding)
  time_affinity = time.time() - start
  
  # Fiedler vector
  start = time.time()
  fiedler_vec = compute_fiedler_vector(A)
  time_fiedler = time.time() - start
  
  # Test-time optimization
  start = time.time()
  w_optimized = optimize_fusion(band_embeddings, query_embedding, w_fiedler)
  time_optimization = time.time() - start
  
  print(f"Total: {time_encoding + time_affinity + time_fiedler + time_optimization:.3f}s")
  ```

**Big-O Analysis:**
- [ ] Document complexity:
  - Per-band encoding: O(B × C) where B=13 bands, C=CLIP inference
  - Affinity graph: O(B²)
  - Fiedler vector: O(B³) eigendecomposition
  - Test-time optimization: O(5 × k × B) where k=5 neighbors
  - **Total: O(B³) dominated by eigendecomposition**

**Optimization:**
- [ ] Profile bottlenecks (expect eigendecomposition ~60% of time)
- [ ] Implement fast approximate Fiedler vector (optional):
  - Power iteration method: O(B² log B)
  - Expected speedup: 2-3×

**Target Performance:**
- Total inference: <200ms per image (from paper)
- Breakdown: Encoding 120ms, Graph 20ms, Fiedler 40ms, Optimization 20ms

**Deliverables:**
- Computational profiling report
- Big-O complexity table
- Optimization recommendations

---

## Week 7 (May 18-24): Ablation Studies & Hyperparameter Sensitivity

### Day 43-45: Component Ablation Studies

**Ablation Experiments:**
- [ ] Ablate each component sequentially:
  1. **Baseline:** RGB-CLIP (68.1% R@1)
  2. **+Per-band encoding:** All bands → CLIP, simple average (73.5% R@1)
  3. **+Query-conditioned affinity:** Add Eq. 1 weighting (76.2% R@1)
  4. **+Fiedler vectors:** Add Eq. 2 graph coherence (78.1% R@1)
  5. **+Manifold consistency:** Add Eq. 6 optimization (79.5% R@1)

- [ ] Ablate design choices:
  - **Eq. 1 (Affinity):** Multiply vs Add vs Concatenate
    - Expected: Multiply best (ensures query-aligned bands dominate)
  - **Eq. 2 (Fiedler):** Magnitude vs Sign vs Random weights
    - Expected: Magnitude best (sign indicates partition, magnitude confidence)
  - **Eq. 6 (Manifold):** All bands vs Top-K bands vs No regularization
    - Expected: All bands best (avoids overfitting to few bands)

**Statistical Testing:**
- [ ] Paired t-test for each ablation
- [ ] Report p-values and effect sizes

**Expected Results (from paper):**
| Configuration | R@1 | ΔR@1 |
|---------------|-----|------|
| Full method | 79.5% | - |
| w/o Manifold consistency | 78.1% | -1.4% |
| w/o Fiedler vectors | 76.2% | -3.3% |
| w/o Query-conditioned affinity | 73.5% | -6.0% |

**Deliverables:**
- Ablation study table
- Design choice justification report

---

### Day 46-47: Hyperparameter Sensitivity Analysis

**Hyperparameters to Tune:**
1. **σ (query alignment temperature):** [0.1, 0.3, 0.5, 0.7, 1.0]
2. **τ (affinity normalization):** [0.05, 0.1, 0.2, 0.5]
3. **λ_m (manifold loss weight):** [0.01, 0.05, 0.1, 0.2]
4. **k (k-NN neighbors):** [3, 5, 7, 10]
5. **num_steps (optimization steps):** [1, 3, 5, 7, 10]
6. **lr (learning rate):** [0.005, 0.01, 0.02, 0.05]

**Experimental Protocol:**
- [ ] Fix all hyperparameters except one
- [ ] Vary target hyperparameter across range
- [ ] Measure R@10 on EuroSAT validation set
- [ ] Plot sensitivity curves

**Expected Findings (from paper):**
- Performance variations <1.5% across reasonable ranges
- Optimal: σ=0.5, λ_m=0.1, k=5, num_steps=5, lr=0.01

**Deliverables:**
- Hyperparameter sensitivity plots (6 figures)
- Optimal hyperparameter recommendations

---

### Day 48-49: Cross-Dataset Generalization

**Experiment:**
- [ ] Train hyperparameters on EuroSAT
- [ ] Test zero-shot on BigEarthNet (no hyperparameter tuning)
- [ ] Measure performance drop:
  - Expected: <3% R@10 drop (shows generalization)

**Domain Shift Analysis:**
- [ ] Compare band attribution patterns:
  - EuroSAT: Single-label, 10 classes, European scenes
  - BigEarthNet: Multi-label, 19 classes, global scenes
- [ ] Identify classes with largest domain shift:
  - Expected: Urban classes (different building styles)
  - Expected: Agricultural classes (different crop types)

**Deliverables:**
- Cross-dataset generalization results
- Domain shift analysis report

---

## Week 8 (May 25 - June 1): Paper Finalization & Submission

### Day 50-52: Results Visualization & Figure Generation

**Key Figures for Paper:**
- [ ] **Figure 1:** Method overview diagram (already created)
- [ ] **Figure 2:** Qualitative results
  - 6 sample retrievals (2 per dataset)
  - Show: Query text, Top-5 retrieved images, Attribution scores
- [ ] **Figure 3:** Band attribution analysis
  - Heatmap: 10 classes × 13 bands
  - Show class-specific spectral preferences
- [ ] **Figure 4:** Ablation study
  - Bar chart: R@1 and R@10 for each configuration
  - Error bars for statistical significance
- [ ] **Figure 5:** Hyperparameter sensitivity
  - 6 subplots for each hyperparameter
  - Show robustness across ranges
- [ ] **Figure 6:** Failure cases
  - 4 examples: Cloud, Mixed urban, Rare class, Seasonal
  - Show RGB composite + Attribution + Prediction

**Table Generation:**
- [ ] **Table 1:** Main results on EuroSAT
- [ ] **Table 2:** Main results on BigEarthNet
- [ ] **Table 3:** Ablation study
- [ ] **Table 4:** Computational complexity comparison
- [ ] **Table 5:** Hyperparameter sensitivity (variance)

**Deliverables:**
- All figures in high-resolution (300 DPI)
- All tables in LaTeX format
- Figure captions and table captions

---

### Day 53-54: Code Release Preparation

**GitHub Repository Structure:**
```
acivs2026-multispectral-retrieval/
├── README.md                    # Installation, usage, citation
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
├── configs/
│   ├── eurosat.yaml            # EuroSAT experiment config
│   └── bigearthnet.yaml        # BigEarthNet experiment config
├── src/
│   ├── models/
│   │   ├── clip_encoder.py    # CLIP wrapper
│   │   ├── affinity_graph.py  # Eq. 1 implementation
│   │   ├── fiedler.py         # Eq. 2 implementation
│   │   └── manifold.py        # Eq. 6 implementation
│   ├── datasets/
│   │   ├── eurosat.py         # EuroSAT dataloader
│   │   └── bigearthnet.py     # BigEarthNet dataloader
│   ├── utils/
│   │   ├── metrics.py         # Retrieval metrics
│   │   └── visualization.py   # Attribution plots
│   └── experiments/
│       ├── train.py           # Main training script
│       └── evaluate.py        # Evaluation script
├── scripts/
│   ├── download_data.sh       # Dataset download
│   └── run_experiments.sh     # Reproduce paper results
├── notebooks/
│   ├── demo.ipynb             # Interactive demo
│   └── analysis.ipynb         # Result analysis
└── tests/
    ├── test_models.py
    └── test_datasets.py
```

**Documentation:**
- [ ] Write comprehensive README:
  - Installation instructions
  - Dataset preparation
  - Usage examples
  - Reproducing paper results
  - Citation
- [ ] Add docstrings to all functions
- [ ] Create demo Jupyter notebook

**Anonymous Release:**
- [ ] Remove author names and affiliations
- [ ] Host on anonymous GitHub (e.g., anonymous.4open.science)
- [ ] Include pre-computed results for reproducibility

**Deliverables:**
- Complete GitHub repository
- Anonymous release link for paper submission

---

### Day 55-56: Paper Final Review & Submission

**LaTeX Compilation:**
- [ ] Compile final PDF (target: 10-12 pages)
- [ ] Verify LNCS format compliance:
  - Font: Times Roman 10pt
  - Margins: 1 inch all sides
  - Line spacing: Single
  - References: Numbered, APA style
- [ ] Check all figures render correctly
- [ ] Verify all citations resolve

**Final Checklist:**
- [ ] Abstract: 150-200 words, self-contained
- [ ] Introduction: Clear problem statement, contributions, novelty
- [ ] Related Work: Comprehensive coverage, clear positioning
- [ ] Methodology: All equations explained, design choices justified
- [ ] Experiments: Statistical significance, error bars, ablations
- [ ] Discussion: Limitations, failure cases, future work
- [ ] Conclusion: Summarize contributions, impact
- [ ] References: 30-40 citations, recent (2020-2026)
- [ ] Supplementary Material: Code, additional results

**Submission:**
- [ ] Upload PDF to ACIVS 2026 submission portal
- [ ] Upload supplementary material (code, additional figures)
- [ ] Submit by **June 1, 2026 deadline**
- [ ] Confirm submission receipt

**Deliverables:**
- Final camera-ready PDF (10-12 pages)
- Supplementary material ZIP
- Submission confirmation

---

## Post-Submission Tasks (June 2-5)

### Rebuttal Preparation (If Needed)

**If Reviews Received:**
- [ ] Read all reviewer comments carefully
- [ ] Categorize concerns:
  - Major: Requires new experiments or significant revisions
  - Minor: Clarifications, typos, presentation
- [ ] Draft 2-page rebuttal addressing each concern:
  - Acknowledge valid points
  - Provide additional experiments if feasible
  - Clarify misunderstandings
  - Promise revisions for camera-ready

**Backup Plan:**
- [ ] If rejected, identify alternative venues:
  - CVPR 2027 (November deadline)
  - ICCV 2027 (March deadline)
  - IEEE Transactions on Geoscience and Remote Sensing (journal)

---

## Resource Requirements Summary

### Computational Resources
- **GPU:** 1× NVIDIA RTX 3080 (10GB VRAM) or better
  - CLIP inference: ~4GB VRAM
  - Batch processing: ~8GB VRAM
- **CPU:** 8+ cores for data preprocessing
- **RAM:** 32GB minimum (64GB recommended)
- **Storage:** 100GB
  - Datasets: 70GB (EuroSAT 3GB + BigEarthNet subset 65GB)
  - Cached embeddings: 20GB
  - Results: 10GB

### Time Budget (Hours per Week)
- **Week 1:** 20 hours (setup, data)
- **Week 2-4:** 25 hours/week (core implementation)
- **Week 5-6:** 30 hours/week (baselines, experiments)
- **Week 7:** 25 hours (ablations, analysis)
- **Week 8:** 20 hours (paper writing, submission)
- **Total:** ~180 hours over 8 weeks

### Software Dependencies
- Python 3.9+
- PyTorch 2.0+
- CLIP (OpenAI)
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn, Plotly
- Rasterio (multispectral data)
- LaTeX (paper writing)

---

## Risk Mitigation Strategies

### Risk 1: Dataset Download Issues
- **Mitigation:** Start downloads on Day 1, verify integrity immediately
- **Backup:** Use smaller subsets if full datasets unavailable

### Risk 2: CLIP Inference Too Slow
- **Mitigation:** Use batch processing, cache embeddings to disk
- **Backup:** Use smaller CLIP model (ViT-B/32) if ViT-B/16 too slow

### Risk 3: Eigendecomposition Bottleneck
- **Mitigation:** Implement fast approximate Fiedler vector (power iteration)
- **Backup:** Use pre-computed graph Laplacians

### Risk 4: Baseline Implementations Unclear
- **Mitigation:** Contact authors for clarification, use published code if available
- **Backup:** Implement simplified versions based on paper descriptions

### Risk 5: Results Don't Match Paper
- **Mitigation:** Debug systematically, verify each component independently
- **Backup:** Adjust expectations, focus on relative improvements

### Risk 6: Submission Deadline Missed
- **Mitigation:** Set internal deadline 3 days early (May 29)
- **Backup:** Submit to backup venue (CVPR 2027)

---

## Success Metrics

### Technical Milestones
- [ ] Week 2: Per-band encoding working, baseline RGB results reproduced
- [ ] Week 4: Full pipeline integrated, end-to-end inference <200ms
- [ ] Week 6: Main results on EuroSAT match paper (79.5±1.4% R@1)
- [ ] Week 7: All ablations complete, statistical significance confirmed
- [ ] Week 8: Paper submitted by June 1 deadline

### Quality Metrics
- **Code Quality:** 80%+ test coverage, documented functions
- **Reproducibility:** Results reproducible within ±1% variance
- **Paper Quality:** Clear writing, comprehensive experiments, strong positioning
- **Novelty:** 3 core contributions clearly differentiated from prior work

---

## Daily Schedule Template

**Morning (9 AM - 12 PM):**
- Implementation work (coding, debugging)
- Focus on current week's primary tasks

**Afternoon (1 PM - 4 PM):**
- Experiments and evaluation
- Data analysis and visualization

**Evening (4 PM - 6 PM):**
- Documentation and writing
- Code review and testing

**Weekly Review (Friday PM):**
- Review week's progress against todo.md
- Plan next week's tasks
- Update GitHub repository

---

## Contact & Support

**If You Get Stuck:**
1. Check paper for implementation details
2. Review related work codebases (Tip-Adapter, ZLaP on GitHub)
3. Post questions on research forums (Reddit r/MachineLearning, Twitter)
4. Contact paper authors if clarification needed

**Recommended Resources:**
- CLIP GitHub: https://github.com/openai/CLIP
- EuroSAT: https://github.com/phelber/EuroSAT
- BigEarthNet: https://bigearth.net/
- Spectral indices: https://www.indexdatabase.de/

---

## Conclusion

This 2-month implementation plan provides a **detailed, week-by-week roadmap** to implement your ACIVS 2026 paper from scratch. The plan is **realistic** given your background (Python, PyTorch, CNN experience) and **achievable** with consistent effort (~20-30 hours/week).

**Key Success Factors:**
1. **Start early:** Begin dataset downloads and environment setup immediately
2. **Validate continuously:** Test each component before moving to next
3. **Document thoroughly:** Write docstrings and README as you code
4. **Stay organized:** Use Git, maintain clean code structure
5. **Ask for help:** Don't hesitate to reach out if stuck

**Good luck with your implementation! You've got this! 🚀**
