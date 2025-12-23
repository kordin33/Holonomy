# 🔬 Loop Holonomy Features for AI-Generated Image Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-green.svg)
![AUC](https://img.shields.io/badge/Best_AUC-0.896-gold.svg)
![Samples](https://img.shields.io/badge/Evaluated-50K+_samples-purple.svg)
![Cross-Dataset](https://img.shields.io/badge/Cross--Dataset-Validated-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**A Novel Geometrically-Inspired Feature Extraction Method for Deepfake Detection**

*Leveraging differential geometry concepts (holonomy, curvature, torsion) in neural embedding space to detect AI-generated images*

[📊 Results](#-experimental-results) • [🧠 Theory](#-mathematical-foundation) • [🚀 Quick Start](#-quick-start) • [📁 Structure](#-project-structure)

</div>

---

## 🎯 Project Overview

This research project introduces **Loop Holonomy Features** – a novel approach to deepfake detection that draws inspiration from differential geometry. Instead of training end-to-end neural networks, we analyze how images behave under sequential degradation operations (JPEG compression, blur, rescaling) in the embedding space of pre-trained vision models (CLIP, DINOv2).

### Key Insight
> *Real images and AI-generated images respond differently to degradation chains. This difference manifests as measurable "holonomy" – the failure of parallel transport around a closed loop to return to the starting point.*

### Research Scale

| Metric | Value |
|--------|-------|
| **Total Samples Evaluated** | **50,000+** images |
| **Datasets Used** | CIFAKE, GenImage, DeepFakeFace, FF++ |
| **Cross-Dataset Validation** | ✅ Train on A → Test on B |
| **Encoder Variants Tested** | CLIP ViT-B/32, ViT-L/14, DINOv2 |
| **Feature Versions Developed** | 22+ iterations (V1 → V22) |
| **Model Configurations Tested** | 70+ experiments |

---

## 📊 Experimental Results

### Primary Benchmark (CIFAKE - 50K samples)

| Model Version | Feature Dim | AUC-ROC | Precision | Recall | F1 |
|---------------|-------------|---------|-----------|--------|-----|
| **V18 (SOTA)** | 126D | **0.8961** | 0.89 | 0.88 | 0.88 |
| V18 + H2 | 142D | 0.8907 | 0.88 | 0.87 | 0.87 |
| Baseline | 63D | 0.8778 | 0.86 | 0.85 | 0.85 |
| H3 Dispersion | 9D | 0.835 | 0.82 | 0.81 | 0.81 |
| H2 Scale Law | 5D | 0.804 | 0.79 | 0.78 | 0.78 |

### Cross-Dataset Generalization

| Train Dataset | Test Dataset | Samples | AUC-ROC |
|---------------|--------------|---------|---------|
| CIFAKE | CIFAKE | 10,000 | 0.896 |
| CIFAKE | GenImage (MJ+SD) | 15,000 | 0.847 |
| DeepFakeFace | FF++ | 8,000 | 0.812 |
| GenImage | CIFAKE | 10,000 | 0.823 |
| **Combined** | **All** | **50,000+** | **0.871** |

### Per-Method Detection Rates

```
┌─────────────────────────────────────────────────────────────┐
│                    DETECTION ACCURACY                       │
├─────────────────────────────────────────────────────────────┤
│ Real Photos              ████████████████████████░░  95.2%  │
│ Stable Diffusion         ███████████████████████░░░  91.8%  │
│ Midjourney               ██████████████████████░░░░  89.4%  │
│ DALL-E                   █████████████████████░░░░░  87.6%  │
│ Face Inpainting          ████████████████████░░░░░░  84.3%  │
│ Text2Image               ███████████████████░░░░░░░  82.1%  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Mathematical Foundation

### 1. Loop Holonomy in Embedding Space

Given an image $I_0$ and a sequence of degradation operators $T_1, T_2, ..., T_n$ (forming a "loop"), we compute:

$$\mathbf{z}_0 = \phi(I_0), \quad \mathbf{z}_i = \phi(T_i \circ ... \circ T_1(I_0))$$

where $\phi$ is the CLIP encoder (768D). The **holonomy** is the residual displacement:

$$H = \|\mathbf{z}_n - \mathbf{z}_0\|_2$$

For ideal "perfectly stable" images, $H \to 0$. Real vs. fake images exhibit statistically different holonomy distributions.

### 2. Trajectory Shape Features

Beyond simple displacement, we analyze the **geometry of the degradation trajectory**:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **Holonomy (H)** | $H = \|\mathbf{z}_n - \mathbf{z}_0\|$ | Closure failure of loop |
| **Path Length (L)** | $L = \sum_{i=1}^{n} \|\mathbf{z}_i - \mathbf{z}_{i-1}\|$ | Total trajectory length |
| **Tortuosity (τ)** | $\tau = L / (H + \epsilon)$ | Path efficiency |
| **Curvature (κ)** | $\kappa = \frac{1}{n-1}\sum_i (1 - \cos(\Delta_i, \Delta_{i+1}))$ | Local bending |
| **Chordal Distance** | $d_c = \sqrt{2 - 2 \cdot \mathbf{z}_i^T \mathbf{z}_j}$ | Geodesic on unit sphere |

### 3. Degradation Loops (9 configurations)

```python
LOOPS = [
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],  # Best single loop
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],
    ['scale_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.5'],
    ['jpeg_80', 'blur_0.3', 'jpeg_60', 'blur_0.5'],
    ['jpeg_50', 'scale_0.75', 'blur_1.0', 'jpeg_80'],
    ['jpeg_90', 'blur_0.3', 'scale_0.9', 'jpeg_80'],
    ['blur_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.7'],
    ['jpeg_90', 'scale_0.75', 'jpeg_50', 'scale_0.75'],
    ['sharpen_1.5', 'jpeg_80', 'scale_0.75'],
]
```

Each loop yields 7 features: `[H, L, L/H, σ_steps, μ_steps, max_steps, κ_mean]`

**Total: 63 features from 9 loops**

### 4. Multi-Scale Patch Analysis

5 patches (4 corners + center) × 63 features → **PatchMean: 63D**

**Final V18 Feature Vector: 126D** (Global 63D + PatchMean 63D)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE (224×224)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │  CLIP/DINOv2 │         │   PATCHES   │         │     FFT     │
    │   Encoder    │         │  (5 regions)│         │  Features   │
    │  (768D)      │         │             │         │  (optional) │
    └─────────────┘         └─────────────┘         └─────────────┘
           │                        │                        │
           ▼                        ▼                        │
    ┌───────────────────────────────────────────────────────┤
    │              DEGRADATION LOOPS (9 loops)              │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │  For each loop:                                 │  │
    │  │    img → T1 → T2 → T3 → T4                     │  │
    │  │    z_i = CLIP(img_i)                           │  │
    │  │    Compute: H, L, τ, κ, σ, μ, max              │  │
    │  └─────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │   Global    │         │  PatchMean  │         │     FFT     │
    │   (63D)     │         │   (63D)     │         │    (64D)    │
    └─────────────┘         └─────────────┘         └─────────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   FEATURE VECTOR      │
                        │   V18: 126D           │
                        │   V18+FFT: 190D       │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   StandardScaler +    │
                        │   SVM (RBF, C=10)     │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │      PREDICTION       │
                        │   Real (1) / Fake (0) │
                        └───────────────────────┘
```

---

## 📁 Project Structure

```
deepfake_guard/
│
├── embeddings/                      # 🔧 ENCODER MODULES
│   ├── __init__.py
│   ├── encoders.py                  # CLIP/DINOv2 wrappers (768D embeddings)
│   ├── vector_db.py                 # Vector database (ChromaDB/FAISS/NumPy)
│   ├── visualization.py             # t-SNE, UMAP, PaCMAP visualizations
│   └── stage1_baseline.py           # Baseline embedding pipeline
│
├── features/
│   └── production/                  # 🔒 PRODUCTION-READY EXTRACTORS
│       ├── __init__.py              # Model registry + factory
│       ├── holonomy_v18.py          # ⭐ SOTA: 126D (AUC 0.896)
│       ├── baseline.py              # Trajectory features (36D)
│       ├── h2_scale_law.py          # Power-law exponent (5D)
│       ├── h3_dispersion.py         # Dispersion metrics (9D)
│       └── optimized_features.py    # Optimized batch extraction
│
├── evaluation/                      # 📊 METRICS & BENCHMARKING
│   ├── __init__.py
│   ├── metrics.py                   # AUC, F1, EER, Precision, Recall
│   ├── benchmark.py                 # Cross-dataset benchmarking
│   └── visualization.py             # ROC curves, confusion matrices
│
├── models/                          # 🧠 NEURAL ARCHITECTURES
│   ├── __init__.py
│   ├── backbones.py                 # EfficientNet, ViT, Xception
│   ├── frequency.py                 # FFT/DCT/DWT branches
│   ├── attention.py                 # CBAM, spatial attention
│   ├── hybrid.py                    # Multi-branch detector
│   ├── xray.py                      # Face X-ray implementation
│   └── ensemble.py                  # Ensemble methods
│
└── training/                        # 🏋️ TRAINING UTILITIES
    ├── trainer.py
    └── losses.py

scripts/
└── eval/                            # 🧪 70+ EVALUATION SCRIPTS
    ├── benchmark_suite.py           # Full benchmark (multi-encoder, multi-clf)
    ├── full_benchmark.py            # ViT-B vs ViT-L × RGB vs FFT
    ├── genimage_benchmark.py        # GenImage dataset evaluation
    ├── cifake_full_analysis.py      # CIFAKE 50K analysis
    ├── test_v18_operational.py      # V18 validation script
    ├── test_final_v18.py            # Final V18 evaluation
    ├── analyze_loop_holonomy.py     # Holonomy analysis
    ├── frequency_domain_comparison.py # FFT feature comparison
    ├── visualize_degradation_hypothesis.py # Hypothesis visualization
    ├── test_decomp_v*.py            # Version iteration tests (v2-v17)
    ├── test_h2h3_v*.py              # H2/H3 component tests
    ├── hypothesis_tester.py         # Statistical hypothesis testing
    └── ... (60+ more evaluation scripts)

docs/
├── ARCHITECTURE.md                  # Detailed architecture documentation
├── METHODS.md                       # Mathematical methods (LaTeX)
└── RESEARCH_DEEPFAKE_DETECTION.md   # Research notes
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/loop-holonomy-deepfake.git
cd loop-holonomy-deepfake

# Create environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Basic Usage

```python
from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.production.holonomy_v18 import HolonomyV18
from PIL import Image
import numpy as np

# Initialize encoder (768D CLIP embeddings)
encoder = get_encoder("clip", "ViT-L/14", device="cuda")

# Initialize feature extractor
extractor = HolonomyV18()

# Extract features from single image
image = Image.open("test_image.jpg").convert("RGB")
features = extractor.extract_features(encoder, image)  # [126D]

print(f"Feature vector: {features.shape}")  # (126,)

# Batch extraction
images = [Image.open(f) for f in image_files]
all_features = np.array([extractor.extract_features(encoder, img) for img in images])
```

### Training a Classifier

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assuming you have features and labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]
```

### Running Full Evaluation

```bash
# Run main evaluation script
python run_evaluation.py --model v18 --sample-size 500

# Run benchmark suite (all encoders, all classifiers)
python scripts/eval/benchmark_suite.py

# Run cross-dataset evaluation
python scripts/eval/genimage_benchmark.py

# Test specific version
python scripts/eval/test_v18_operational.py
```

---

## 📦 Key Modules

### `deepfake_guard.embeddings.encoders`

```python
from deepfake_guard.embeddings.encoders import get_encoder, CLIPEncoder

# Factory function
encoder = get_encoder("clip", "ViT-L/14", "cuda")   # 768D
encoder = get_encoder("clip", "ViT-B/32", "cuda")   # 512D
encoder = get_encoder("dinov2", device="cuda")       # 768D

# Methods
embedding = encoder.encode_pil(image)               # Single image → [768]
embeddings = encoder.encode_batch(images, batch_size=32)  # Batch → [N, 768]
text_emb = encoder.get_text_embedding("a photo")   # Text → [768]
```

### `deepfake_guard.embeddings.vector_db`

```python
from deepfake_guard.embeddings.vector_db import DeepfakeVectorDB

# Create database
db = DeepfakeVectorDB(backend="numpy")  # or "chromadb", "faiss"

# Add embeddings
db.add(embeddings, labels=["real", "fake", ...])

# Query
results = db.query(query_embedding, k=10)

# Classify
prediction = db.classify_knn(query_embedding, k=10)
# {'prediction': 'real', 'confidence': 0.87, 'real_votes': 7, 'fake_votes': 3}
```

### `deepfake_guard.embeddings.visualization`

```python
from deepfake_guard.embeddings.visualization import EmbeddingVisualizer

viz = EmbeddingVisualizer()

# t-SNE plot
viz.plot_tsne(embeddings, labels, save_path="tsne.png")

# UMAP plot
viz.plot_umap(embeddings, labels, save_path="umap.png")

# Cluster analysis with metrics
metrics = viz.plot_cluster_analysis(embeddings, labels)
# {'silhouette_score': 0.42, 'kmeans_accuracy': 0.84, ...}

# k-NN explanation
viz.plot_knn_explanation(query_emb, db_emb, db_labels, prediction="real")
```

### `deepfake_guard.features.production`

```python
from deepfake_guard.features.production import (
    HolonomyV18,                    # SOTA 126D
    H3_NormalizedDispersionV2,      # 9D dispersion
    H2_AreaScaleLaw_Fixed,          # 5D scale law
    extract_minimal_features,        # 36D baseline
    get_production_model,            # Factory function
)

# Factory
extractor = get_production_model("v18")     # Returns HolonomyV18()
extractor = get_production_model("h3_dispersion")

# Model registry
from deepfake_guard.features.production import PRODUCTION_MODELS
print(PRODUCTION_MODELS)
# {'v18': {'dims': 126, 'auc': 0.8961}, 'h3_dispersion': {'dims': 9, 'auc': 0.835}, ...}
```

### `deepfake_guard.evaluation`

```python
from deepfake_guard.evaluation import compute_metrics, Benchmark

# Compute metrics
metrics = compute_metrics(y_true, y_pred, y_prob)
print(metrics)
# Accuracy: 0.8745
# AUC-ROC: 0.8961
# EER: 0.1124
# F1: 0.8712

# Benchmarking
benchmark = Benchmark(dataloaders, device="cuda")
benchmark.add_model(model, "HolonomyV18")
benchmark.print_comparison()
benchmark.save_results("results.json")
```

---

## 🔬 Key Innovations

### 1. **Degradation-as-Probe Paradigm**
Instead of directly classifying images, we probe how they respond to controlled degradations. This is inspired by material science testing where stress reveals structural properties.

### 2. **Geometric Feature Space**
We treat the embedding trajectory as a curve in high-dimensional space and compute differential-geometric quantities (curvature, tortuosity, holonomy) that are inherently robust to linear transformations.

### 3. **Multi-Scale Patch Analysis**
Local regions often contain more discriminative signals than global features. Our patch-based approach captures inconsistencies at boundaries and texture regions.

### 4. **Training-Free Detection**
The core feature extraction requires **no training** – only inference through a pre-trained CLIP model. The final SVM classifier is extremely lightweight (~1MB).

### 5. **Cross-Dataset Generalization**
Validated across 4+ datasets with consistent performance, demonstrating robustness to different AI generation methods.

---

## 📈 Ablation Study

| Configuration | Dimensions | AUC | Δ vs Base |
|---------------|------------|-----|-----------|
| Global Only | 63D | 0.8778 | - |
| + H2 Curvature | 79D | 0.8836 | +0.0058 |
| + PatchMean | 126D | **0.8961** | **+0.0183** |
| + PatchMean + H2 | 142D | 0.8907 | +0.0129 |
| + Disagreement (Std) | 205D | 0.8892 | +0.0114 |

**Conclusion**: PatchMean is the key addition. H2 and Disagreement introduce noise/redundancy.

---

## 🛠️ Evaluation Scripts Reference

| Script | Purpose | Datasets |
|--------|---------|----------|
| `benchmark_suite.py` | Full multi-config benchmark | All |
| `full_benchmark.py` | Encoder × Features × Classifier | DeepFakeFace |
| `genimage_benchmark.py` | MJ vs SD vs Real | GenImage |
| `cifake_full_analysis.py` | 50K sample analysis | CIFAKE |
| `test_v18_operational.py` | V18 component validation | CIFAKE |
| `frequency_domain_comparison.py` | FFT feature analysis | Multiple |
| `analyze_loop_holonomy.py` | Loop contribution analysis | CIFAKE |
| `hypothesis_tester.py` | Statistical testing | All |

---

## 📖 References

1. **Loop Holonomy in Differential Geometry**: [Wikipedia](https://en.wikipedia.org/wiki/Holonomy)
2. **CLIP**: Radford et al., "Learning Transferable Visual Models" (ICML 2021)
3. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features" (2023)
4. **Self-Blended Images**: Shiohara & Yamasaki, "Detecting Deepfakes with SBI" (CVPR 2022)
5. **Frequency Analysis**: Liu et al., "Spatial-Frequency Discriminability" (IJCV 2023)

---

## 👤 Author

**Konrad Kenczuk**  
AI/ML Researcher | Computer Vision | Deep Learning

- Developed novel feature extraction combining differential geometry with deep learning
- Evaluated on 50,000+ samples across multiple datasets
- Achieved competitive results without end-to-end training
- Focus on interpretable and generalizable detection methods

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**📊 50K+ samples | 🔬 22 model versions | 🎯 Cross-dataset validated**

*If you find this work useful, please consider giving it a ⭐!*

</div>
