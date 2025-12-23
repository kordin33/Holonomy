# Loop Holonomy: Geometric Features for Deepfake Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-grey)
![Status](https://img.shields.io/badge/Status-Production-green)

## Abstract

This research introduces **Loop Holonomy Features**, a novel method for detecting AI-generated imagery by analyzing the geometric properties of image traversals in pre-trained neural embedding manifolds. Unlike traditional supervised approaches that rely on end-to-end training, this method measures the **holonomic failure**—the discrepancy between defined transformations—when an image undergoes a cyclic sequence of degradation operators (e.g., scaling, compression, Gaussian blur). Empirical evaluation on over **50,000 samples** across diverse datasets (CIFAKE, GenImage, FF++) demonstrates that real and synthetic images exhibit statistically distinct holonomy signatures, achieving an AUC of **0.896** with a lightweight 126-dimensional feature vector.

---

## 1. Mathematical Framework

The core hypothesis rests on the observation that the latent manifold $\mathcal{M}$ of a Vision Transformer (e.g., CLIP ViT-L/14) preserves isometric properties for organic images differently than for synthetic ones. We define a degradation loop $\gamma$ as a sequence of operators acting on the image space.

### 1.1 Discrete Holonomy
Let $I_0$ be the initial image state and $\phi: \mathcal{I} \to \mathbb{R}^d$ be the encoder mapping image space to a $d$-dimensional embedding space. A loop consists of $n$ transformations $T_1, \dots, T_n$ such that ideally $T_n \circ \dots \circ T_1 \approx Identity$.

The **Holonomy ($H$)** is defined as the $L_2$ norm of the residual displacement vector after traversing the loop:

$$ H(\gamma) = \| \mathbf{z}_n - \mathbf{z}_0 \|_2 $$

Where $\mathbf{z}_0 = \phi(I_0)$ and $\mathbf{z}_k = \phi(T_k(\mathbf{z}_{k-1}))$.

### 1.2 Trajectory Geometry
Beyond simple displacement, we analyze the differential geometry of the discrete curve formed by the sequence $\{\mathbf{z}_0, \dots, \mathbf{z}_n\}$.

**Tortuosity ($\tau$)**
Measures the efficiency of the path in the embedding space. A higher tortuosity indicates the image is "struggling" against the degradation operators.

$$ \tau = \frac{L}{H + \epsilon} = \frac{\sum_{i=1}^{n} \| \mathbf{z}_i - \mathbf{z}_{i-1} \|_2}{\| \mathbf{z}_n - \mathbf{z}_0 \|_2 + \epsilon} $$

**Discrete Curvature ($\kappa$)**
Quantifies local bending of the trajectory. High curvature implies high sensitivity to specific degradation transitions.

$$ \kappa = \frac{1}{n-1} \sum_{i=1}^{n-1} \left( 1 - \frac{(\mathbf{z}_{i} - \mathbf{z}_{i-1}) \cdot (\mathbf{z}_{i+1} - \mathbf{z}_{i})}{\| \mathbf{z}_{i} - \mathbf{z}_{i-1} \| \| \mathbf{z}_{i+1} - \mathbf{z}_{i} \|} \right) $$

**Chordal Distance ($d_c$)**
Utilizing the hyperspherical nature of the contrastive embedding space (normalized vectors), we compute the chordal distance:

$$ d_c(\mathbf{z}_i, \mathbf{z}_j) = \sqrt{ 2 - 2 \langle \mathbf{z}_i, \mathbf{z}_j \rangle } $$

---

## 2. Methodology & Architecture

The system utilizes a **Probe-and-Measure** paradigm. The image is not just "seen" by the classifier; it is actively probed with stress tests (degradations) to reveal its structural integrity.

### 2.1 Feature Extraction Pipeline

```
[ INPUT IMAGE ]
      │
      ├── (Global Branch) ───────────────────────────┐
      │                                             │
      ├── (Patch Branch: 4 Corners + Center) ───────┼──► [ Aggregation ]
      │                                             │
      ▼                                             ▼
[ DEGRADATION LOOPS ]                         [ ENCODER ]
Sequence of transformations:                  CLIP ViT-L/14 or DINOv2
T1 → T2 → T3 → ...                            (Frozen Weights)
      │
      ▼
[ GEOMETRIC ANALYSIS ]
Compute per-loop metrics:
H, L, τ, κ, σ_step, μ_step, max_step
      │
      ▼
[ FEATURE FUSION ]
Global Holonomy (63D) || Patch Mean Holonomy (63D)
Total: 126 Dimensions
      │
      ▼
[ CLASSIFICATION ]
Support Vector Machine (RBF Kernel)
Output: P(Real | Image)
```

### 2.2 Degradation Operators
We utilize 9 distinct transformation loops designed to isolate specific artifacts:
*   **Frequency Cutoff**: JPEG compression ($Q=50..90$) combined with Gaussian Blur.
*   **Resampling**: Bicubic up/down-scaling to probe anti-aliasing consistency.
*   **Identity Restoration**: Chains intended to return to the original pixel state.

---

## 3. Experimental Results

Evaluations were conducted on a diverse set of over **50,000 images** ranging from classical GANs (StyleGAN) to modern Diffusion Models (Midjourney v6, Stable Diffusion XL, DALL-E 3).

### 3.1 Quantitative Benchmarks

| Model Configuration | Feature Dim | AUC-ROC | Precision | Recall |
|:-------------------|:-----------:|:-------:|:---------:|:------:|
| **Holonomy V18 (SOTA)** | **126** | **0.896** | **0.89** | **0.88** |
| Holonomy V18 + H2 | 142 | 0.891 | 0.88 | 0.87 |
| Baseline (Global Only) | 63 | 0.878 | 0.86 | 0.85 |
| H3 Dispersion | 9 | 0.835 | 0.82 | 0.81 |

### 3.2 Cross-Dataset Generalization
Models trained on **CIFAKE** and evaluated on unseen architectures (Zero-Shot setting).

| Training Set | Test Set | Generalization Score (AUC) |
|:-------------|:---------|:--------------------------:|
| CIFAKE | **GenImage (Midjourney)** | 0.847 |
| CIFAKE | **DeepFakeFace** | 0.823 |
| DeepFakeFace | FF++ (Video Frames) | 0.812 |

---

## 4. Usage

### Feature Extraction
```python
from deepfake_guard.features.production import HolonomyV18, get_encoder
from PIL import Image

# 1. Initialize Encoder (ViT-L/14)
encoder = get_encoder("clip", "ViT-L/14", device="cuda")

# 2. Extract Geometric Features
extractor = HolonomyV18()
image = Image.open("sample.jpg")
features = extractor.extract_features(encoder, image)

print(f"Feature Vector Shape: {features.shape}") # (126,)
```

### Evaluation
```bash
# Run full benchmark suite
python scripts/eval/benchmark_suite.py

# Evaluate specific dataset
python run_evaluation.py --data_dir ./data/cifake --model v18
```

---

## 5. References

1.  **Metric Space**: Gromov, M. (2007). *Metric Structures for Riemannian and Non-Riemannian Spaces*. Birkhäuser.
2.  **CLIP**: Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
3.  **DINOv2**: Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193.
4.  **Self-Blended Images**: Shiohara, K., & Yamasaki, T. (2022). *Detecting Deepfakes with Self-Blended Images*. CVPR.
5.  **Frequency Analysis**: Frank, J., et al. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML.

---

## Author

**Konrad Izdebski**
