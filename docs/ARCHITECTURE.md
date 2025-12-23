# 🔬 Architecture Deep Dive

## Loop Holonomy Theory

### What is Holonomy?

In differential geometry, **holonomy** describes the behavior of parallel transport around a closed loop. When you transport a vector around a closed path on a curved surface, it may not return to its original orientation – this "rotation" is the holonomy.

### Application to Image Analysis

We adapt this concept to neural embedding spaces:

1. **Embedding Space as Manifold**: CLIP/DINOv2 embeddings live on a high-dimensional manifold (approximately a hypersphere due to L2 normalization)

2. **Degradation as Transport**: Sequential image transformations (JPEG → blur → scale) define a "path" through embedding space

3. **Holonomy Measurement**: The failure of this path to return to the origin reveals image characteristics

```
         z_0 (original)
        /  \
       /    \
      z_1    z_n (after loop)
       \    /
        \  /
         H = ||z_n - z_0||  ← This is the holonomy
```

### Why It Works for Deepfake Detection

AI-generated images and real photographs respond differently to degradations:

| Property | Real Images | AI-Generated |
|----------|-------------|--------------|
| **Frequency spectrum** | Natural 1/f noise | Often missing high frequencies |
| **JPEG response** | Predictable DCT behavior | Unusual artifact patterns |
| **Interpolation** | Natural sub-pixel variation | Grid-like patterns |
| **Texture** | Local irregularities | Over-smoothness or repetition |

These differences manifest in the **holonomy signature**.

---

## Feature Extraction Pipeline

### Stage 1: Degradation Loops

We apply 9 carefully designed degradation loops:

```python
LOOPS = [
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],  # Loop 1
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],    # Loop 2
    # ... 7 more loops
]
```

Each loop creates a trajectory in embedding space.

### Stage 2: Trajectory Analysis

For each trajectory z_0 → z_1 → ... → z_n:

```
┌────────────────────────────────────────────────────────┐
│                    TRAJECTORY METRICS                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│   H = ||z_n - z_0||           Holonomy (closure gap)   │
│                                                        │
│   L = Σ||z_{i+1} - z_i||      Path length              │
│                                                        │
│   τ = L / H                  Tortuosity (efficiency)   │
│                                                        │
│   κ = mean(1 - cos(Δ_i, Δ_{i+1}))  Curvature          │
│                                                        │
│   σ, μ, max of step sizes    Statistical moments       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Stage 3: Multi-Scale Analysis

Apply the same analysis at multiple scales:

```
┌─────────────────┐
│ GLOBAL (full)   │ → 63D features
└─────────────────┘

┌─────┬─────┐         ┌─────┐
│ TL  │ TR  │         │ CTR │ → 5 patches
├─────┼─────┤   +     └─────┘
│ BL  │ BR  │
└─────┴─────┘
        ↓
   Patch Mean → 63D features

TOTAL: 126D feature vector
```

---

## Model Architecture Comparison

### V18 (Production SOTA) - 126D

```
Input Image
    │
    ├────────────────┬────────────────┐
    │                │                │
    ▼                ▼                ▼
 GLOBAL           PATCHES          (unused)
 Analysis         (5x)              
    │                │                
    │           ┌────┴────┐          
    │           │  MEAN   │          
    │           └────┬────┘          
    ▼                ▼                
  [63D]            [63D]             
    │                │                
    └────────┬───────┘                
             │                        
             ▼                        
          [126D]                      
             │                        
       StandardScaler                 
             │                        
        SVM (RBF)                     
             │                        
       Real / Fake                    
```

### Why Not More Complex?

We tested many configurations:

| Config | Dims | AUC | Result |
|--------|------|-----|--------|
| Global only | 63 | 0.878 | Baseline |
| + H2 curvature | 79 | 0.884 | +0.6% |
| + PatchMean | 126 | **0.896** | **+1.8%** ✓ |
| + PatchMean + H2 | 142 | 0.891 | -0.5% ✗ |
| + Disagreement | 205 | 0.889 | -0.7% ✗ |

**Conclusion**: Simpler is better. Additional features add noise.

---

## Encoder Details

### CLIP ViT-L/14

```
┌─────────────────────────────────────────┐
│           CLIP Vision Encoder           │
├─────────────────────────────────────────┤
│ Architecture:  Vision Transformer       │
│ Patch Size:    14 × 14 pixels           │
│ Input Size:    224 × 224 RGB            │
│ Hidden Dim:    1024                     │
│ Output Dim:    768                      │
│ Parameters:    ~300M                    │
│ Normalization: L2 on output             │
└─────────────────────────────────────────┘
```

### Why CLIP?

1. **Pre-trained on 400M image-text pairs** – robust representations
2. **Contrastive learning** – captures semantic AND visual features
3. **L2 normalized outputs** – natural for geometric analysis
4. **No fine-tuning needed** – zero-shot transfer

---

## Computational Analysis

### Per-Image Cost

```
Operations per image (V18):
├── Global features (1 image, 9 loops)
│   └── 9 loops × ~5 transforms = 45 CLIP forward passes
│
└── Patch features (5 patches)
    └── 5 patches × 9 loops × ~5 transforms = 225 CLIP forward passes
    
TOTAL: ~270 CLIP forward passes per image
       ~50 ms on RTX 3080
```

### Optimization Opportunities

1. **Batch processing**: Collect all transforms, single batch encode
2. **Caching**: Store intermediate embeddings
3. **Loop pruning**: Use fewer loops (top 5 contribute 90% of signal)
4. **Patch reduction**: 3 patches instead of 5

---

## Mathematical Derivations

### Chordal Distance

For L2-normalized vectors a, b on unit sphere:

```
d_chordal(a, b) = ||a - b||_2
                = √(||a||² + ||b||² - 2a·b)
                = √(1 + 1 - 2·cos(θ))
                = √(2 - 2·cos(θ))
                = √(2(1 - cos(θ)))
                = √(2) · √(1 - cos(θ))
                = 2·sin(θ/2)
```

For small angles: d_chordal ≈ θ (geodesic distance on sphere)

### Curvature Estimation

Curvature measures local "bending" of the trajectory:

```
κ_i = angle between consecutive displacements
    = arccos(Δ_i · Δ_{i+1} / (||Δ_i|| · ||Δ_{i+1}||))

where Δ_i = z_{i+1} - z_i

Mean curvature = (1/n) Σ κ_i
```

High curvature → trajectory is "wiggly" (often sign of instability)

### Tortuosity

Ratio of path length to direct distance:

```
τ = L / H = (Σ||Δ_i||) / ||z_n - z_0||
```

- τ ≈ 1: Nearly straight path
- τ >> 1: Very curved/inefficient path

Real images tend to have higher tortuosity (more stable to degradations).

---

## Empirical Insights

### Feature Importance (from SVM weights)

```
Global Features:
├── H (holonomy): 35% contribution
├── L/H (tortuosity): 25% contribution  
├── max_step: 15% contribution
├── mean_step: 10% contribution
└── others: 15%

Patch Features:
├── Most informative: center patch
├── Least informative: corner patches (often background)
└── Mean aggregation >> Std aggregation
```

### Per-Loop Contribution

```
Loop 1 (scale-blur-jpeg): 18% ← Most discriminative
Loop 4 (double-jpeg):     15%
Loop 5 (heavy degrad):    14%
Loop 3 (multi-scale):     12%
...
```

---

## Future Directions

### Potential Improvements

1. **Learnable loops**: Optimize transform sequences end-to-end
2. **Multi-encoder fusion**: Combine CLIP + DINOv2 holonomy
3. **Temporal extension**: Apply to video (frame-to-frame holonomy)
4. **Adversarial robustness**: Test against anti-detection attacks

### Theoretical Questions

1. Why does patch MEAN outperform patch STD?
2. What is the optimal number of degradation steps?
3. Can we derive theoretical bounds on discriminability?
4. Connection to manifold curvature of training data?
