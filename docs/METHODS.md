# 📐 Mathematical Methods Summary

## Overview

This document provides a concise mathematical reference for all methods used in the Loop Holonomy deepfake detection system.

---

## 1. Embedding Space Geometry

### 1.1 Vision Encoder Mapping

The CLIP encoder maps images to a high-dimensional embedding space:

$$\phi: \mathcal{I} \to \mathbb{R}^d, \quad d = 768 \text{ (ViT-L/14)}$$

After L2 normalization, embeddings lie on the unit hypersphere $\mathbb{S}^{d-1}$.

### 1.2 Distance Metrics

**Chordal Distance** (used for holonomy computation):
$$d_c(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{2 - 2\mathbf{a}^\top\mathbf{b}}$$

**Geodesic Distance** (arc length on sphere):
$$d_g(\mathbf{a}, \mathbf{b}) = \arccos(\mathbf{a}^\top\mathbf{b})$$

**Relation**: For $\theta = d_g$: $d_c = 2\sin(\theta/2) \approx \theta$ for small $\theta$.

---

## 2. Degradation Operators

### 2.1 JPEG Compression
Simulates DCT-based lossy compression:
$$T_{jpeg}(I; q) = \text{IDCT}(\text{Quantize}(\text{DCT}(I); q))$$

### 2.2 Gaussian Blur
Convolution with Gaussian kernel:
$$T_{blur}(I; \sigma) = I * G_\sigma, \quad G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 2.3 Rescaling
Downscale then upscale:
$$T_{scale}(I; s) = \text{Upsample}(\text{Downsample}(I; s); 1/s)$$

---

## 3. Loop Holonomy Features

### 3.1 Degradation Loop

A loop $\mathcal{L}$ is a sequence of transformations:
$$\mathcal{L} = [T_1, T_2, \ldots, T_n]$$

Applied sequentially:
$$I_0 \xrightarrow{T_1} I_1 \xrightarrow{T_2} \cdots \xrightarrow{T_n} I_n$$

### 3.2 Embedding Trajectory

$$\mathbf{z}_i = \phi(I_i), \quad i = 0, 1, \ldots, n$$

### 3.3 Holonomy (H)

The residual after completing the loop:
$$H = d_c(\mathbf{z}_0, \mathbf{z}_n) = \|\mathbf{z}_n - \mathbf{z}_0\|_2$$

**Interpretation**: $H = 0$ for perfect reconstruction; $H > 0$ indicates irreversible changes.

### 3.4 Path Length (L)

Total distance traveled along trajectory:
$$L = \sum_{i=0}^{n-1} d_c(\mathbf{z}_i, \mathbf{z}_{i+1})$$

### 3.5 Tortuosity (τ)

Ratio of path length to direct distance (efficiency measure):
$$\tau = \frac{L}{H + \epsilon}, \quad \epsilon = 10^{-8}$$

- $\tau \approx 1$: Near-straight path
- $\tau \gg 1$: Highly curved path

### 3.6 Curvature (κ)

Average bending of trajectory:
$$\kappa = \frac{1}{n-1} \sum_{i=1}^{n-1} \left(1 - \frac{\Delta_i \cdot \Delta_{i+1}}{\|\Delta_i\| \|\Delta_{i+1}\|}\right)$$

where $\Delta_i = \mathbf{z}_{i+1} - \mathbf{z}_i$.

### 3.7 Step Statistics

- **Mean step**: $\mu = \frac{1}{n}\sum_i \|\Delta_i\|$
- **Step std**: $\sigma = \sqrt{\frac{1}{n}\sum_i (\|\Delta_i\| - \mu)^2}$
- **Max step**: $\max_i \|\Delta_i\|$

---

## 4. Feature Vector Construction

### 4.1 Per-Loop Features (7D)

For each loop $\mathcal{L}_j$:
$$\mathbf{f}_j = [H, L, \tau, \sigma, \mu, \max, \kappa] \in \mathbb{R}^7$$

### 4.2 Global Features (63D)

Concatenation over 9 loops:
$$\mathbf{f}_{global} = [\mathbf{f}_1, \mathbf{f}_2, \ldots, \mathbf{f}_9] \in \mathbb{R}^{63}$$

### 4.3 Patch Features

For 5 patches $\{p_1, \ldots, p_5\}$:
$$\mathbf{f}_{patch,k} = \text{LoopFeatures}(p_k) \in \mathbb{R}^{63}$$

**PatchMean**: $\mathbf{f}_{pm} = \frac{1}{5}\sum_{k=1}^5 \mathbf{f}_{patch,k}$

### 4.4 Final Feature Vector (V18)

$$\mathbf{f}_{V18} = [\mathbf{f}_{global}, \mathbf{f}_{pm}] \in \mathbb{R}^{126}$$

---

## 5. Classification

### 5.1 Preprocessing

Standardization:
$$\tilde{\mathbf{f}} = \frac{\mathbf{f} - \boldsymbol{\mu}_{train}}{\boldsymbol{\sigma}_{train}}$$

### 5.2 SVM with RBF Kernel

$$k(\mathbf{f}_i, \mathbf{f}_j) = \exp\left(-\gamma \|\mathbf{f}_i - \mathbf{f}_j\|^2\right)$$

Decision function:
$$y = \text{sign}\left(\sum_i \alpha_i y_i k(\mathbf{f}, \mathbf{f}_i) + b\right)$$

**Hyperparameters**:
- $C \in \{0.1, 1, 10, 100\}$
- $\gamma \in \{\text{scale}, 0.01\}$

Selected via 5-fold cross-validation on AUC-ROC.

---

## 6. H2: Scale Law Features

### 6.1 Multi-Intensity Probes

For intensity levels $(q_1, \sigma_1), \ldots, (q_K, \sigma_K)$:

$$H_k = \|\phi(T_{blur}(T_{jpeg}(I; q_k); \sigma_k)) - \phi(I)\|$$

### 6.2 Power Law Fit

$$\log H \approx \alpha \log s + c$$

where $s$ = measured degradation strength.

**Features**: $[\alpha, R^2, \sigma_{residual}, \bar{H}, \text{stability}]$

---

## 7. H3: Dispersion Features

### 7.1 Cosine Dispersion

$$D_{cos} = \frac{2}{K(K-1)} \sum_{i<j} (1 - \cos(\mathbf{z}_i, \mathbf{z}_j))$$

### 7.2 Path Holonomy (Cosine)

$$D_{path} = \frac{1 - \cos(\mathbf{z}_0, \mathbf{z}_K)}{\sum_i (1 - \cos(\mathbf{z}_i, \mathbf{z}_{i+1}))}$$

### 7.3 Covariance Trace

$$D_{cov} = \frac{\text{tr}(\text{Cov}(\mathbf{Z}))}{\|\bar{\mathbf{z}}\|^2}$$

where $\mathbf{Z} = [\mathbf{z}_1, \ldots, \mathbf{z}_K]$ (excluding $\mathbf{z}_0$).

---

## 8. Evaluation Metrics

### 8.1 AUC-ROC

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

### 8.2 Equal Error Rate

$$\text{EER} = \text{FPR}^* = 1 - \text{TPR}^*$$

where $\text{FPR}^* = \text{FNR}^*$.

### 8.3 F1 Score

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## Symbol Reference

| Symbol | Meaning |
|--------|---------|
| $\phi$ | Vision encoder (CLIP) |
| $\mathbf{z}$ | Embedding vector |
| $d$ | Embedding dimension (768) |
| $T$ | Degradation transform |
| $\mathcal{L}$ | Degradation loop |
| $H$ | Holonomy |
| $L$ | Path length |
| $\tau$ | Tortuosity |
| $\kappa$ | Curvature |
| $\Delta_i$ | Displacement vector |
