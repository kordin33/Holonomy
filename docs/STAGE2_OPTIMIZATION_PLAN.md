# Stage 2: Optimization & Advanced Classification Plan

## 1. Analysis of Current Status (Stage 1)

**Achievements:**
- Successfully separated "Easy" Fakes (Inpainting/Insight) with >90% accuracy.
- Established a clean, reproducible baseline using CLIP ViT-B/32.
- Visualization (t-SNE) revealed specific clusters for different generation methods.

**Identified Bottlenecks:**
1.  **Metric Gap:** Classification accuracy on `Text2Img` (Stable Diffusion/DALL-E) drops to ~57% (Cross-Domain) because embeddings overlap significantly with Real images.
2.  **Inference Latency:** Current PyTorch inference is optimal for GPU (~3ms) but heavy for CPU deployment.
3.  **Algorithm Limit:** k-NN is computationally expensive ($O(N \cdot D)$) and sensitive to local noise/outliers.

---

## 2. Optimization Strategy

### A. Embedding Process Optimization (Low Latency)
To verify real-world usability (e.g., mobile/browser), we must reduce model weight and latency without losing accuracy.

**1. ONNX Runtime Export**
- **Action:** Convert CLIP Vision Encoder to `.onnx` format.
- **Benefit:** Removes Python/PyTorch overhead. Optimized graph execution.
- **Target:** <50ms latency on standard CPU.

**2. INT8 Quantization**
- **Action:** Quantize weights from FP32 to INT8.
- **Benefit:** 4x model size reduction, ~2-3x speedup on CPU.
- **Risk:** Slight accuracy drop (<1%).

### B. Multi-dimensional Classification (Accuracy Boost)
k-NN is a lazy learner. We will replace/augment it with **Eager Learning** classifiers trained on the embeddings.

**1. Support Vector Machines (SVM)**
- **Why:** Effective in finding optimal separating hyperplanes in high-dimensional space (512-dim).
- **Kernel Trick:** Can map overlapping Text2Img/Real data to higher dimensions where they might be separable.

**2. Multi-Layer Perceptron (MLP)**
- **Why:** Neural networks can learn non-linear decision boundaries (manifolds).
- **Architecture:** `Input(512) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Output(1, Sigmoid)`
- **Speed:** Inference is matrix multiplication (~0.05ms), much faster than searching neighbors in a large DB.

### C. Feature Engineering (Addressing Text2Img Failure)
The t-SNE shows Text2Img mixing with Real. This implies CLIP (semantic) features are identical.

**1. Frequency Domain Analysis (FFT)**
- **Hypothesis:** Deepfakes often leave artifacts in the high-frequency spectrum (checkerboard artifacts).
- **Action:** Compute Averaged Power Spectrum (APS) of images.
- **Fusion:** Concatenate `[CLIP_Embedding (512)] + [APS_Features (64)]` -> Train Classifier.

---

## 3. Implementation Plan

1.  **`optimize_classifiers.py`**:
    - Train SVM, Random Forest, and MLP on the *existing* embedding database.
    - Compare AUC/Accuracy against k-NN baseline.
    - **Goal:** Beat 57% on Text2Img without retraining the image encoder.

2.  **`export_onnx.py`**:
    - Export ViT-B/32 to ONNX.
    - Measure latency (PyTorch vs ONNX vs ONNX+INT8).

3.  **`feature_fusion.py`**:
    - Implement simple FFT feature extractor.
    - Test if adding simple frequency stats improves separation.

## 4. Expected Outcomes

| Component | Current (Stage 1) | Target (Stage 2) |
|-----------|-------------------|------------------|
| **Classification** | k-NN | SVM / MLP |
| **Inference** | PyTorch (GPU) | ONNX Runtime (CPU) |
| **Text2Img Acc** | ~57% | >75% |
| **Latency** | ~10ms (GPU) | ~1-5ms (CPU optimized) |
