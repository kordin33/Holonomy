"""
optimized_features.py - ZOPTYMALIZOWANE H2, H3, Baseline w COSINE-WORLD

Wszystkie cechy są teraz spójne:
- L2 normalizacja embeddingów
- Cosine distance zamiast L2
- Wspólne pętle dla synergii

H2 OPTIMIZATIONS:
- Cosine distance: 1 - dot(z_end, z0)
- Cumulative path strength
- Alpha_low vs alpha_high split
- Robust Theil-Sen fit

H3 OPTIMIZATIONS:
- Step stats (mean, std, max)
- Gram eigenvalues zamiast O(K²) pairwise
- D_cov bez niestabilnej normalizacji
- Patch disagreement features
- Te same loopy co baseline

BASELINE OPTIMIZATIONS:
- Pełne cosine (L2 norm)
- Loop consistency features
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple
from scipy.stats import theilslopes
import io


# ============================================================================
# COMMON DEGRADATIONS & LOOPS
# ============================================================================

def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB').copy()
    buffer.close()
    return img

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def downscale_upscale(image: Image.Image, scale_factor: float) -> Image.Image:
    w, h = image.size
    return image.resize((int(w*scale_factor), int(h*scale_factor)), Image.LANCZOS).resize((w, h), Image.LANCZOS)


# WSPÓLNE LOOPY (dla synergii między H2, H3, Baseline!)
SHARED_LOOPS = [
    [('jpeg_70', lambda img: jpeg_compression(img, 70)),
     ('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
     ('scale_0.9', lambda img: downscale_upscale(img, 0.9))],
    
    [('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
     ('jpeg_80', lambda img: jpeg_compression(img, 80)),
     ('scale_0.75', lambda img: downscale_upscale(img, 0.75))],
    
    [('jpeg_60', lambda img: jpeg_compression(img, 60)),
     ('blur_0.7', lambda img: gaussian_blur(img, 0.7))],
]


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize each embedding."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity) dla L2-normalized wektorów."""
    return float(1.0 - np.dot(a, b))


# ============================================================================
# H2 OPTIMIZED (COSINE SCALE LAW)
# ============================================================================

class H2_Optimized:
    """
    Scale Law w cosine-world z robust fit.
    
    Optymalizacje:
    - Cosine distance zamiast L2
    - Cumulative path strength jako "area"
    - Alpha_low / alpha_high split
    - Theil-Sen robust fit
    """
    
    def __init__(self):
        # 11 poziomów intensywności
        self.intensities = [
            (95, 0.2), (90, 0.3), (85, 0.4), (80, 0.5), (75, 0.6),
            (70, 0.7), (65, 0.8), (60, 0.9), (55, 1.0), (50, 1.1), (45, 1.2),
        ]
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # Get z_0
        z_0 = encoder.encode_batch([image], batch_size=1, show_progress=False)[0]
        z_0 = z_0 / (np.linalg.norm(z_0) + 1e-8)  # L2 norm
        
        holonomies = []
        cumulative_strengths = []
        cumulative_s = 0.0
        prev_z = z_0
        
        for jpeg_q, blur_sigma in self.intensities:
            # Apply degradation
            img1 = jpeg_compression(image, jpeg_q)
            img2 = gaussian_blur(img1, blur_sigma)
            
            emb = encoder.encode_batch([img2], batch_size=1, show_progress=False)[0]
            z = emb / (np.linalg.norm(emb) + 1e-8)
            
            # Holonomy (cosine distance from z_0)
            H = cosine_dist(z, z_0)
            holonomies.append(H)
            
            # Cumulative strength (path-based)
            cumulative_s += cosine_dist(z, prev_z)
            cumulative_strengths.append(cumulative_s)
            prev_z = z
        
        H = np.array(holonomies)
        s = np.array(cumulative_strengths)
        
        # Filter valid points
        valid = (H > 1e-8) & (s > 1e-8)
        if valid.sum() < 5:
            return np.zeros(6, dtype=np.float32)
        
        log_H = np.log(H[valid])
        log_s = np.log(s[valid])
        
        # Theil-Sen robust fit
        slope, intercept, _, _ = theilslopes(log_H, log_s)
        alpha = float(slope)
        
        # Residuals
        predicted = alpha * log_s + intercept
        residuals = log_H - predicted
        residual_std = float(residuals.std())
        
        # Alpha split (low vs high)
        n = len(log_s)
        mid = n // 2
        
        if mid >= 2:
            sl_low, _, _, _ = theilslopes(log_H[:mid], log_s[:mid])
            sl_high, _, _, _ = theilslopes(log_H[mid:], log_s[mid:])
            alpha_low = float(sl_low)
            alpha_high = float(sl_high)
            delta_alpha = alpha_high - alpha_low
        else:
            alpha_low = alpha
            alpha_high = alpha
            delta_alpha = 0.0
        
        return np.array([
            alpha,
            residual_std,
            float(H.mean()),
            alpha_low,
            alpha_high,
            delta_alpha,
        ], dtype=np.float32)


# ============================================================================
# H3 OPTIMIZED (DISPERSION + TRAJECTORY SHAPE)
# ============================================================================

class H3_Optimized:
    """
    Dispersion + Trajectory Shape w cosine-world.
    
    Optymalizacje:
    - Step stats (mean, std, max cosine steps)
    - Gram eigenvalues zamiast O(K²)
    - Stabilny D_cov
    - Te same loopy co baseline
    """
    
    def extract_loop_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Ekstraktuje cechy z jednej trajektorii."""
        # L2 normalize
        embeddings = l2_normalize(embeddings)
        
        K = len(embeddings) - 1
        z_0 = embeddings[0]
        z_K = embeddings[-1]
        
        # Step distances (cosine)
        steps = [cosine_dist(embeddings[i], embeddings[i+1]) for i in range(K)]
        steps = np.array(steps)
        
        step_mean = float(steps.mean()) if len(steps) > 0 else 0.0
        step_std = float(steps.std()) if len(steps) > 1 else 0.0
        step_max = float(steps.max()) if len(steps) > 0 else 0.0
        
        # Path length & closure (cosine)
        path_length = float(steps.sum())
        closure = cosine_dist(z_K, z_0)
        D_path = closure / (path_length + 1e-8)
        
        # Gram eigenvalues (zamiast O(K²) pairwise)
        G = embeddings @ embeddings.T  # (K+1, K+1) Gram matrix
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(np.abs(eigvals))[::-1]
        
        eigsum = eigvals.sum() + 1e-10
        eig_ratio = float(eigvals[0] / eigsum) if len(eigvals) > 0 else 0.0
        
        # Eigenvalue entropy
        p = eigvals / eigsum
        p = p[p > 1e-10]
        eig_entropy = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0
        
        # D_cov (stabilny - bez dzielenia przez ||mean||²)
        var = np.var(embeddings, axis=0, ddof=1)
        D_cov = float(var.sum())
        
        # Curvature (cosine): jak bardzo "szarpie" trajektoria
        if len(steps) > 1:
            step_diffs = np.abs(np.diff(steps))
            curvature = float(step_diffs.mean())
        else:
            curvature = 0.0
        
        return np.array([
            closure,
            D_path,
            step_mean,
            step_std,
            step_max,
            eig_ratio,
            eig_entropy,
            D_cov,
            curvature,
        ], dtype=np.float32)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy dla wszystkich shared loops."""
        all_features = []
        
        for loop in SHARED_LOOPS:
            images = [image]
            current = image
            for name, fn in loop:
                current = fn(current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            
            feat = self.extract_loop_features(embeddings)
            all_features.extend(feat)
        
        return np.array(all_features, dtype=np.float32)


# ============================================================================
# BASELINE OPTIMIZED (COSINE + LOOP CONSISTENCY)
# ============================================================================

class BaselineOptimized:
    """
    Baseline w pełnym cosine-world + loop consistency.
    
    Optymalizacje:
    - Pełne cosine (L2 norm)
    - Loop consistency features (std, iqr, max-median)
    """
    
    def extract_loop_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Cechy z jednej pętli."""
        embeddings = l2_normalize(embeddings)
        
        z_0 = embeddings[0]
        z_end = embeddings[-1]
        
        # H_raw (cosine)
        H_raw = cosine_dist(z_end, z_0)
        
        # Steps (cosine)
        steps = np.array([cosine_dist(embeddings[i], embeddings[i+1]) 
                          for i in range(len(embeddings)-1)])
        
        path_length = float(steps.sum())
        
        # Step stats
        std_step = float(steps.std()) if len(steps) > 1 else 0.0
        
        # Curvature (zmiana kierunku)
        if len(steps) > 1:
            curvature = float(np.abs(np.diff(steps)).mean())
        else:
            curvature = 0.0
        
        # Tortuosity
        tortuosity = path_length / (H_raw + 1e-8)
        
        return np.array([H_raw, path_length, std_step, curvature, tortuosity], dtype=np.float32)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy + loop consistency."""
        loop_features = []
        H_values = []
        
        for loop in SHARED_LOOPS:
            images = [image]
            current = image
            for name, fn in loop:
                current = fn(current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            
            feat = self.extract_loop_features(embeddings)
            loop_features.append(feat)
            H_values.append(feat[0])  # H_raw
        
        # Flatten loop features
        flat = np.concatenate(loop_features)
        
        # Loop consistency features
        H_arr = np.array(H_values)
        H_std = float(H_arr.std())
        H_iqr = float(np.percentile(H_arr, 75) - np.percentile(H_arr, 25))
        H_max_med = float(H_arr.max() - np.median(H_arr))
        
        # Append consistency features
        consistency = np.array([H_std, H_iqr, H_max_med], dtype=np.float32)
        
        return np.concatenate([flat, consistency])


# ============================================================================
# COMBINED EXTRACTOR (with per-block scaling)
# ============================================================================

class CombinedOptimized:
    """
    Łączy Baseline + H2 + H3 z per-block scaling.
    """
    
    def __init__(self):
        self.baseline = BaselineOptimized()
        self.h2 = H2_Optimized()
        self.h3 = H3_Optimized()
    
    def extract_features(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        """Zwraca słownik z oddzielnymi blokami cech."""
        return {
            'baseline': self.baseline.extract_features(encoder, image),
            'h2': self.h2.extract_features(encoder, image),
            'h3': self.h3.extract_features(encoder, image),
        }
    
    def extract_all_flat(self, encoder, image: Image.Image) -> np.ndarray:
        """Zwraca wszystkie cechy jako jeden wektor."""
        feats = self.extract_features(encoder, image)
        return np.concatenate([feats['baseline'], feats['h2'], feats['h3']])
