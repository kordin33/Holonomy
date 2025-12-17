"""
holonomy_decomposition_v11.py - SPECTRAL HOLONOMY (Eigen-Decay Analysis)

Cel: 0.90 AUC
Podejście: Analiza widma macierzy Grama degradacji.
Hipoteza: Realne zdjęcia reagują różnicowo na różne typy degradacji (wysoki rząd macierzy odpowiedzi).
Deepfake'i reagują schematycznie (niski rząd, szybki zanik wartości własnych).
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
import io


# ============================================================================
# UTILS
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

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(factor)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)

def chordal_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))


# ============================================================================
# SPECTRAL DECOMPOSITION
# ============================================================================

BASELINE_LOOPS = [
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],
    ['scale_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.5'],
    ['jpeg_80', 'blur_0.3', 'jpeg_60', 'blur_0.5'],
    ['jpeg_50', 'scale_0.75', 'blur_1.0', 'jpeg_80'],
    ['jpeg_90', 'blur_0.3', 'scale_0.9', 'jpeg_80'],
    ['blur_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.7'],
    ['jpeg_90', 'scale_0.75', 'jpeg_50', 'scale_0.75'],
    ['sharpen_1.5', 'jpeg_80', 'scale_0.75'],
]

TRANSFORMS = {
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'jpeg_50': lambda img: jpeg_compression(img, 50),
    'blur_0.3': lambda img: gaussian_blur(img, 0.3),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'blur_1.0': lambda img: gaussian_blur(img, 1.0),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.5': lambda img: downscale_upscale(img, 0.5),
    'sharpen_1.5': lambda img: sharpen(img, 1.5),
    'identity': lambda img: img,
}

class HolonomyDecompositionV11:
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Obliczamy odpowiedzi na końcach pętli (wektory z_end)
        loop_endpoints = []
        baseline_feats = []
        
        z0 = l2_normalize(encoder.encode_batch([image], batch_size=1, show_progress=False)[0])
        
        for loop in BASELINE_LOOPS:
            imgs = [image]
            curr = image
            for name in loop:
                curr = TRANSFORMS[name](curr)
                imgs.append(curr)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            emb = l2_normalize(np.asarray(emb))
            
            # Endpoint vector (względem z0 - różnica)
            # Używamy różnicy wektorów, bo interesuje nas kierunek przesunięcia
            diff = emb[-1] - emb[0]
            loop_endpoints.append(diff)
            
            # Standard baseline features (for safety)
            H = chordal_dist(emb[0], emb[-1])
            baseline_feats.append(H)
            
        # Macierz odpowiedzi M (9 x 768)
        M = np.array(loop_endpoints, dtype=np.float32)
        
        # 2. Spectral Analysis
        # Gram Matrix G = M @ M.T (9 x 9) - korelacje między pętlami
        G = np.dot(M, M.T).astype(np.float64)
        
        # Wartości własne (Eigenvalues)
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1] # Malejąco
        
        # Normalizacja widma (żeby suma była 1 - probability distribution)
        eig_sum = np.sum(eigvals) + 1e-10
        eig_prob = eigvals / eig_sum
        
        # Spectral Entropy
        entropy = -np.sum(eig_prob * np.log(eig_prob + 1e-10))
        
        # Effective Rank (liczba istotnych wartości własnych)
        # exp(entropy) to jedna miara, ale prościej: sum(p^2) (Inverse Participation Ratio)
        ipr = np.sum(eig_prob ** 2)
        
        # Ratios
        ratio_1_2 = eigvals[1] / (eigvals[0] + 1e-10)
        ratio_last_first = eigvals[-1] / (eigvals[0] + 1e-10)
        
        # Features: Eigenvalues + Entropy + Ratios + Baseline Scalars
        feats = np.concatenate([
            eigvals,            # 9 cech (widmo)
            [entropy, ipr, ratio_1_2, ratio_last_first], # 4 cechy spektralne
            baseline_feats      # 9 cech (H_chordal)
        ])
        
        return feats.astype(np.float32)
