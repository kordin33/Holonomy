"""
h3_normalized_dispersion.py - NAPRAWIONY H3: Normalized Dispersion Features

PROBLEM STAREGO H3:
- Rozrzut embeddingów rośnie głównie od siły degradacji, nie od real/fake
- Brak normalizacji = dominacja szumu

NOWY H3 (3 znormalizowane metryki):

1) D_cos = Mean pairwise cosine dispersion
   D_cos = (2 / K(K-1)) * Σ_{i<j} (1 - cos(z_i, z_j))

2) D_path = Normalized closure drift (odporny na siłę perturbacji)
   D_path = ||z_K - z_0|| / (Σ ||z_{i+1} - z_i|| + ε)
   To jest nasze holonomy / path_length!

3) D_Σ = Trace covariance normalized
   D_Σ = tr(Cov(z_1..z_K)) / (||E[z_k]||² + ε)

Plus: Patch-consistency (median + p80 per patch)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple
import io


# ============================================================================
# DEGRADATIONS
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
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    downscaled = image.resize((new_w, new_h), Image.LANCZOS)
    return downscaled.resize((w, h), Image.LANCZOS)


# Sekwencja transformacji (identyczna dla obu klas)
TRANSFORM_SEQUENCE = [
    ('jpeg_90', lambda img: jpeg_compression(img, 90)),
    ('blur_0.3', lambda img: gaussian_blur(img, 0.3)),
    ('jpeg_75', lambda img: jpeg_compression(img, 75)),
    ('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
    ('scale_0.9', lambda img: downscale_upscale(img, 0.9)),
    ('jpeg_60', lambda img: jpeg_compression(img, 60)),
    ('blur_0.7', lambda img: gaussian_blur(img, 0.7)),
]


# ============================================================================
# NORMALIZED DISPERSION FEATURES
# ============================================================================

class H3_NormalizedDispersion:
    """
    Znormalizowane metryki rozproszenia embeddingów pod degradacją.
    
    Features:
    - D_cos: mean pairwise cosine dispersion
    - D_path: normalized closure drift (holonomy / path_length)
    - D_cov: trace covariance normalized
    - Patch-level stats: median + p80
    """
    
    def __init__(self, n_patches: int = 4):
        self.n_patches = n_patches
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def compute_dispersion_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Oblicza 3 znormalizowane metryki rozproszenia.
        
        Args:
            embeddings: (K+1, d) - z_0, z_1, ..., z_K
        
        Returns:
            D_cos, D_path, D_cov
        """
        K = len(embeddings) - 1  # z_0 to original
        z_0 = embeddings[0]
        z_K = embeddings[-1]
        
        # (1) D_cos: Mean pairwise cosine dispersion
        # D_cos = (2 / K(K-1)) * Σ_{i<j} (1 - cos(z_i, z_j))
        cos_dispersions = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = self.cosine_similarity(embeddings[i], embeddings[j])
                cos_dispersions.append(1.0 - cos_sim)
        
        if len(cos_dispersions) > 0:
            D_cos = float(np.mean(cos_dispersions))
        else:
            D_cos = 0.0
        
        # (2) D_path: Normalized closure drift
        # D_path = ||z_K - z_0|| / (Σ ||z_{i+1} - z_i|| + ε)
        closure = np.linalg.norm(z_K - z_0)
        path_length = 0.0
        for i in range(len(embeddings) - 1):
            path_length += np.linalg.norm(embeddings[i + 1] - embeddings[i])
        
        D_path = float(closure / (path_length + 1e-8))
        
        # (3) D_cov: Trace covariance normalized
        # D_Σ = tr(Cov(z_1..z_K)) / (||E[z_k]||² + ε)
        if K > 1:
            # Covariance of embeddings (excluding z_0 or including, both work)
            cov_matrix = np.cov(embeddings.T)  # (d, d)
            trace_cov = float(np.trace(cov_matrix))
            
            mean_emb = embeddings.mean(axis=0)
            mean_norm_sq = float(np.linalg.norm(mean_emb) ** 2)
            
            D_cov = trace_cov / (mean_norm_sq + 1e-8)
        else:
            D_cov = 0.0
        
        return {
            'D_cos': D_cos,
            'D_path': D_path,
            'D_cov': D_cov,
            'closure': float(closure),
            'path_length': float(path_length),
        }
    
    def compute_image_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza cechy dla całego obrazu.
        """
        # Zbierz embeddingi po sekwencji transformacji
        images = [image]
        current = image
        for name, fn in TRANSFORM_SEQUENCE:
            current = fn(current)
            images.append(current)
        
        embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        return self.compute_dispersion_metrics(embeddings)
    
    def compute_patch_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza cechy per-patch i agreguje.
        
        Daje odporność na "wklejony fragment" - jeśli część obrazu jest inna,
        median i percentyle to wykryją.
        """
        w, h = image.size
        patch_size = min(w, h) // 2
        
        # Losowe patche (deterministyczne pozycje)
        rng = np.random.default_rng(42)
        
        patch_D_cos = []
        patch_D_path = []
        patch_D_cov = []
        
        for _ in range(self.n_patches):
            # Losowa pozycja
            x = rng.integers(0, max(1, w - patch_size))
            y = rng.integers(0, max(1, h - patch_size))
            
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patch = patch.resize((112, 112), Image.LANCZOS)
            
            # Zbierz embeddingi
            images = [patch]
            current = patch
            for name, fn in TRANSFORM_SEQUENCE[:4]:  # Krótsza sekwencja dla patchy
                current = fn(current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            
            metrics = self.compute_dispersion_metrics(embeddings)
            patch_D_cos.append(metrics['D_cos'])
            patch_D_path.append(metrics['D_path'])
            patch_D_cov.append(metrics['D_cov'])
        
        # Agregacja: median + p80
        return {
            'patch_D_cos_median': float(np.median(patch_D_cos)),
            'patch_D_cos_p80': float(np.percentile(patch_D_cos, 80)),
            'patch_D_path_median': float(np.median(patch_D_path)),
            'patch_D_path_p80': float(np.percentile(patch_D_path, 80)),
            'patch_D_cov_median': float(np.median(patch_D_cov)),
            'patch_D_cov_p80': float(np.percentile(patch_D_cov, 80)),
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """
        Ekstraktuje wszystkie cechy.
        """
        # Cechy z całego obrazu
        img_metrics = self.compute_image_features(encoder, image)
        
        # Cechy z patchy
        patch_metrics = self.compute_patch_features(encoder, image)
        
        features = [
            # Image-level (3)
            img_metrics['D_cos'],
            img_metrics['D_path'],
            img_metrics['D_cov'],
            
            # Patch-level (6)
            patch_metrics['patch_D_cos_median'],
            patch_metrics['patch_D_cos_p80'],
            patch_metrics['patch_D_path_median'],
            patch_metrics['patch_D_path_p80'],
            patch_metrics['patch_D_cov_median'],
            patch_metrics['patch_D_cov_p80'],
        ]
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        return [
            'D_cos', 'D_path', 'D_cov',
            'patch_D_cos_median', 'patch_D_cos_p80',
            'patch_D_path_median', 'patch_D_path_p80',
            'patch_D_cov_median', 'patch_D_cov_p80',
        ]


# ============================================================================
# SIMPLIFIED VERSION (faster)
# ============================================================================

class H3_NormalizedDispersionFast:
    """
    Szybsza wersja bez patchy, tylko image-level.
    """
    
    def __init__(self):
        # Krótsza sekwencja
        self.transforms = [
            lambda img: jpeg_compression(img, 85),
            lambda img: gaussian_blur(img, 0.4),
            lambda img: jpeg_compression(img, 70),
            lambda img: downscale_upscale(img, 0.9),
        ]
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        images = [image]
        current = image
        for fn in self.transforms:
            current = fn(current)
            images.append(current)
        
        embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        K = len(embeddings) - 1
        z_0 = embeddings[0]
        z_K = embeddings[-1]
        
        # D_cos
        cos_dispersions = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dot = np.dot(embeddings[i], embeddings[j])
                norms = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                cos_dispersions.append(1.0 - dot / (norms + 1e-10))
        D_cos = float(np.mean(cos_dispersions))
        
        # D_path
        closure = np.linalg.norm(z_K - z_0)
        path_length = sum(np.linalg.norm(embeddings[i + 1] - embeddings[i]) 
                         for i in range(len(embeddings) - 1))
        D_path = float(closure / (path_length + 1e-8))
        
        # D_cov
        cov = np.cov(embeddings.T)
        trace_cov = float(np.trace(cov))
        mean_norm_sq = float(np.linalg.norm(embeddings.mean(axis=0)) ** 2)
        D_cov = trace_cov / (mean_norm_sq + 1e-8)
        
        return np.array([D_cos, D_path, D_cov], dtype=np.float32)
