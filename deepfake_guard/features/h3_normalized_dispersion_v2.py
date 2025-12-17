"""
h3_normalized_dispersion_v2.py - NAPRAWIONY H3 V2

NAPRAWY:
1. L2 normalizacja embeddingów przed metrykami
2. D_path-cos w metryce cosinusowej (1-cos zamiast L2)
3. trace_cov bez macierzy (suma wariancji, pomijając z0)
4. Grid patchy (4 rogi + środek) zamiast losowych
5. Patch resize do 224×224 (standard encodera)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict
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


# Sekwencja transformacji
TRANSFORM_SEQUENCE = [
    lambda img: jpeg_compression(img, 90),
    lambda img: gaussian_blur(img, 0.3),
    lambda img: jpeg_compression(img, 75),
    lambda img: gaussian_blur(img, 0.5),
    lambda img: downscale_upscale(img, 0.9),
    lambda img: jpeg_compression(img, 60),
    lambda img: gaussian_blur(img, 0.7),
]


# ============================================================================
# HELPERS
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize each embedding."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


# ============================================================================
# H3 NORMALIZED DISPERSION V2
# ============================================================================

class H3_NormalizedDispersionV2:
    """
    NAPRAWIONY H3: Normalized Dispersion Features V2
    
    Features:
    - D_cos: mean pairwise cosine dispersion
    - D_path_cos: cosine path holonomy (1-cos(z0,zK)) / Σ(1-cos(zi,zi+1))
    - D_cov: trace covariance normalized (suma wariancji, pomijając z0)
    
    Patch features:
    - Grid: 4 rogi + środek (5 patchy)
    - Patchy resize do 224×224
    """
    
    def __init__(self):
        pass
    
    def compute_dispersion_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Oblicza 3 znormalizowane metryki rozproszenia.
        
        Args:
            embeddings: (K+1, d) - z_0, z_1, ..., z_K (już L2 normalized!)
        """
        K = len(embeddings) - 1
        z_0 = embeddings[0]
        z_K = embeddings[-1]
        
        # (1) D_cos: Mean pairwise cosine dispersion
        # D_cos = (2 / K(K-1)) * Σ_{i<j} (1 - cos(z_i, z_j))
        cos_dispersions = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = cosine_similarity(embeddings[i], embeddings[j])
                cos_dispersions.append(1.0 - cos_sim)
        
        D_cos = float(np.mean(cos_dispersions)) if cos_dispersions else 0.0
        
        # (2) D_path_cos: Cosine path holonomy
        # D_path_cos = (1 - cos(z0, zK)) / Σ(1 - cos(zi, zi+1)) + ε
        closure_cos = 1.0 - cosine_similarity(z_0, z_K)
        
        path_cos = 0.0
        for i in range(len(embeddings) - 1):
            path_cos += 1.0 - cosine_similarity(embeddings[i], embeddings[i + 1])
        
        D_path_cos = float(closure_cos / (path_cos + 1e-8))
        
        # (3) D_cov: Trace covariance normalized (bez z0!)
        # trace_cov = Σ var(dim) / ||mean||²
        E = embeddings[1:]  # tylko degradacje, bez z0
        if len(E) > 1:
            var = np.var(E, axis=0, ddof=1)  # (d,)
            trace_cov = float(var.sum())
            mean_norm_sq = float(np.linalg.norm(E.mean(axis=0)) ** 2)
            D_cov = trace_cov / (mean_norm_sq + 1e-8)
        else:
            D_cov = 0.0
        
        return {
            'D_cos': D_cos,
            'D_path_cos': D_path_cos,
            'D_cov': D_cov,
        }
    
    def compute_image_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza cechy dla całego obrazu.
        """
        images = [image]
        current = image
        for fn in TRANSFORM_SEQUENCE:
            current = fn(current)
            images.append(current)
        
        embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # L2 normalizacja!
        embeddings = l2_normalize(embeddings)
        
        return self.compute_dispersion_metrics(embeddings)
    
    def get_grid_patches(self, image: Image.Image) -> List[Image.Image]:
        """
        Grid patchy: 4 rogi + środek.
        Resize do 224×224 (standard encodera).
        """
        w, h = image.size
        patch_size = min(w, h) // 2
        
        # Pozycje: (x, y)
        positions = [
            (0, 0),                          # top-left
            (w - patch_size, 0),             # top-right
            (0, h - patch_size),             # bottom-left
            (w - patch_size, h - patch_size),# bottom-right
            ((w - patch_size) // 2, (h - patch_size) // 2),  # center
        ]
        
        patches = []
        for x, y in positions:
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patch = patch.resize((224, 224), Image.LANCZOS)  # standard encoder size!
            patches.append(patch)
        
        return patches
    
    def compute_patch_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza cechy per-patch i agreguje.
        """
        patches = self.get_grid_patches(image)
        
        patch_D_cos = []
        patch_D_path_cos = []
        patch_D_cov = []
        
        # Krótsza sekwencja dla patchy (szybciej)
        short_sequence = TRANSFORM_SEQUENCE[:4]
        
        for patch in patches:
            images = [patch]
            current = patch
            for fn in short_sequence:
                current = fn(current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            embeddings = l2_normalize(embeddings)
            
            metrics = self.compute_dispersion_metrics(embeddings)
            patch_D_cos.append(metrics['D_cos'])
            patch_D_path_cos.append(metrics['D_path_cos'])
            patch_D_cov.append(metrics['D_cov'])
        
        # Agregacja: median + p80
        return {
            'patch_D_cos_median': float(np.median(patch_D_cos)),
            'patch_D_cos_p80': float(np.percentile(patch_D_cos, 80)),
            'patch_D_path_median': float(np.median(patch_D_path_cos)),
            'patch_D_path_p80': float(np.percentile(patch_D_path_cos, 80)),
            'patch_D_cov_median': float(np.median(patch_D_cov)),
            'patch_D_cov_p80': float(np.percentile(patch_D_cov, 80)),
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """
        Ekstraktuje wszystkie cechy.
        """
        # Image-level
        img_metrics = self.compute_image_features(encoder, image)
        
        # Patch-level
        patch_metrics = self.compute_patch_features(encoder, image)
        
        features = [
            # Image-level (3)
            img_metrics['D_cos'],
            img_metrics['D_path_cos'],
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
            'D_cos', 'D_path_cos', 'D_cov',
            'patch_D_cos_median', 'patch_D_cos_p80',
            'patch_D_path_median', 'patch_D_path_p80',
            'patch_D_cov_median', 'patch_D_cov_p80',
        ]


# ============================================================================
# FAST VERSION (image-only, no patches)
# ============================================================================

class H3_NormalizedDispersionV2Fast:
    """
    Szybsza wersja bez patchy.
    """
    
    def __init__(self):
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
        
        # L2 normalizacja
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        K = len(embeddings) - 1
        z_0 = embeddings[0]
        z_K = embeddings[-1]
        
        # D_cos
        cos_dispersions = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dot = np.dot(embeddings[i], embeddings[j])
                cos_dispersions.append(1.0 - dot)  # already L2 normalized
        D_cos = float(np.mean(cos_dispersions))
        
        # D_path_cos
        closure_cos = 1.0 - float(np.dot(z_0, z_K))
        path_cos = sum(1.0 - float(np.dot(embeddings[i], embeddings[i+1])) 
                      for i in range(len(embeddings) - 1))
        D_path_cos = closure_cos / (path_cos + 1e-8)
        
        # D_cov (bez z0)
        E = embeddings[1:]
        var = np.var(E, axis=0, ddof=1)
        trace_cov = float(var.sum())
        mean_norm_sq = float(np.linalg.norm(E.mean(axis=0)) ** 2)
        D_cov = trace_cov / (mean_norm_sq + 1e-8)
        
        return np.array([D_cos, D_path_cos, D_cov], dtype=np.float32)
