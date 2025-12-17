"""
holonomy_decomposition_v3.py - POPRAWIONA Decomposition V3

NAPRAWY:
1. subspace_dist z S² (bez projektorów 768×768!)
2. Usunięcie redundancji: tylko H_trans, theta_max, subspace_dist
3. Dodanie geometrii chmury: sv_ratio, sv_entropy, radius
4. Test wariantów: loop0 only vs all loops
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
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
    return image.resize((new_w, new_h), Image.LANCZOS).resize((w, h), Image.LANCZOS)

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(factor)

def brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)

def contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


# 12 mikro-interwencji
MICRO_INTERVENTIONS = [
    lambda img: jpeg_compression(img, 95),
    lambda img: jpeg_compression(img, 90),
    lambda img: jpeg_compression(img, 85),
    lambda img: gaussian_blur(img, 0.2),
    lambda img: gaussian_blur(img, 0.4),
    lambda img: downscale_upscale(img, 0.95),
    lambda img: downscale_upscale(img, 0.90),
    lambda img: sharpen(img, 1.15),
    lambda img: brightness(img, 0.98),
    lambda img: brightness(img, 1.02),
    lambda img: contrast(img, 0.98),
    lambda img: contrast(img, 1.02),
]

# Loops
LOOPS = [
    ['jpeg_70', 'blur_0.5', 'scale_0.9'],  # loop0 (najsilniejszy)
    ['blur_0.5', 'jpeg_80', 'scale_0.75'],
    ['jpeg_60', 'blur_0.7'],
]

TRANSFORMATIONS = {
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
}


# ============================================================================
# CLOUD GEOMETRY
# ============================================================================

def compute_cloud_geometry(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Oblicza geometrię chmury embeddingów.
    
    - sv_ratio: anizotropia (S[0] / sum(S))
    - sv_entropy: entropia spektralna
    - radius: średnia odległość od center
    """
    center = embeddings.mean(axis=0)
    centered = embeddings - center
    
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # sv_ratio (anizotropia)
    sv_ratio = float(S[0] / (S.sum() + 1e-10))
    
    # sv_entropy
    S_norm = S / (S.sum() + 1e-10)
    S_norm = S_norm[S_norm > 1e-10]
    sv_entropy = float(-np.sum(S_norm * np.log(S_norm))) if len(S_norm) > 0 else 0.0
    
    # radius (średnia odległość od center)
    distances = np.linalg.norm(centered, axis=1)
    radius = float(distances.mean())
    
    return {
        'center': center,
        'frame': Vt[:6].T,  # (768, 6)
        'sv_ratio': sv_ratio,
        'sv_entropy': sv_entropy,
        'radius': radius,
        'S': S,
    }


# ============================================================================
# DECOMPOSITION V3
# ============================================================================

class HolonomyDecompositionV3:
    """
    Holonomy Decomposition V3 - POPRAWIONA
    
    Cechy (per loop):
    - H_trans: translacyjna holonomia  
    - theta_max: max principal angle (najważniejszy do forensics)
    - subspace_dist: z S² (bez projektorów!)
    - delta_sv_ratio: zmiana anizotropii chmury
    - delta_sv_entropy: zmiana entropii chmury
    - delta_radius: zmiana promienia chmury
    """
    
    def __init__(self, frame_dim: int = 6, use_all_loops: bool = True):
        self.frame_dim = frame_dim
        self.use_all_loops = use_all_loops
    
    def compute_local_frame_with_geometry(self, encoder, image: Image.Image) -> Dict:
        """Oblicza ramkę i geometrię chmury."""
        cloud_images = [image] + [fn(image) for fn in MICRO_INTERVENTIONS]
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # L2 normalizacja
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return compute_cloud_geometry(embeddings)
    
    def compute_loop_features(self, encoder, image: Image.Image, loop: List[str]) -> np.ndarray:
        """Oblicza cechy dla jednej pętli."""
        # Początek
        geom_0 = self.compute_local_frame_with_geometry(encoder, image)
        
        # Pętla
        current = image
        for t_name in loop:
            current = TRANSFORMATIONS[t_name](current)
        
        # Koniec
        geom_1 = self.compute_local_frame_with_geometry(encoder, current)
        
        F_0 = geom_0['frame']
        F_1 = geom_1['frame']
        center_0 = geom_0['center']
        center_1 = geom_1['center']
        
        # H_trans
        H_trans = float(np.linalg.norm(center_1 - center_0))
        
        # Principal angles z SVD
        r = min(F_0.shape[1], F_1.shape[1])
        R = F_0[:, :r].T @ F_1[:, :r]
        _, S, _ = np.linalg.svd(R, full_matrices=False)
        S = np.clip(S, -1.0, 1.0)
        theta = np.arccos(S)
        
        theta_max = float(theta.max())
        
        # subspace_dist z S² (bez projektorów!)
        # ||P0 - P1||_F² = 2r - 2*sum(cos²(θ)) = 2r - 2*sum(S²)
        subspace_dist = float(np.sqrt(max(0, 2 * r - 2 * np.sum(S ** 2))))
        
        # Geometria chmury - różnice
        delta_sv_ratio = geom_1['sv_ratio'] - geom_0['sv_ratio']
        delta_sv_entropy = geom_1['sv_entropy'] - geom_0['sv_entropy']
        delta_radius = geom_1['radius'] - geom_0['radius']
        
        return np.array([
            H_trans,
            theta_max,
            subspace_dist,
            delta_sv_ratio,
            delta_sv_entropy,
            delta_radius,
        ], dtype=np.float32)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy dla jednego obrazu."""
        if self.use_all_loops:
            loops_to_use = LOOPS
        else:
            loops_to_use = [LOOPS[0]]  # tylko loop0
        
        all_features = []
        for loop in loops_to_use:
            feat = self.compute_loop_features(encoder, image, loop)
            all_features.extend(feat)
        
        return np.array(all_features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        base = ['H_trans', 'theta_max', 'subspace_dist', 
                'delta_sv_ratio', 'delta_sv_entropy', 'delta_radius']
        if self.use_all_loops:
            return [f"loop{i}_{n}" for i in range(len(LOOPS)) for n in base]
        else:
            return [f"loop0_{n}" for n in base]


# ============================================================================
# ABLATION VARIANTS
# ============================================================================

class DecompV3_HtransOnly:
    """Tylko H_trans per loop."""
    def __init__(self, use_all_loops: bool = True):
        self.decomp = HolonomyDecompositionV3(use_all_loops=use_all_loops)
        self.use_all_loops = use_all_loops
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        full = self.decomp.extract_features(encoder, image)
        n_loops = len(LOOPS) if self.use_all_loops else 1
        # H_trans jest index 0 w każdej pętli (6 cech per loop)
        indices = [i * 6 for i in range(n_loops)]
        return full[indices]


class DecompV3_Minimal:
    """Tylko H_trans + theta_max per loop (2 cechy per loop)."""
    def __init__(self, use_all_loops: bool = True):
        self.decomp = HolonomyDecompositionV3(use_all_loops=use_all_loops)
        self.use_all_loops = use_all_loops
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        full = self.decomp.extract_features(encoder, image)
        n_loops = len(LOOPS) if self.use_all_loops else 1
        # H_trans (0), theta_max (1) per loop
        indices = []
        for i in range(n_loops):
            indices.extend([i * 6, i * 6 + 1])
        return full[indices]


class DecompV3_Loop0Only:
    """Tylko loop0 (najsilniejszy)."""
    def __init__(self):
        self.decomp = HolonomyDecompositionV3(use_all_loops=False)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        return self.decomp.extract_features(encoder, image)
