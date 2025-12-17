"""
holonomy_decomposition_v2.py - NAPRAWIONA Holonomy Decomposition

NAPRAWY:
1. Principal angles z SVD zamiast eigvals: θ_i = arccos(clip(s_i, -1, 1))
2. Subspace distance ||P0 - P1||_F zamiast ||I - R||_F
3. Usunięte: det, eig-angle (nie mają sensu dla nie-ortogonalnego R)
4. L2 normalizacja embeddingów przed geometrią
5. Większy cloud (20-30 mikro-interwencji)
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
    downscaled = image.resize((new_w, new_h), Image.LANCZOS)
    return downscaled.resize((w, h), Image.LANCZOS)

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def brightness(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def contrast(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


# 25 mikro-interwencji (większy cloud!)
MICRO_INTERVENTIONS = [
    # JPEG (7)
    lambda img: jpeg_compression(img, 98),
    lambda img: jpeg_compression(img, 95),
    lambda img: jpeg_compression(img, 92),
    lambda img: jpeg_compression(img, 89),
    lambda img: jpeg_compression(img, 86),
    lambda img: jpeg_compression(img, 83),
    lambda img: jpeg_compression(img, 80),
    # Blur (5)
    lambda img: gaussian_blur(img, 0.15),
    lambda img: gaussian_blur(img, 0.25),
    lambda img: gaussian_blur(img, 0.35),
    lambda img: gaussian_blur(img, 0.45),
    lambda img: gaussian_blur(img, 0.55),
    # Scale (5)
    lambda img: downscale_upscale(img, 0.98),
    lambda img: downscale_upscale(img, 0.95),
    lambda img: downscale_upscale(img, 0.92),
    lambda img: downscale_upscale(img, 0.89),
    lambda img: downscale_upscale(img, 0.86),
    # Sharpen (3)
    lambda img: sharpen(img, 1.1),
    lambda img: sharpen(img, 1.2),
    lambda img: sharpen(img, 1.3),
    # Brightness (2)
    lambda img: brightness(img, 0.98),
    lambda img: brightness(img, 1.02),
    # Contrast (2)
    lambda img: contrast(img, 0.98),
    lambda img: contrast(img, 1.02),
]


BASE_LOOPS = [
    ['jpeg_70', 'blur_0.5', 'scale_0.9'],
    ['blur_0.5', 'jpeg_80', 'scale_0.75'],
    ['jpeg_60', 'blur_0.7'],
]

TRANSFORMATIONS = {
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
}


# ============================================================================
# NAPRAWIONA HOLONOMY DECOMPOSITION
# ============================================================================

class HolonomyDecompositionV2:
    """
    NAPRAWIONA wersja Holonomy Decomposition.
    
    Cechy:
    - H_trans: translacyjna holonomia (znormalizowana)
    - subspace_dist: ||P0 - P1||_F (odległość między podprzestrzeniami)
    - theta_sum, theta_max, theta_mean: principal angles z SVD
    - cos_sum: Σcos(θ_i) = Σs_i (suma podobieństw osi)
    """
    
    def __init__(self, frame_dim: int = 8):
        self.frame_dim = frame_dim
    
    def compute_local_frame(self, encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza lokalną ramkę przez mikro-interwencje.
        
        Returns:
            center: środek chmury (768D), L2 normalized
            frame: ramka ortonormalna (768 x frame_dim)
        """
        # Generuj chmurę
        cloud_images = [image]
        for fn in MICRO_INTERVENTIONS:
            cloud_images.append(fn(image))
        
        # Encode
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # L2 normalizacja (przed geometrią!)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Środek chmury
        center = embeddings.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)  # L2 norm center too
        
        # PCA na chmurze
        centered = embeddings - center
        
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        frame_dim = min(self.frame_dim, Vt.shape[0])
        frame = Vt[:frame_dim].T  # (768, frame_dim)
        
        return center, frame
    
    def compute_principal_angles(self, F_0: np.ndarray, F_1: np.ndarray) -> Dict[str, float]:
        """
        Oblicza principal angles między podprzestrzeniami.
        
        θ_i = arccos(clip(s_i, -1, 1)) gdzie s_i to singular values R = F_0^T @ F_1
        """
        r = min(F_0.shape[1], F_1.shape[1])
        F_0 = F_0[:, :r]
        F_1 = F_1[:, :r]
        
        # R = F_0^T @ F_1
        R = F_0.T @ F_1  # (r, r)
        
        # SVD -> singular values = cos(principal angles)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        S = np.clip(S, -1.0, 1.0)
        
        # Principal angles
        theta = np.arccos(S)  # w radianach
        
        theta_sum = float(theta.sum())
        theta_max = float(theta.max())
        theta_mean = float(theta.mean())
        cos_sum = float(S.sum())  # suma cosinusów
        
        return {
            'theta_sum': theta_sum,
            'theta_max': theta_max,
            'theta_mean': theta_mean,
            'cos_sum': cos_sum,
        }
    
    def compute_subspace_distance(self, F_0: np.ndarray, F_1: np.ndarray) -> float:
        """
        Oblicza odległość między podprzestrzeniami.
        
        D_sub = ||P0 - P1||_F, gdzie P = F @ F^T (projektor)
        """
        r = min(F_0.shape[1], F_1.shape[1])
        F_0 = F_0[:, :r]
        F_1 = F_1[:, :r]
        
        # Projectors
        P0 = F_0 @ F_0.T  # (768, 768) - może być duże, ale OK
        P1 = F_1 @ F_1.T
        
        # Distance
        return float(np.linalg.norm(P0 - P1, 'fro'))
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """
        Ekstraktuje cechy dla jednego obrazu.
        """
        all_features = []
        
        for loop in BASE_LOOPS:
            # Początek
            center_0, F_0 = self.compute_local_frame(encoder, image)
            
            # Pętla
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
            
            # Koniec
            center_1, F_1 = self.compute_local_frame(encoder, current)
            
            # H_trans (znormalizowana - już L2 normalized centery)
            H_trans = float(np.linalg.norm(center_1 - center_0))
            
            # Subspace distance
            subspace_dist = self.compute_subspace_distance(F_0, F_1)
            
            # Principal angles
            angles = self.compute_principal_angles(F_0, F_1)
            
            # Cechy dla tej pętli (5 liczb)
            loop_features = [
                H_trans,
                subspace_dist,
                angles['theta_sum'],
                angles['theta_mean'],
                angles['cos_sum'],
            ]
            all_features.extend(loop_features)
        
        return np.array(all_features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        names = []
        for i in range(len(BASE_LOOPS)):
            prefix = f"loop{i}"
            names.extend([
                f"{prefix}_H_trans",
                f"{prefix}_subspace_dist",
                f"{prefix}_theta_sum",
                f"{prefix}_theta_mean",
                f"{prefix}_cos_sum",
            ])
        return names


# ============================================================================
# FAST VERSION (mniej mikro-interwencji)
# ============================================================================

class HolonomyDecompositionV2Fast:
    """
    Szybsza wersja z 12 mikro-interwencjami.
    """
    
    def __init__(self, frame_dim: int = 6):
        self.frame_dim = frame_dim
        
        self.micro = [
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
    
    def compute_local_frame(self, encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        cloud_images = [image] + [fn(image) for fn in self.micro]
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # L2 normalizacja
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        center = embeddings.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        
        centered = embeddings - center
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        frame_dim = min(self.frame_dim, Vt.shape[0])
        frame = Vt[:frame_dim].T
        
        return center, frame
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        all_features = []
        
        for loop in BASE_LOOPS:
            center_0, F_0 = self.compute_local_frame(encoder, image)
            
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
            
            center_1, F_1 = self.compute_local_frame(encoder, current)
            
            # H_trans
            H_trans = float(np.linalg.norm(center_1 - center_0))
            
            # Subspace distance
            r = min(F_0.shape[1], F_1.shape[1])
            P0 = F_0[:, :r] @ F_0[:, :r].T
            P1 = F_1[:, :r] @ F_1[:, :r].T
            subspace_dist = float(np.linalg.norm(P0 - P1, 'fro'))
            
            # Principal angles
            R = F_0[:, :r].T @ F_1[:, :r]
            _, S, _ = np.linalg.svd(R, full_matrices=False)
            S = np.clip(S, -1.0, 1.0)
            theta = np.arccos(S)
            
            theta_sum = float(theta.sum())
            theta_mean = float(theta.mean())
            cos_sum = float(S.sum())
            
            all_features.extend([H_trans, subspace_dist, theta_sum, theta_mean, cos_sum])
        
        return np.array(all_features, dtype=np.float32)
