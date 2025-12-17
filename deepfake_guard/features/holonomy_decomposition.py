"""
holonomy_decomposition.py - Holonomy Decomposition: Translacyjna vs Rotacyjna

HIPOTEZA:
- H_trans = ||z_end - z_0|| mierzy tylko "jak daleko"
- H_rot mierzy jak "ramka lokalna" obraca się po pętli

Fake i real mogą różnić się w rotacyjnej holonomii bardziej niż w samej translacji,
bo artefakty AI generatorów często są "orientacyjne" (rotacja cech w podprzestrzeni).

IMPLEMENTACJA:
1. Dla z_0: generuj cloud przez mikro-interwencje δ_j (mini JPEG, mini blur, mini resize)
2. PCA na cloud → ramka F_0 ∈ R^{d×r} (ortonormalna)
3. To samo na x_end → ramka F_1
4. Rotacyjna holonomia: R = F_0^T @ F_1 ∈ R^{r×r}
5. Cechy: ||I - R||_F, widmo R, trace, entropia SVD
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
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


# Mikro-interwencje (bardzo małe perturbacje)
MICRO_INTERVENTIONS = [
    ('jpeg_98', lambda img: jpeg_compression(img, 98)),
    ('jpeg_95', lambda img: jpeg_compression(img, 95)),
    ('jpeg_92', lambda img: jpeg_compression(img, 92)),
    ('blur_0.1', lambda img: gaussian_blur(img, 0.1)),
    ('blur_0.2', lambda img: gaussian_blur(img, 0.2)),
    ('blur_0.3', lambda img: gaussian_blur(img, 0.3)),
    ('scale_0.98', lambda img: downscale_upscale(img, 0.98)),
    ('scale_0.95', lambda img: downscale_upscale(img, 0.95)),
    ('scale_0.92', lambda img: downscale_upscale(img, 0.92)),
]

# Pętla do testowania holonomii
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
# HOLONOMY DECOMPOSITION
# ============================================================================

class HolonomyDecomposition:
    """
    Rozkład holonomii na część translacyjną i rotacyjną.
    
    - Translacyjna: ||z_end - z_0|| (jak daleko)
    - Rotacyjna: jak ramka lokalna obraca się (R = F_0^T @ F_1)
    """
    
    def __init__(self, frame_dim: int = 8):
        """
        Args:
            frame_dim: wymiar ramki lokalnej (ile wektorów własnych)
        """
        self.frame_dim = frame_dim
    
    def compute_local_frame(self, encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza lokalną ramkę przez mikro-interwencje.
        
        Returns:
            center: środek chmury (768D)
            frame: ramka ortonormalna (768 x frame_dim)
        """
        # Generuj chmurę przez mikro-interwencje
        cloud_images = [image]  # oryginał
        for name, fn in MICRO_INTERVENTIONS:
            cloud_images.append(fn(image))
        
        # Encode (konwersja do float32 dla linalg!)
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # Środek chmury
        center = embeddings.mean(axis=0)
        
        # PCA na chmurze (względem środka)
        centered = embeddings - center
        
        # Użyj SVD zamiast PCA (szybsze dla małych cloud)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Ramka = pierwsze frame_dim wektorów własnych
        frame_dim = min(self.frame_dim, Vt.shape[0])
        frame = Vt[:frame_dim].T  # (768, frame_dim)
        
        return center, frame
    
    def compute_rotation_holonomy(self, F_0: np.ndarray, F_1: np.ndarray) -> Dict[str, float]:
        """
        Oblicza rotacyjną holonomię między ramkami.
        
        R = F_0^T @ F_1
        
        Cechy:
        - ||I - R||_F: jak mocno ramka się przekręciła
        - trace(R): suma cosinusów kątów
        - det(R): orientacja (powinien być ~1 dla czystej rotacji)
        - spectral features z SVD(R)
        """
        r = min(F_0.shape[1], F_1.shape[1])
        F_0 = F_0[:, :r]
        F_1 = F_1[:, :r]
        
        # Macierz rotacji (powinna być bliska ortogonalnej)
        R = F_0.T @ F_1  # (r, r)
        
        # ||I - R||_F - jak daleko od identyczności
        I = np.eye(r)
        frame_deviation = float(np.linalg.norm(I - R, 'fro'))
        
        # Trace(R) - suma cosinusów kątów (dla ortogonalnej R, trace = Σcos(θ_i))
        trace_R = float(np.trace(R))
        
        # Determinant - powinien być ~1 dla rotacji, ~-1 dla refleksji
        det_R = float(np.linalg.det(R))
        
        # SVD dla spektrum
        U, S, Vt = np.linalg.svd(R)
        
        # Wartości singularne (powinny być ~1 dla ortogonalnej)
        sv_mean = float(S.mean())
        sv_std = float(S.std())
        sv_min = float(S.min())
        
        # Entropia SVD
        S_norm = S / (S.sum() + 1e-10)
        S_norm = S_norm[S_norm > 1e-10]
        sv_entropy = float(-np.sum(S_norm * np.log(S_norm))) if len(S_norm) > 0 else 0.0
        
        # Kąty rotacji (z eigenvalues zespolonych R)
        try:
            eigvals = np.linalg.eigvals(R)
            angles = np.abs(np.angle(eigvals))  # kąty w radianach
            total_rotation = float(angles.sum())
            max_rotation = float(angles.max())
        except:
            total_rotation = 0.0
            max_rotation = 0.0
        
        return {
            'frame_deviation': frame_deviation,  # ||I - R||_F
            'trace': trace_R,
            'det': det_R,
            'sv_mean': sv_mean,
            'sv_std': sv_std,
            'sv_min': sv_min,
            'sv_entropy': sv_entropy,
            'total_rotation': total_rotation,
            'max_rotation': max_rotation,
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """
        Ekstraktuje cechy dla jednego obrazu.
        """
        all_features = []
        
        for loop in BASE_LOOPS:
            # Początek: oblicz ramkę F_0
            center_0, F_0 = self.compute_local_frame(encoder, image)
            
            # Przejdź pętlę
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
            
            # Koniec: oblicz ramkę F_1
            center_1, F_1 = self.compute_local_frame(encoder, current)
            
            # Translacyjna holonomia
            H_trans = float(np.linalg.norm(center_1 - center_0))
            
            # Rotacyjna holonomia
            rot_stats = self.compute_rotation_holonomy(F_0, F_1)
            
            # Zbierz cechy dla tej pętli
            loop_features = [
                H_trans,
                rot_stats['frame_deviation'],
                rot_stats['trace'],
                rot_stats['det'],
                rot_stats['sv_entropy'],
                rot_stats['total_rotation'],
                rot_stats['max_rotation'],
            ]
            all_features.extend(loop_features)
        
        return np.array(all_features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Zwraca nazwy cech."""
        names = []
        for i, loop in enumerate(BASE_LOOPS):
            prefix = f"loop{i}"
            names.extend([
                f"{prefix}_H_trans",
                f"{prefix}_frame_dev",
                f"{prefix}_trace",
                f"{prefix}_det",
                f"{prefix}_sv_entropy",
                f"{prefix}_total_rot",
                f"{prefix}_max_rot",
            ])
        return names


# ============================================================================
# SIMPLIFIED VERSION (szybsza, mniej mikro-interwencji)
# ============================================================================

class HolonomyDecompositionFast:
    """
    Szybsza wersja z mniejszą liczbą mikro-interwencji.
    """
    
    def __init__(self, frame_dim: int = 6):
        self.frame_dim = frame_dim
        
        # Tylko 5 mikro-interwencji
        self.micro = [
            lambda img: jpeg_compression(img, 95),
            lambda img: jpeg_compression(img, 90),
            lambda img: gaussian_blur(img, 0.2),
            lambda img: downscale_upscale(img, 0.95),
            lambda img: downscale_upscale(img, 0.90),
        ]
    
    def compute_local_frame(self, encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        cloud_images = [image] + [fn(image) for fn in self.micro]
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)  # Fix float16!
        
        center = embeddings.mean(axis=0)
        centered = embeddings - center
        
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        frame_dim = min(self.frame_dim, Vt.shape[0])
        frame = Vt[:frame_dim].T
        
        return center, frame
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
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
            
            # Translacyjna
            H_trans = float(np.linalg.norm(center_1 - center_0))
            
            # Rotacyjna
            r = min(F_0.shape[1], F_1.shape[1])
            R = F_0[:, :r].T @ F_1[:, :r]
            
            I = np.eye(r)
            frame_dev = float(np.linalg.norm(I - R, 'fro'))
            trace_R = float(np.trace(R))
            
            try:
                eigvals = np.linalg.eigvals(R)
                angles = np.abs(np.angle(eigvals))
                total_rot = float(angles.sum())
            except:
                total_rot = 0.0
            
            all_features.extend([H_trans, frame_dev, trace_R, total_rot])
        
        return np.array(all_features, dtype=np.float32)
