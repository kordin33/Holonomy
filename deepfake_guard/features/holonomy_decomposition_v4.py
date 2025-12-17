"""
holonomy_decomposition_v4.py - PRAWDZIWA Holonomia: Path-Ordered Frame Transport

V3 problem: Porównywało tylko F_0 i F_1 (dwie lokalne aproksymacje)
V4 rozwiązanie: Transport ramki po KOLEJNYCH krokach pętli

ALGORYTM:
1. Dla każdego punktu x_i na pętli policz lokalną ramkę F_i
2. Policz operator transportu Q_i między F_i i F_{i+1} przez Procrustes:
   SVD(F_i^T @ F_{i+1}) = U Σ V^T
   Q_i = U @ V^T
3. Holonomia pętli jako iloczyn uporządkowany:
   Q_loop = Q_0 @ Q_1 @ ... @ Q_{m-1}
4. Cechy z Q_loop: ||I - Q||_F, kąty własne, trace, entropia widma

DODATKOWO:
- Log-map na sferze dla spójności z cosine-worldem
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
    return image.resize((int(w*scale_factor), int(h*scale_factor)), Image.LANCZOS).resize((w, h), Image.LANCZOS)

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(factor)

def brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


# Mikro-interwencje dla ramki lokalnej
MICRO_INTERVENTIONS = [
    lambda img: jpeg_compression(img, 95),
    lambda img: jpeg_compression(img, 90),
    lambda img: gaussian_blur(img, 0.2),
    lambda img: gaussian_blur(img, 0.4),
    lambda img: downscale_upscale(img, 0.95),
    lambda img: downscale_upscale(img, 0.90),
    lambda img: sharpen(img, 1.15),
    lambda img: brightness(img, 0.98),
    lambda img: brightness(img, 1.02),
]

# Pętla degradacji (kolejne kroki!)
LOOP_STEPS = [
    ('jpeg_80', lambda img: jpeg_compression(img, 80)),
    ('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
    ('scale_0.9', lambda img: downscale_upscale(img, 0.9)),
    ('jpeg_70', lambda img: jpeg_compression(img, 70)),
]


# ============================================================================
# SPHERICAL GEOMETRY
# ============================================================================

def spherical_mean(embeddings: np.ndarray) -> np.ndarray:
    """Oblicza średnią sferyczną (Fréchet mean na S^{d-1})."""
    # Prostsza wersja: znormalizowana suma
    mean = embeddings.sum(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-10)


def log_map(point: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Log-map na sferze: mapuje punkt na tangent space przy base.
    
    v = point - (point · base) * base
    Opcjonalnie skalowane przez arccos(point · base)
    """
    dot = np.dot(point, base)
    dot = np.clip(dot, -1.0, 1.0)
    v = point - dot * base
    
    # Skalowanie przez kąt (opcjonalne, ale spójne z geodezyjną odległością)
    norm_v = np.linalg.norm(v)
    if norm_v > 1e-8:
        theta = np.arccos(dot)
        v = v * (theta / norm_v)
    
    return v


def compute_tangent_frame(embeddings: np.ndarray, frame_dim: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oblicza ramkę w tangent space używając log-map na sferze.
    
    Returns:
        base: punkt bazowy (spherical mean)
        frame: ramka ortonormalna w tangent space (d x frame_dim)
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Spherical mean
    base = spherical_mean(embeddings)
    
    # Log-map każdego punktu do tangent space przy base
    tangents = np.array([log_map(e, base) for e in embeddings])
    
    # SVD na tangent vectors
    U, S, Vt = np.linalg.svd(tangents, full_matrices=False)
    
    r = min(frame_dim, Vt.shape[0])
    frame = Vt[:r].T  # (d, r)
    
    return base, frame


# ============================================================================
# PROCRUSTES TRANSPORT
# ============================================================================

def procrustes_rotation(F_i: np.ndarray, F_j: np.ndarray) -> np.ndarray:
    """
    Oblicza optymalną rotację Q ∈ O(r) która mapuje F_j na F_i.
    
    Q = argmin_Q ||F_i - F_j @ Q||_F
    
    Rozwiązanie: SVD(F_i^T @ F_j) = U Σ V^T, Q = V @ U^T
    """
    r = min(F_i.shape[1], F_j.shape[1])
    F_i = F_i[:, :r]
    F_j = F_j[:, :r]
    
    # SVD
    M = F_i.T @ F_j  # (r, r)
    U, S, Vt = np.linalg.svd(M)
    
    # Optymalna rotacja
    Q = Vt.T @ U.T  # V @ U^T
    
    return Q


def extract_rotation_features(Q: np.ndarray) -> Dict[str, float]:
    """
    Ekstraktuje cechy z macierzy rotacji Q.
    """
    r = Q.shape[0]
    
    # ||I - Q||_F: jak daleko od identyczności
    I = np.eye(r)
    deviation = float(np.linalg.norm(I - Q, 'fro'))
    
    # Trace(Q): suma cosinusów kątów rotacji
    trace = float(np.trace(Q))
    
    # Kąty rotacji z eigenvalues zespolonych
    try:
        eigvals = np.linalg.eigvals(Q)
        angles = np.abs(np.angle(eigvals))  # w radianach
        total_angle = float(angles.sum())
        max_angle = float(angles.max())
    except:
        total_angle = 0.0
        max_angle = 0.0
    
    # Determinant (powinien być ~1 dla SO(r), ~-1 dla odbicia)
    det = float(np.linalg.det(Q))
    
    return {
        'deviation': deviation,
        'trace': trace,
        'total_angle': total_angle,
        'max_angle': max_angle,
        'det': det,
    }


# ============================================================================
# HOLONOMY DECOMPOSITION V4
# ============================================================================

class HolonomyDecompositionV4:
    """
    PRAWDZIWA Holonomia: Path-Ordered Frame Transport.
    
    Zamiast porównywać tylko F_0 i F_end, transportujemy ramkę
    po KAŻDYM kroku pętli i mierzymy skumulowaną holonomię.
    """
    
    def __init__(self, frame_dim: int = 6):
        self.frame_dim = frame_dim
    
    def compute_local_frame(self, encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Oblicza lokalną ramkę z mikro-chmury (używając log-map!)."""
        cloud_images = [image] + [fn(image) for fn in MICRO_INTERVENTIONS]
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        return compute_tangent_frame(embeddings, self.frame_dim)
    
    def compute_path_holonomy(self, encoder, start_image: Image.Image) -> Dict[str, float]:
        """
        Oblicza holonomię pętli przez path-ordered frame transport.
        
        Q_loop = Q_0 @ Q_1 @ ... @ Q_{m-1}
        """
        # Zbierz wszystkie punkty na pętli
        images = [start_image]
        current = start_image
        for name, fn in LOOP_STEPS:
            current = fn(current)
            images.append(current)
        
        # Oblicz ramki w każdym punkcie
        frames = []
        centers = []
        for img in images:
            base, frame = self.compute_local_frame(encoder, img)
            frames.append(frame)
            centers.append(base)
        
        # Oblicz operatory transportu Q_i między kolejnymi ramkami
        Q_operators = []
        for i in range(len(frames) - 1):
            Q_i = procrustes_rotation(frames[i], frames[i + 1])
            Q_operators.append(Q_i)
        
        # Holonomia jako iloczyn uporządkowany
        r = frames[0].shape[1]
        Q_loop = np.eye(r)
        for Q in Q_operators:
            Q_loop = Q_loop @ Q
        
        # Cechy z Q_loop
        loop_features = extract_rotation_features(Q_loop)
        
        # Dodatkowe cechy: H_trans (translacja centerów)
        H_trans = float(np.linalg.norm(centers[-1] - centers[0]))
        
        # Path length w tangent space
        path_length = sum(np.linalg.norm(centers[i+1] - centers[i]) for i in range(len(centers)-1))
        normalized_closure = H_trans / (path_length + 1e-8)
        
        return {
            'H_trans': H_trans,
            'normalized_closure': normalized_closure,
            'Q_deviation': loop_features['deviation'],
            'Q_trace': loop_features['trace'],
            'Q_total_angle': loop_features['total_angle'],
            'Q_max_angle': loop_features['max_angle'],
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy dla jednego obrazu."""
        features = self.compute_path_holonomy(encoder, image)
        
        return np.array([
            features['H_trans'],
            features['normalized_closure'],
            features['Q_deviation'],
            features['Q_trace'],
            features['Q_total_angle'],
            features['Q_max_angle'],
        ], dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        return [
            'H_trans',
            'normalized_closure',
            'Q_deviation',
            'Q_trace',
            'Q_total_angle',
            'Q_max_angle',
        ]


# ============================================================================
# MULTIPLE LOOPS VERSION
# ============================================================================

LOOPS = [
    # Loop 1
    [('jpeg_80', lambda img: jpeg_compression(img, 80)),
     ('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
     ('scale_0.9', lambda img: downscale_upscale(img, 0.9))],
    # Loop 2
    [('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
     ('jpeg_70', lambda img: jpeg_compression(img, 70)),
     ('scale_0.85', lambda img: downscale_upscale(img, 0.85))],
]


class HolonomyDecompositionV4Multi:
    """Wersja z wieloma pętlami."""
    
    def __init__(self, frame_dim: int = 6):
        self.frame_dim = frame_dim
        self.v4 = HolonomyDecompositionV4(frame_dim)
    
    def compute_path_holonomy_for_loop(self, encoder, start_image: Image.Image, loop_steps) -> Dict[str, float]:
        """Oblicza holonomię dla jednej pętli."""
        images = [start_image]
        current = start_image
        for name, fn in loop_steps:
            current = fn(current)
            images.append(current)
        
        frames = []
        centers = []
        for img in images:
            base, frame = self.v4.compute_local_frame(encoder, img)
            frames.append(frame)
            centers.append(base)
        
        r = frames[0].shape[1]
        Q_loop = np.eye(r)
        for i in range(len(frames) - 1):
            Q_i = procrustes_rotation(frames[i], frames[i + 1])
            Q_loop = Q_loop @ Q_i
        
        loop_features = extract_rotation_features(Q_loop)
        H_trans = float(np.linalg.norm(centers[-1] - centers[0]))
        
        return {
            'H_trans': H_trans,
            'Q_deviation': loop_features['deviation'],
            'Q_trace': loop_features['trace'],
            'Q_total_angle': loop_features['total_angle'],
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        all_features = []
        for loop_steps in LOOPS:
            feat = self.compute_path_holonomy_for_loop(encoder, image, loop_steps)
            all_features.extend([
                feat['H_trans'],
                feat['Q_deviation'],
                feat['Q_trace'],
                feat['Q_total_angle'],
            ])
        return np.array(all_features, dtype=np.float32)
