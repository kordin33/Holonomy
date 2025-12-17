"""
holonomy_decomposition_v5.py - NAPRAWIONA Holonomia V5

NAPRAWY:
1. Wspólny tangent space (base0 ze startu) - wszystkie ramki w tym samym układzie!
2. Cosine/geodesic distance zamiast L2
3. Commutator loop (A∘B vs B∘A) - prawdziwe domknięcie
4. Step-wise features (nie tylko Q_loop)
5. Większy micro-cloud (25 punktów)
6. Wymuszone SO(r) (det=+1)
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

def contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


# WIĘKSZY micro-cloud (25 punktów!)
MICRO_INTERVENTIONS = [
    lambda img: jpeg_compression(img, 98),
    lambda img: jpeg_compression(img, 95),
    lambda img: jpeg_compression(img, 92),
    lambda img: jpeg_compression(img, 89),
    lambda img: jpeg_compression(img, 86),
    lambda img: gaussian_blur(img, 0.15),
    lambda img: gaussian_blur(img, 0.25),
    lambda img: gaussian_blur(img, 0.35),
    lambda img: gaussian_blur(img, 0.45),
    lambda img: downscale_upscale(img, 0.98),
    lambda img: downscale_upscale(img, 0.95),
    lambda img: downscale_upscale(img, 0.92),
    lambda img: sharpen(img, 1.05),
    lambda img: sharpen(img, 1.10),
    lambda img: sharpen(img, 1.15),
    lambda img: brightness(img, 0.97),
    lambda img: brightness(img, 0.99),
    lambda img: brightness(img, 1.01),
    lambda img: brightness(img, 1.03),
    lambda img: contrast(img, 0.97),
    lambda img: contrast(img, 0.99),
    lambda img: contrast(img, 1.01),
    lambda img: contrast(img, 1.03),
]


# COMMUTATOR LOOP: A∘B vs B∘A (prawdziwe domknięcie!)
# A = jpeg, B = blur
COMMUTATOR_A = lambda img: jpeg_compression(img, 75)
COMMUTATOR_B = lambda img: gaussian_blur(img, 0.6)


# ============================================================================
# SPHERICAL GEOMETRY
# ============================================================================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize vector or array of vectors."""
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)


def spherical_mean(embeddings: np.ndarray) -> np.ndarray:
    """Spherical mean (znormalizowana suma)."""
    mean = embeddings.sum(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-10)


def geodesic_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic distance na sferze (arccos)."""
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.arccos(dot))


def chordal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chordal distance: sqrt(2 - 2*dot)."""
    dot = np.dot(a, b)
    return float(np.sqrt(max(0, 2.0 - 2.0 * dot)))


def log_map_to_base(point: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Log-map: projekcja punktu na tangent space przy base."""
    dot = np.clip(np.dot(point, base), -1.0, 1.0)
    v = point - dot * base
    norm_v = np.linalg.norm(v)
    if norm_v > 1e-8:
        theta = np.arccos(dot)
        v = v * (theta / norm_v)
    return v


# ============================================================================
# FRAME COMPUTATION (wspólny tangent space!)
# ============================================================================

def compute_frame_in_shared_tangent(embeddings: np.ndarray, base: np.ndarray, 
                                     frame_dim: int = 8) -> np.ndarray:
    """
    Oblicza ramkę we WSPÓLNYM tangent space przy base.
    
    Wszystkie ramki są teraz w tym samym układzie współrzędnych!
    """
    # L2 normalize
    embeddings = l2_normalize(embeddings)
    
    # Log-map wszystkich punktów do tangent space przy base
    tangents = np.array([log_map_to_base(e, base) for e in embeddings])
    
    # Usuń zerowe wektory
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > 1e-8
    if valid.sum() < 3:
        return np.eye(embeddings.shape[1], frame_dim)
    
    tangents = tangents[valid]
    
    # SVD na tangent vectors
    U, S, Vt = np.linalg.svd(tangents, full_matrices=False)
    
    r = min(frame_dim, Vt.shape[0])
    frame = Vt[:r].T  # (d, r)
    
    return frame


# ============================================================================
# PROCRUSTES z wymuszonym SO(r)
# ============================================================================

def procrustes_so(F_i: np.ndarray, F_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procrustes z wymuszonym SO(r) (bez odbicia).
    
    Returns:
        Q: macierz rotacji w SO(r)
        S: singular values (cosines of principal angles)
    """
    r = min(F_i.shape[1], F_j.shape[1])
    F_i = F_i[:, :r]
    F_j = F_j[:, :r]
    
    M = F_i.T @ F_j
    U, S, Vt = np.linalg.svd(M)
    
    # Wymuś det = +1 (SO(r) zamiast O(r))
    Q = Vt.T @ U.T
    if np.linalg.det(Q) < 0:
        # Flip ostatni wektor singularny
        Vt[-1, :] *= -1
        Q = Vt.T @ U.T
    
    return Q, S


def extract_rotation_features(Q: np.ndarray, S: np.ndarray) -> Dict[str, float]:
    """Cechy z macierzy rotacji i singular values."""
    r = Q.shape[0]
    
    # ||I - Q||_F
    deviation = float(np.linalg.norm(np.eye(r) - Q, 'fro'))
    
    # Trace(Q) = Σcos(θ_i)
    trace = float(np.trace(Q))
    
    # Principal angles z S (cos(θ) = s)
    S_clipped = np.clip(S, -1.0, 1.0)
    theta = np.arccos(S_clipped)
    theta_sum = float(theta.sum())
    theta_max = float(theta.max())
    theta_mean = float(theta.mean())
    
    return {
        'deviation': deviation,
        'trace': trace,
        'theta_sum': theta_sum,
        'theta_max': theta_max,
        'theta_mean': theta_mean,
    }


# ============================================================================
# HOLONOMY DECOMPOSITION V5
# ============================================================================

class HolonomyDecompositionV5:
    """
    NAPRAWIONA Holonomia V5 z:
    - Wspólnym tangent space
    - Geodesic distance
    - Commutator loop
    - Step-wise features
    """
    
    def __init__(self, frame_dim: int = 8):
        self.frame_dim = frame_dim
    
    def get_micro_cloud(self, encoder, image: Image.Image) -> np.ndarray:
        """Pobiera embeddingi z micro-cloud."""
        cloud_images = [image] + [fn(image) for fn in MICRO_INTERVENTIONS]
        embeddings = encoder.encode_batch(cloud_images, batch_size=len(cloud_images), show_progress=False)
        return np.asarray(embeddings, dtype=np.float32)
    
    def compute_commutator_holonomy(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza holonomię przez COMMUTATOR: A∘B vs B∘A.
        To jest prawdziwe domknięcie!
        """
        # Oryginał
        emb_0 = encoder.encode_batch([image], batch_size=1, show_progress=False)[0]
        emb_0 = l2_normalize(emb_0)
        
        # Ścieżka A∘B
        img_A = COMMUTATOR_A(image)
        img_AB = COMMUTATOR_B(img_A)
        emb_AB = encoder.encode_batch([img_AB], batch_size=1, show_progress=False)[0]
        emb_AB = l2_normalize(emb_AB)
        
        # Ścieżka B∘A
        img_B = COMMUTATOR_B(image)
        img_BA = COMMUTATOR_A(img_B)
        emb_BA = encoder.encode_batch([img_BA], batch_size=1, show_progress=False)[0]
        emb_BA = l2_normalize(emb_BA)
        
        # Commutator: różnica między A∘B i B∘A
        commutator_dist = geodesic_distance(emb_AB, emb_BA)
        
        # Closure (jak daleko od startu)
        closure_AB = geodesic_distance(emb_0, emb_AB)
        closure_BA = geodesic_distance(emb_0, emb_BA)
        
        return {
            'commutator_dist': commutator_dist,
            'closure_AB': closure_AB,
            'closure_BA': closure_BA,
            'closure_diff': abs(closure_AB - closure_BA),
        }
    
    def compute_frame_holonomy(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza holonomię ramek w WSPÓLNYM tangent space.
        """
        # Micro-cloud na starcie - to definiuje base0
        cloud_0 = self.get_micro_cloud(encoder, image)
        cloud_0 = l2_normalize(cloud_0)
        base_0 = spherical_mean(cloud_0)
        
        # Ramka startowa
        frame_0 = compute_frame_in_shared_tangent(cloud_0, base_0, self.frame_dim)
        
        # Sekwencja degradacji
        steps = [
            COMMUTATOR_A,
            COMMUTATOR_B,
            lambda img: downscale_upscale(img, 0.9),
        ]
        
        current = image
        frames = [frame_0]
        centers = [base_0]
        step_thetas = []
        step_traces = []
        
        for step_fn in steps:
            current = step_fn(current)
            
            # Micro-cloud w tym punkcie
            cloud_i = self.get_micro_cloud(encoder, current)
            cloud_i = l2_normalize(cloud_i)
            center_i = spherical_mean(cloud_i)
            
            # Ramka we WSPÓLNYM tangent space (base_0!)
            frame_i = compute_frame_in_shared_tangent(cloud_i, base_0, self.frame_dim)
            
            # Procrustes między kolejnymi ramkami
            Q_i, S_i = procrustes_so(frames[-1], frame_i)
            feat = extract_rotation_features(Q_i, S_i)
            
            step_thetas.append(feat['theta_sum'])
            step_traces.append(feat['trace'])
            
            frames.append(frame_i)
            centers.append(center_i)
        
        # Q_loop jako iloczyn
        r = frame_0.shape[1]
        Q_loop = np.eye(r)
        for i in range(len(frames) - 1):
            Q_i, _ = procrustes_so(frames[i], frames[i+1])
            Q_loop = Q_loop @ Q_i
        
        # Cechy z Q_loop
        deviation = float(np.linalg.norm(np.eye(r) - Q_loop, 'fro'))
        trace = float(np.trace(Q_loop))
        
        # Path length (geodesic)
        path_length = sum(geodesic_distance(centers[i], centers[i+1]) 
                         for i in range(len(centers)-1))
        
        # Closure
        closure = geodesic_distance(centers[0], centers[-1])
        normalized_closure = closure / (path_length + 1e-8)
        
        # Step-wise stats
        theta_sum = float(np.sum(step_thetas))
        theta_std = float(np.std(step_thetas)) if len(step_thetas) > 1 else 0.0
        trace_std = float(np.std(step_traces)) if len(step_traces) > 1 else 0.0
        
        return {
            'Q_deviation': deviation,
            'Q_trace': trace,
            'closure': closure,
            'normalized_closure': normalized_closure,
            'step_theta_sum': theta_sum,
            'step_theta_std': theta_std,
            'step_trace_std': trace_std,
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje wszystkie cechy."""
        # Commutator holonomy
        comm = self.compute_commutator_holonomy(encoder, image)
        
        # Frame holonomy
        frame = self.compute_frame_holonomy(encoder, image)
        
        return np.array([
            # Commutator (4)
            comm['commutator_dist'],
            comm['closure_AB'],
            comm['closure_BA'],
            comm['closure_diff'],
            # Frame (7)
            frame['Q_deviation'],
            frame['Q_trace'],
            frame['closure'],
            frame['normalized_closure'],
            frame['step_theta_sum'],
            frame['step_theta_std'],
            frame['step_trace_std'],
        ], dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        return [
            'commutator_dist', 'closure_AB', 'closure_BA', 'closure_diff',
            'Q_deviation', 'Q_trace', 'closure', 'normalized_closure',
            'step_theta_sum', 'step_theta_std', 'step_trace_std',
        ]
