"""
holonomy_h1_h2_fixed.py - NAPRAWIONE H1 (Spectrum) i H2 (Scale Law)

NAPRAWY:
H1:
  - PCA na holonomy vectors (h = z_end - z_0), NIE na embeddingach!
  - whiten=True dla wyrównania skal
  - Stabilniejsze agregaty: E_head, E_tail, ratio, kurtosis

H2:
  - "area" = zmierzona siła degradacji w embedding-space, nie ręczna
  - 11 punktów skali (więcej = stabilniejsze)
  - Bez interceptu w featurach
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from scipy.stats import linregress, kurtosis as scipy_kurtosis
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


# ============================================================================
# LOOPS
# ============================================================================

BASE_LOOPS = [
    ['jpeg_70', 'blur_0.5', 'scale_0.9'],
    ['blur_0.5', 'jpeg_80', 'scale_0.75'],
    ['jpeg_60', 'blur_0.7', 'scale_0.9'],
]

TRANSFORMATIONS = {
    'jpeg_95': lambda img: jpeg_compression(img, 95),
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_85': lambda img: jpeg_compression(img, 85),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_75': lambda img: jpeg_compression(img, 75),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_65': lambda img: jpeg_compression(img, 65),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'jpeg_55': lambda img: jpeg_compression(img, 55),
    'jpeg_50': lambda img: jpeg_compression(img, 50),
    'blur_0.3': lambda img: gaussian_blur(img, 0.3),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'blur_0.9': lambda img: gaussian_blur(img, 0.9),
    'blur_1.1': lambda img: gaussian_blur(img, 1.1),
    'blur_1.3': lambda img: gaussian_blur(img, 1.3),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
}


# ============================================================================
# H1: HOLONOMY SPECTRUM - FIXED
# ============================================================================

class H1_HolonomySpectrum_Fixed:
    """
    Widmo niedomknięcia w PCA space.
    
    NAPRAWIONE:
    - PCA na HOLONOMY VECTORS (h = z_end - z_0), nie na embeddingach!
    - whiten=True dla wyrównania skal
    - Stabilniejsze agregaty: E_head, E_tail, ratio, kurtosis, gini
    """
    
    def __init__(self, pca_dim: int = 32, k_head: int = 8):
        self.pca_dim = pca_dim
        self.k_head = k_head  # ile komponentów to "head"
        self.pca: Optional[PCA] = None
    
    def collect_holonomy_vectors(self, encoder, images: List[Image.Image]) -> np.ndarray:
        """
        Zbiera wektory holonomii dla wszystkich obrazów i pętli.
        Returns: (n_images * n_loops, embedding_dim)
        """
        all_h_vectors = []
        
        for img in images:
            for loop in BASE_LOOPS:
                # Apply loop
                imgs = [img]
                current = img
                for t_name in loop:
                    current = TRANSFORMATIONS[t_name](current)
                    imgs.append(current)
                
                emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
                h = emb[-1] - emb[0]  # holonomy vector (768D)
                all_h_vectors.append(h)
        
        return np.array(all_h_vectors)
    
    def fit_pca(self, encoder, train_images: List[Image.Image]):
        """
        Fit PCA na HOLONOMY VECTORS z train images.
        """
        print("  Collecting holonomy vectors for PCA...")
        H_train = self.collect_holonomy_vectors(encoder, train_images)
        
        n_components = min(self.pca_dim, H_train.shape[1], H_train.shape[0])
        self.pca = PCA(n_components=n_components, whiten=True)  # whiten=True!
        self.pca.fit(H_train)
        print(f"  PCA fitted: {H_train.shape} -> {n_components}D (whiten=True)")
    
    def compute_spectrum_features(self, h_pca: np.ndarray) -> Dict[str, float]:
        """
        Oblicza stabilne cechy spektralne z h_pca.
        """
        r = len(h_pca)
        k = min(self.k_head, r // 2)  # head = pierwsze k komponentów
        
        h_sq = h_pca ** 2
        
        # Energia
        energy = float(np.sqrt(h_sq.sum()))
        
        # E_head, E_tail
        sorted_sq = np.sort(h_sq)[::-1]
        E_head = float(sorted_sq[:k].sum())
        E_tail = float(sorted_sq[k:].sum()) + 1e-10
        head_tail_ratio = E_head / E_tail
        
        # Entropy
        p = h_sq / (h_sq.sum() + 1e-10)
        p = p[p > 1e-10]
        entropy = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0
        
        # Kurtosis (excess kurtosis)
        try:
            kurt = float(scipy_kurtosis(h_pca, fisher=True))
            if not np.isfinite(kurt):
                kurt = 0.0
        except:
            kurt = 0.0
        
        # Gini coefficient (mierzy nierówność rozkładu energii)
        n = len(sorted_sq)
        if n > 1:
            index = np.arange(1, n + 1)
            gini = float((2 * np.sum(index * sorted_sq) / (n * sorted_sq.sum() + 1e-10)) - (n + 1) / n)
        else:
            gini = 0.0
        
        return {
            'energy': energy,
            'E_head': E_head,
            'E_tail': E_tail,
            'head_tail_ratio': head_tail_ratio,
            'entropy': entropy,
            'kurtosis': kurt,
            'gini': gini,
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy dla jednego obrazu."""
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca(encoder, train_images) first.")
        
        features = []
        
        for loop in BASE_LOOPS:
            # Apply loop
            imgs = [image]
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
                imgs.append(current)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            h = emb[-1] - emb[0]
            
            # Project to PCA space
            h_pca = self.pca.transform(h.reshape(1, -1))[0]
            
            # Compute features
            stats = self.compute_spectrum_features(h_pca)
            
            features.extend([
                stats['energy'],
                stats['head_tail_ratio'],
                stats['entropy'],
                stats['kurtosis'],
                stats['gini'],
            ])
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# H2: AREA/SCALE LAW - FIXED
# ============================================================================

class H2_AreaScaleLaw_Fixed:
    """
    Wykładnik potęgowy holonomii.
    
    NAPRAWIONE:
    - "area" = zmierzona siła degradacji w embedding-space (||z_mid - z_0||)
    - 11 punktów skali (stabilniejsze)
    - Bez interceptu w featurach
    """
    
    def __init__(self):
        # 11 poziomów intensywności: JPEG quality + blur sigma
        # Stała struktura: jpeg -> blur
        self.intensities = [
            (95, 0.2),   # bardzo łagodne
            (90, 0.3),
            (85, 0.4),
            (80, 0.5),
            (75, 0.6),
            (70, 0.7),
            (65, 0.8),
            (60, 0.9),
            (55, 1.0),
            (50, 1.1),
            (45, 1.2),   # bardzo mocne
        ]
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """
        Oblicza prawo skali z mierzoną siłą degradacji.
        """
        # Get z_0
        z_0 = encoder.encode_batch([image], batch_size=1, show_progress=False)[0]
        
        holonomies = []
        measured_strengths = []  # "area" = zmierzona siła degradacji
        
        for jpeg_q, blur_sigma in self.intensities:
            # Apply degradation sequence
            img1 = jpeg_compression(image, jpeg_q)
            img2 = gaussian_blur(img1, blur_sigma)
            
            # Get embeddings
            emb = encoder.encode_batch([img1, img2], batch_size=2, show_progress=False)
            z_mid = emb[0]  # po JPEG
            z_end = emb[1]  # po JPEG+blur
            
            # H = holonomy (displacement from start)
            H = float(np.linalg.norm(z_end - z_0))
            
            # s = "area" = measured degradation strength = ||z_mid - z_0||
            s = float(np.linalg.norm(z_mid - z_0))
            
            holonomies.append(H)
            measured_strengths.append(s)
        
        holonomies = np.array(holonomies)
        measured_strengths = np.array(measured_strengths)
        
        # Filter out near-zero to avoid log issues
        valid = (holonomies > 1e-6) & (measured_strengths > 1e-6)
        if valid.sum() < 3:
            # Not enough points
            return {
                'alpha': 0.0,
                'r_squared': 0.0,
                'residual_std': 1.0,
                'mean_H': float(holonomies.mean()),
                'slope_stability': 0.0,
            }
        
        log_H = np.log(holonomies[valid])
        log_s = np.log(measured_strengths[valid])
        
        # Fit: log H ≈ α log s + c (but we only use alpha, r_squared)
        result = linregress(log_s, log_H)
        
        alpha = result.slope
        r_squared = result.rvalue ** 2
        residuals = log_H - (alpha * log_s + result.intercept)
        residual_std = float(residuals.std())
        
        # Slope stability: variance of alpha from bootstrap (simplified: just use stderr)
        slope_stability = 1.0 / (result.stderr + 0.01)  # higher = more stable
        
        return {
            'alpha': float(alpha),
            'r_squared': float(r_squared),
            'residual_std': residual_std,
            'mean_H': float(holonomies.mean()),
            'slope_stability': float(slope_stability),
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['alpha'],
            stats['r_squared'],
            stats['residual_std'],
            stats['mean_H'],
            stats['slope_stability'],
        ], dtype=np.float32)
