"""
holonomy_decomposition_v8.py - GRADIENT FRAME HOLONOMY (Target: 0.90)

To jest próba przebicia 0.90 AUC poprzez "Prawdziwą Holonomię" na stabilnych ramkach.

NOWOŚĆ: Gradient Frame
Zamiast SVD na chmurze punktów, definiujemy bazę (ramkę) w każdym punkcie pętli
za pomocą deterministycznych "kierunków stresu":
- u_J: kierunek zmian pod wpływem JPEG
- u_B: kierunek zmian pod wpływem Blur
- u_S: kierunek zmian pod wpływem Scale

Holonomia Q mierzy rotację tyhc fizycznych kierunków w trakcie pętli.
Dodatkowo zawiera "Scalar Holonomy" (Baseline V7), który już ma 0.87.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
import io


# ============================================================================
# UTILS (CHORDAL METRIC - THE KING)
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
# PART 1: SCALAR HOLONOMY (Baseline V7 Enhanced) -> 0.87+
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

class ScalarHolonomy:
    """To jest to co w V7 dało 0.8728."""
    def process(self, encoder, image) -> np.ndarray:
        features = []
        for loop in BASELINE_LOOPS:
            imgs = [image]
            curr = image
            for name in loop:
                curr = TRANSFORMS[name](curr)
                imgs.append(curr)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            emb = l2_normalize(np.asarray(emb))
            
            steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
            H_chordal = chordal_dist(emb[0], emb[-1])
            path_len = sum(steps)
            
            features.extend([
                H_chordal,
                path_len,
                path_len / (H_chordal + 1e-8),
                np.std(steps) if len(steps)>1 else 0.0,
            ])
        return np.array(features, dtype=np.float32)


# ============================================================================
# PART 2: GRADIENT FRAME HOLONOMY (NEW!)
# ============================================================================

class GradientFrameHolonomy:
    """
    Buduje macierz rotacji Q porównując 'ramki stresu' na początku i na końcu pętli.
    Ramka = [kierunek_jpeg, kierunek_blur, kierunek_scale]
    """
    
    def __init__(self):
        # Definicja 'gradientów' (kierunków stresu)
        self.probes = [
            ('J', lambda img: jpeg_compression(img, 60)),
            ('B', lambda img: gaussian_blur(img, 0.8)),
            ('S', lambda img: downscale_upscale(img, 0.6))
        ]
        
        # Pętla testowa: SHARPEN -> JPEG -> BLUR (Hard Non-linear Loop)
        # To wymusza silną rotację ramki (utrata informacji strukturalnej)
        self.loop_fn = [
            lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
            lambda img: jpeg_compression(img, 70),
            lambda img: gaussian_blur(img, 0.5) 
        ]

    def _compute_frame(self, encoder, image, base_emb) -> np.ndarray:
        probe_imgs = [fn(image) for _, fn in self.probes]
        probe_embs = encoder.encode_batch(probe_imgs, batch_size=3, show_progress=False)
        probe_embs = l2_normalize(np.asarray(probe_embs))
        
        vecs = probe_embs - base_emb  # (3, Dim)
        self._last_norms = np.linalg.norm(vecs, axis=1)
        return vecs

    def process(self, encoder, image) -> np.ndarray:
        # 1. Start Frame
        z0 = l2_normalize(encoder.encode_batch([image], batch_size=1, show_progress=False)[0])
        F0 = self._compute_frame(encoder, image, z0) 
        norms0 = self._last_norms.copy()
        
        # 2. End Frame
        curr = image
        for fn in self.loop_fn:
            curr = fn(curr)
        zEnd = l2_normalize(encoder.encode_batch([curr], batch_size=1, show_progress=False)[0])
        FEnd = self._compute_frame(encoder, curr, zEnd)
        normsEnd = self._last_norms.copy()
        
        # 3. Features
        G0 = F0 @ F0.T
        GEnd = FEnd @ FEnd.T
        
        distortion_frob = float(np.linalg.norm(G0 - GEnd, 'fro'))
        norm_change = float(np.linalg.norm(norms0 - normsEnd))
        
        # Safe correlations
        def safe_cos(v1, v2):
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            return float(np.dot(v1, v2) / (n1 * n2))
            
        corrs = [safe_cos(F0[i], FEnd[i]) for i in range(3)]
        
        return np.array([
            distortion_frob,
            norm_change,
            1.0 - corrs[0], 
            1.0 - corrs[1], 
            1.0 - corrs[2], 
            float(norms0.mean()),
        ], dtype=np.float32)


# ============================================================================
# DECOMP V8 CLASS
# ============================================================================

class HolonomyDecompositionV8:
    def __init__(self):
        self.scalar = ScalarHolonomy()
        self.matrix = GradientFrameHolonomy()
        
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # Scalar Holonomy (Baseline V7)
        f_scalar = self.scalar.process(encoder, image)
        
        # Matrix Holonomy (New Gradient Frame)
        f_matrix = self.matrix.process(encoder, image)
        
        return np.concatenate([f_scalar, f_matrix])
