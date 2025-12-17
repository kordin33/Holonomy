"""
holonomy_decomposition_v7.py - SURGICAL STRIKE HOLONOMY

Skupia się na tym, co faktycznie działa:
1. Baseline Enhanced (9 loops, Chordal metric) - solidny fundament (>0.84)
2. Multi-Commutator (3 pary) - rozszerzona ocena przemienności degradacji (>0.76)
3. Patch Disagreement Focus (9 patchy) - tylko przestrzenna niespójność, zero globalnego szumu H3.

Eliminuje redundantne i zaszumione cechy (H2 Scale Law, Global H3).
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
import io


# ============================================================================
# SHARED UTILS (GEOMETRY & DEGRADATIONS)
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
    """Stabilna metryka na sferze: sqrt(2 - 2*dot)."""
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))

def cosine_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Kąt między wektorami."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    dot = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(dot))


# Registry
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


# ============================================================================
# PART 1: BASELINE ENHANCED (9 Loops)
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

class PartBaselineEnhanced:
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
            
            # Wektory różnic (do curvature)
            deltas = emb[1:] - emb[:-1]
            
            # Cechy geometryczne (Chordal metric!)
            steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
            H_chordal = chordal_dist(emb[0], emb[-1])
            path_len = sum(steps)
            
            # Curvature (angle change)
            curv = 0.0
            if len(deltas) >= 2:
                angles = [cosine_angle(deltas[i], deltas[i+1]) for i in range(len(deltas)-1)]
                curv = np.mean(angles)
            
            features.extend([
                H_chordal,
                path_len,
                path_len / (H_chordal + 1e-8),  # tortuosity
                np.std(steps) if len(steps)>1 else 0.0,
                curv  # New curvature!
            ])
            
        return np.array(features, dtype=np.float32)


# ============================================================================
# PART 2: MULTI-COMMUTATOR (3 Pairs)
# ============================================================================

class PartMultiCommutator:
    def __init__(self):
        # 3 pary operatorów do sprawdzenia przemienności
        self.pairs = [
            # 1. JPEG vs Blur (Standard)
            (lambda img: jpeg_compression(img, 75), lambda img: gaussian_blur(img, 0.6)),
            # 2. Sharpen vs JPEG (Structure vs Noise)
            (lambda img: sharpen(img, 1.5), lambda img: jpeg_compression(img, 80)),
            # 3. Scale vs Blur (Resolution vs Freq)
            (lambda img: downscale_upscale(img, 0.75), lambda img: gaussian_blur(img, 0.5)),
        ]
    
    def process(self, encoder, image) -> np.ndarray:
        features = []
        
        for opA, opB in self.pairs:
            # Paths
            img_A = opA(image)
            img_AB = opB(img_A)
            
            img_B = opB(image)
            img_BA = opA(img_B)
            
            embs = encoder.encode_batch([image, img_AB, img_BA], batch_size=3, show_progress=False)
            z0, zAB, zBA = l2_normalize(np.asarray(embs))
            
            # Commutator energy: d(AB, BA)
            comm_dist = chordal_dist(zAB, zBA)
            
            # Asymmetry: |d(0, AB) - d(0, BA)|
            clos_AB = chordal_dist(z0, zAB)
            clos_BA = chordal_dist(z0, zBA)
            asymmetry = abs(clos_AB - clos_BA)
            
            features.extend([comm_dist, asymmetry])
            
        return np.array(features, dtype=np.float32)


# ============================================================================
# PART 3: PATCH DISAGREEMENT FOCUS (9 Patches)
# ============================================================================

class PartPatchDisagreement:
    def __init__(self):
        # Krótka sekwencja H3 (wystarczy do zbadania spójności)
        self.seq = [
            lambda x: jpeg_compression(x, 80),
            lambda x: gaussian_blur(x, 0.5),
            lambda x: downscale_upscale(x, 0.9)
        ]

    def _get_9_patches(self, image):
        w, h = image.size
        cw, ch = w//3, h//3
        patches = []
        for i in range(3):
            for j in range(3):
                p = image.crop((i*cw, j*ch, (i+1)*cw, (j+1)*ch))
                patches.append(p.resize((224,224), Image.LANCZOS))
        return patches

    def process(self, encoder, image) -> np.ndarray:
        patches = self._get_9_patches(image)
        
        # Oblicz 1 metrykę dla każdego patcha: H_raw (displacement)
        # To najtańsza i najsilniejsza metryka.
        
        patch_H = []
        
        for p in patches:
            imgs = [p]
            curr = p
            for fn in self.seq:
                curr = fn(curr)
                imgs.append(curr)
            
            embs = encoder.encode_batch([imgs[0], imgs[-1]], batch_size=2, show_progress=False)
            z0, zEnd = l2_normalize(np.asarray(embs))
            patch_H.append(chordal_dist(z0, zEnd))
            
        patch_H = np.array(patch_H)
        
        # Zwracamy TYLKO statystyki rozrzutu (niespójności)
        return np.array([
            np.std(patch_H),                  # Standard deviation of response
            np.max(patch_H) - np.min(patch_H),# Range
            np.percentile(patch_H, 75) - np.percentile(patch_H, 25) # IQR
        ], dtype=np.float32)


# ============================================================================
# SURGICAL STRIKE CLASS
# ============================================================================

class HolonomyDecompositionV7:
    def __init__(self):
        self.base = PartBaselineEnhanced()
        self.comm = PartMultiCommutator()
        self.patch = PartPatchDisagreement()
        
    def extract_features(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        f_base = self.base.process(encoder, image)
        f_comm = self.comm.process(encoder, image)
        f_patch = self.patch.process(encoder, image)
        
        return {
            'baseline': f_base,
            'commutator': f_comm,
            'patch': f_patch,
            'all': np.concatenate([f_base, f_comm, f_patch])
        }
