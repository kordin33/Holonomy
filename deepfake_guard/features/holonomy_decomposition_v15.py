"""
holonomy_decomposition_v15.py - V15: COMMUTATOR CURVATURE + V12 CORE

Kluczowe zmiany:
1. V12 Core (Global + Gap + PatchMean) - sprawdzony baseline
2. Commutator Block - gęstość krzywizny z par transformacji (NOWY SYGNAŁ)
3. Improved Disagreement - median(|patch - global|) zamiast std

Commutator Block (frontier):
- 6 par transformacji (A, B)
- Dla każdej: d_comm = dist(eAB, eBA), dens = d_comm / (sA*sB+eps)
- Dodatkowe: ratio, angle

Target: > 0.89 AUC
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple
import io


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB').copy()
    buffer.close()
    return img

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def downscale_upscale(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    return image.resize((int(w*scale), int(h*scale)), Image.LANCZOS).resize((w, h), Image.LANCZOS)

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(factor)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)

def chordal_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))

def cosine_angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


# ============================================================================
# V12 CORE CONFIG
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


# ============================================================================
# COMMUTATOR PAIRS
# ============================================================================

COMMUTATOR_PAIRS = [
    ('blur_0.3', 'jpeg_90'),
    ('blur_0.5', 'scale_0.9'),
    ('jpeg_80', 'scale_0.9'),
    ('sharpen_1.5', 'jpeg_90'),
    ('blur_0.3', 'scale_0.75'),
    ('jpeg_70', 'blur_0.5'),
]


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    return [image.crop((x, y, x+ps, y+ps)).resize((224, 224), Image.LANCZOS) for x, y in positions]


def compute_baseline_features(encoder, image: Image.Image) -> np.ndarray:
    """V12-style baseline: 63D (9 loops × 7 features)."""
    features = []
    
    for loop in BASELINE_LOOPS:
        imgs = [image]
        curr = image
        for name in loop:
            curr = TRANSFORMS[name](curr)
            imgs.append(curr)
        
        emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        emb = l2_normalize(np.asarray(emb, dtype=np.float32))
        
        steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
        H = chordal_dist(emb[0], emb[-1])
        L = sum(steps)
        
        D = emb[1:] - emb[:-1]
        angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D)-1)] if len(D) >= 2 else []
        curv_mean = np.mean(angles) if angles else 0.0
        
        features.extend([
            H, L, L/(H+1e-8),
            np.std(steps) if len(steps)>1 else 0.0,
            np.mean(steps), np.max(steps),
            curv_mean
        ])
    
    return np.array(features, dtype=np.float32)


def compute_commutator_features(encoder, image: Image.Image) -> np.ndarray:
    """
    Commutator Block: 24D (6 pairs × 4 features).
    
    For each pair (A, B):
    - d_comm = dist(eAB, eBA)  (non-commutativity)
    - dens = d_comm / (sA*sB + eps)  (curvature density)
    - ratio = d_comm / (dist(e0, eAB) + eps)
    - angle = angle(eA-e0, eB-e0)
    """
    # Prepare all images for all pairs
    all_imgs = [image]  # e0
    
    for name_A, name_B in COMMUTATOR_PAIRS:
        img_A = TRANSFORMS[name_A](image)
        img_B = TRANSFORMS[name_B](image)
        img_AB = TRANSFORMS[name_B](TRANSFORMS[name_A](image))
        img_BA = TRANSFORMS[name_A](TRANSFORMS[name_B](image))
        all_imgs.extend([img_A, img_B, img_AB, img_BA])
    
    # Batch encode
    embs = encoder.encode_batch(all_imgs, batch_size=32, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float32))
    
    e0 = embs[0]
    features = []
    
    for i, (name_A, name_B) in enumerate(COMMUTATOR_PAIRS):
        idx = 1 + i * 4
        eA, eB, eAB, eBA = embs[idx], embs[idx+1], embs[idx+2], embs[idx+3]
        
        # Distances
        sA = chordal_dist(e0, eA)
        sB = chordal_dist(e0, eB)
        d_comm = chordal_dist(eAB, eBA)
        d_AB = chordal_dist(e0, eAB)
        
        # Curvature density
        dens = d_comm / (sA * sB + 1e-8)
        
        # Ratio
        ratio = d_comm / (d_AB + 1e-8)
        
        # Angle between directions
        u = eA - e0
        v = eB - e0
        angle = cosine_angle(u, v)
        
        features.extend([d_comm, dens, ratio, angle])
    
    return np.array(features, dtype=np.float32)


class HolonomyDecompositionV15:
    """
    V15: V12 Core + Commutator Curvature Block.
    
    Features:
    - Global baseline: 63D
    - Global commutator: 24D
    - Patch mean: 63D
    - Disagreement (robust): 63D
    - Gap: 63D
    
    Total: ~276D
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global features
        global_base = compute_baseline_features(encoder, image)  # 63D
        global_comm = compute_commutator_features(encoder, image)  # 24D
        
        # 2. Patches
        patches = get_5_patches(image)
        patch_feats = np.array([compute_baseline_features(encoder, p) for p in patches])
        
        # 3. Aggregations
        patch_mean = np.mean(patch_feats, axis=0)  # 63D
        
        # Robust disagreement: median(|patch - global|)
        disagreement = np.median(np.abs(patch_feats - global_base), axis=0)  # 63D
        
        # Gap (both directions)
        gap_hi = np.max(patch_feats, axis=0) - global_base  # 63D
        gap_lo = global_base - np.min(patch_feats, axis=0)  # 63D
        gap = np.maximum(gap_hi, gap_lo)  # Take max of both directions
        
        return np.concatenate([
            global_base,      # 63D
            global_comm,      # 24D
            patch_mean,       # 63D  
            disagreement,     # 63D
            gap               # 63D
        ]).astype(np.float32)  # Total: 276D
