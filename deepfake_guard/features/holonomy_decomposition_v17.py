"""
holonomy_decomposition_v17.py - V17.0: GLOBAL-ANCHORED DISAGREEMENT

Focus: Naprawa disagreement poprzez normalizację względem globala i statystyki per-loop (outlier-aware).
Patche: Standard V12 (4 rogi + center).
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple
import io


# ============================================================================
# TRANSFORMS & UTILS (V12 Standard)
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
    dot = np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))

def cosine_angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


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


def get_v12_patches(image: Image.Image) -> List[Image.Image]:
    """Standard V12 fixed patches (4 corners + center)."""
    w, h = image.size
    ps = min(w, h) // 2
    patches = [
        image.crop((0, 0, ps, ps)),
        image.crop((w-ps, 0, w, ps)),
        image.crop((0, h-ps, ps, h)),
        image.crop((w-ps, h-ps, w, h)),
        image.crop(((w-ps)//2, (h-ps)//2, (w+ps)//2, (h+ps)//2))
    ]
    return [p.resize((224, 224), Image.LANCZOS) for p in patches]


def compute_baseline_features(encoder, image: Image.Image) -> np.ndarray:
    """Compute 63D baseline features (9 loops x 7 metrics)."""
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


class HolonomyDecompositionV17:
    """
    V17.0: Global-Anchored Disagreement
    
    Features:
    - Global (63D)
    - Disagreement (27D) - per loop: median|R|, max|R|, spread
    - Patch Mean (63D) - for reference
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global features (63D)
        global_feats = compute_baseline_features(encoder, image)
        
        # 2. Patch features (5 x 63D)
        patches = get_v12_patches(image)
        patch_feats_list = [compute_baseline_features(encoder, p) for p in patches]
        patch_feats = np.array(patch_feats_list)
        
        # 3. Disagreement (Global-Anchored Residuals)
        # Reshape to (9 loops, 7 metrics)
        global_reshaped = global_feats.reshape(9, 7)
        patterns_reshaped = patch_feats.reshape(5, 9, 7)
        
        disagreement_features = []
        
        for l in range(9):
            G_loop = global_reshaped[l]  # (7,)
            P_loop = patterns_reshaped[:, l, :]  # (5, 7)
            
            # Global-anchored residual: R = (P - G) / (|G| + eps)
            R = (P_loop - G_loop) / (np.abs(G_loop) + 1e-6)
            R_abs = np.abs(R)
            
            # Aggregates over patches AND metrics within the loop
            # median_{i,m} |R[i,m]|
            d1_median = np.median(R_abs)
            
            # max_{i,m} |R[i,m]| (outlier-aware)
            d2_max = np.max(R_abs)
            
            # spread
            d3_spread = d2_max - d1_median
            
            disagreement_features.extend([d1_median, d2_max, d3_spread])
            
        disagreement_feats = np.array(disagreement_features, dtype=np.float32) # 27D
        
        # 4. Patch Mean (63D) - Standard aggregation
        patch_mean = np.mean(patch_feats, axis=0)
        
        return np.concatenate([
            global_feats,       # 0-63
            disagreement_feats, # 63-90 (27D)
            patch_mean          # 90-153 (63D)
        ]).astype(np.float32)
