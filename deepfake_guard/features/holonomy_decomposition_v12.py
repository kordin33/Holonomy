"""
holonomy_decomposition_v12.py - GLOBAL + PATCH BASELINE ENSEMBLE

Cel: 0.90 AUC (Standalone Decomp)
Strategia: Zamiast wymyślać skomplikowane feature'y, uruchamiamy sprawdzony
Baseline V4 (0.877) osobno na całym obrazie I osobno na każdym z 5 patchy.

Jeśli deepfake ma artefakt w jednym miejscu (np. oko), patch "top-left" wykryje
inną geometrię degradacji niż reszta. Ta niespójność = sygnał.

Feature vector: 63 (global) + 5 * 63 (patches) = 378D
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List
import io


# ============================================================================
# UTILS
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
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.linalg.norm(a64)
    nb = np.linalg.norm(b64)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    dot = np.clip(np.dot(a64, b64) / (na * nb), -1.0, 1.0)
    return float(np.arccos(dot))


# ============================================================================
# BASELINE (Same as V3/V4 - the 0.877 engine)
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


def compute_baseline_features(encoder, image: Image.Image) -> np.ndarray:
    """Compute 63D baseline features for a single image."""
    features = []
    
    for loop in BASELINE_LOOPS:
        imgs = [image]
        curr = image
        for name in loop:
            curr = TRANSFORMS[name](curr)
            imgs.append(curr)
        
        emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        emb = l2_normalize(np.asarray(emb, dtype=np.float32))
        
        # Step distances
        steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
        H = chordal_dist(emb[0], emb[-1])
        L = sum(steps)
        
        # Curvature
        D = emb[1:] - emb[:-1]
        angles = []
        if len(D) >= 2:
            angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D)-1)]
        curv_mean = np.mean(angles) if angles else 0.0
        curv_max = np.max(angles) if angles else 0.0
        
        features.extend([
            H, L, L/(H+1e-8),
            np.std(steps) if len(steps)>1 else 0.0,
            np.mean(steps), np.max(steps),
            curv_mean
        ])
    
    return np.array(features, dtype=np.float32)


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    """4 corners + center."""
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    patches = []
    for x, y in positions:
        patch = image.crop((x, y, x + ps, y + ps))
        patches.append(patch.resize((224, 224), Image.LANCZOS))
    return patches


# ============================================================================
# DECOMP V12 CLASS
# ============================================================================

class HolonomyDecompositionV12:
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global Baseline (63D)
        global_feats = compute_baseline_features(encoder, image)
        
        # 2. Patch Baselines (5 * 63 = 315D)
        patches = get_5_patches(image)
        patch_feats = []
        for patch in patches:
            pf = compute_baseline_features(encoder, patch)
            patch_feats.append(pf)
        
        patch_feats = np.array(patch_feats)  # (5, 63)
        
        # 3. Aggregated Patch Stats (compress 5x63 to ~30D for efficiency)
        # Mean, Std, Max, Min per feature (4 * 63 = 252D is too much)
        # Better: just flatten + add disagreement stats
        
        # Disagreement: Std across patches for each loop
        disagreement = np.std(patch_feats, axis=0)  # (63,)
        
        # Flatten all patch features
        patch_flat = patch_feats.flatten()  # (315,)
        
        # Combine: Global (63) + Patches (315) + Disagreement (63) = 441D
        # That's a lot. Let's reduce to: Global + Disagreement + Patch Aggregates
        
        patch_mean = np.mean(patch_feats, axis=0)
        patch_max = np.max(patch_feats, axis=0)
        patch_min = np.min(patch_feats, axis=0)
        
        # Global vs Patch Gap (anomaly detector)
        gap = patch_max - global_feats
        
        return np.concatenate([
            global_feats,     # 63D
            disagreement,     # 63D  
            gap,              # 63D
            patch_mean,       # 63D
        ]).astype(np.float32)  # Total: 252D
