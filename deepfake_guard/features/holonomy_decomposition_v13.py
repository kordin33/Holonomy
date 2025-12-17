"""
holonomy_decomposition_v13.py - SMART ANOMALY ENGINE
Target: 0.90 AUC+

Improvements over V12:
1. Baseline Upgrade: 81D features (added curv_max, median).
2. Smart Patch Selection: Select Top-5 high-energy patches from 3x3 grid (ignores background).
3. Outlier Scores: Anomaly detection based on patch divergence from centroid.
4. Robust Aggregations: Range, AbsGap.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageStat
from typing import List, Tuple
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
# BASELINE V13 (81 features)
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
    """Compute 81D baseline features for a single image."""
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
        curv_max = np.max(angles) if angles else 0.0 # NEW: Max curvature
        
        # NEW: Median
        step_median = np.median(steps) if steps else 0.0
        
        features.extend([
            H, L, L/(H+1e-8),
            np.std(steps) if len(steps)>1 else 0.0,
            np.mean(steps), np.max(steps),
            step_median, # NEW
            curv_mean,
            curv_max     # NEW
        ]) # 9 features per loop * 9 loops = 81 total
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# SMART PATCH SELECTOR
# ============================================================================

def get_smart_patches(image: Image.Image, k=5) -> List[Image.Image]:
    """Select Top-K patches with highest 'Energy' (StdDev) from 3x3 grid."""
    w, h = image.size
    dw, dh = w // 3, h // 3
    
    candidates = []
    
    for i in range(3):
        for j in range(3):
            # Crop patch
            box = (i*dw, j*dh, (i+1)*dw, (j+1)*dh)
            patch = image.crop(box)
            
            # Calculate energy (Simple StdDev of grayscale)
            # Efficient: ImageStat
            stat = ImageStat.Stat(patch.convert('L'))
            energy = stat.stddev[0]
            
            candidates.append((energy, patch))
            
    # Sort by energy desc
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return Top-K patches
    top_patches = [c[1].resize((224, 224), Image.LANCZOS) for c in candidates[:k]]
    return top_patches


# ============================================================================
# DECOMP V13 CLASS
# ============================================================================

class HolonomyDecompositionV13:
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global Baseline (81D)
        global_feats = compute_baseline_features(encoder, image)
        
        # 2. Smart Patches (5 * 81 = 405) - not stored fully
        patches = get_smart_patches(image, k=5)
        patch_feats = []
        for patch in patches:
            pf = compute_baseline_features(encoder, patch)
            patch_feats.append(pf)
        
        patch_feats = np.array(patch_feats)  # (5, 81)
        
        # 3. Aggregations (Vector-wise)
        patch_mean = np.mean(patch_feats, axis=0)
        patch_range = np.max(patch_feats, axis=0) - np.min(patch_feats, axis=0)
        abs_gap = np.abs(patch_mean - global_feats)
        
        # 4. Outlier Scores (Scalar stats)
        # Centroid mu = patch_mean
        # Distances r_i = ||p_i - mu||
        
        diffs = patch_feats - patch_mean
        r_dists = np.linalg.norm(diffs, axis=1) # (5,)
        
        r_mean = np.mean(r_dists)
        r_max = np.max(r_dists)
        r_std = np.std(r_dists)
        
        # Only 3 robust scalars from outlier analysis (simplification)
        outlier_stats = np.array([r_mean, r_max, r_std], dtype=np.float32)
        
        # Stack all
        # Global(81) + Mean(81) + Range(81) + Gap(81) + Outliers(3) = 327D
        return np.concatenate([
            global_feats,
            patch_mean,
            patch_range,
            abs_gap,
            outlier_stats
        ]).astype(np.float32)
