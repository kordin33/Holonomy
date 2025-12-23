"""
holonomy_v18_refined_v2.py - V18.1 REFINED V2: ROBUST AGGREGATION & LOOP DELTA
Goal: Break 0.90 barrier by cleaning up V18 SOTA signals.

Architecture (135D):
1. **Global (63D):** Native resolution (preserving micro-artifacts).
2. **Patch Robust (63D):** Mean of 3 patches closest to Global (vector-wise).
   - Filters out background patches that dilute the signal.
   - Preserves vector consistency (unlike per-dim trimming).
3. **Loop Delta (9D):** Compressed disagreement energy per loop.
   - mean(|P - G|) aggregated over metric & patches per loop.

Total Dimensions: 135D.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

# ============================================================================
# TRANSFORMS & UTILS (SOTA V18 Standard)
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
    if na < 1e-10 or nb < 1e-10: return 0.0
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

# ============================================================================
# LOGIC
# ============================================================================

def compute_raw_features(encoder, image: Image.Image) -> np.ndarray:
    """Compute standard 63D features (No Log, No Resize)."""
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


class HolonomyRefinedV2:
    """
    V18.1 Refined V2 (135D)
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        try:
            # 1. Global Features (Native Resolution)
            # Preserves micro-artifacts / aliasing invisible at 224x224
            global_feats = compute_raw_features(encoder, image) # 63D
            
            # 2. Patches (Standard V12 fixed)
            w, h = image.size
            ps = min(w, h) // 2
            patches = [
                image.crop((0, 0, ps, ps)), # TL
                image.crop((w-ps, 0, w, ps)), # TR
                image.crop((0, h-ps, ps, h)), # BL
                image.crop((w-ps, h-ps, w, h)), # BR
                image.crop(((w-ps)//2, (h-ps)//2, (w+ps)//2, (h+ps)//2)) # Center
            ]
            patches = [p.resize((224, 224), Image.LANCZOS) for p in patches]
            
            # (5, 63) matrix
            P = np.array([compute_raw_features(encoder, p) for p in patches])
            
            # 3. Robust Aggregation (Filter Background)
            # Calculate distance of each patch vector to Global vector
            dists = np.linalg.norm(P - global_feats, axis=1) # (5,)
            
            # Select 3 closest patches (most similar to global structure)
            # This implicitly filters "pure background" or "pure outlier" patches
            closest_indices = np.argsort(dists)[:3] 
            patch_robust = np.mean(P[closest_indices], axis=0) # 63D
            
            # 4. Loop Delta Energy (9D Compressed Disagreement)
            # Residuals of chosen patches vs global
            # Reshape to (3, 9, 7)
            P_chosen = P[closest_indices].reshape(3, 9, 7)
            G_reshaped = global_feats.reshape(9, 7)
            
            # Absolute diff per metric
            abs_res = np.abs(P_chosen - G_reshaped[None, :, :]) # (3, 9, 7)
            
            # Mean over 3 patches AND 7 metrics -> 9 values (one per loop)
            # This captures "which loop is disagreeing most"
            delta_loop = np.mean(abs_res, axis=(0, 2)) # 9D
            
            # Silent Fail Protection
            if np.isnan(global_feats).any() or np.isnan(patch_robust).any():
                 return np.zeros(135, dtype=np.float32)

            # Total 135D
            return np.concatenate([
                global_feats,
                patch_robust,
                delta_loop
            ]).astype(np.float32)
            
        except Exception as e:
            return np.zeros(135, dtype=np.float32)
