"""
holonomy_v18_refined.py - V18 PRO: LOG-SPACE & DELTA FEATURES
Refining V18 SOTA (0.896) via feature engineering, not architecture.

Improvements:
1. **Scale Consistency:** Global image resized to 224x224 (matching patches).
2. **Log-Space Metrics:** log1p(H), log1p(L) etc. to fix heavy tails.
3. **Trimmed Mean:** Robust patch aggregation (mean of middle 3).
4. **Delta Encoding:** Features = [Global, Patch - Global].

Total Dimensions: 126D.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

# ============================================================================
# TRANSFORMS & UTILS
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

def compute_log_features(encoder, image: Image.Image) -> np.ndarray:
    """Compute 63D features with LOG-SPACE transformation."""
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
        
        # LOG-SPACE TRANSFORMS (Addressing heavy tails)
        # H, L, std, mean, max are logarithmic in nature
        features.extend([
            np.log1p(H), 
            np.log1p(L), 
            np.log((L+1e-8)/(H+1e-8)), # Ratio in log space
            np.log1p(np.std(steps)) if len(steps)>1 else 0.0,
            np.log1p(np.mean(steps)), 
            np.log1p(np.max(steps)),
            curv_mean # Angles are already constrained [0, pi], no log needed
        ])
    
    return np.array(features, dtype=np.float32)


class HolonomyV18Refined:
    """
    V18 PRO (126D)
    Global(Log) + Delta(LogPatch - LogGlobal)
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        try:
            # 1. Scale Consistency: Resize Global to 224x224
            # Prevents sensor noise variance due to resolution differences
            global_img = image.resize((224, 224), Image.LANCZOS)
            global_feats = compute_log_features(encoder, global_img)
            
            # 2. Patches (Standard V12 fixed)
            w, h = image.size
            ps = min(w, h) // 2
            patches = [
                image.crop((0, 0, ps, ps)),
                image.crop((w-ps, 0, w, ps)),
                image.crop((0, h-ps, ps, h)),
                image.crop((w-ps, h-ps, w, h)),
                image.crop(((w-ps)//2, (h-ps)//2, (w+ps)//2, (h+ps)//2))
            ]
            patches = [p.resize((224, 224), Image.LANCZOS) for p in patches]
            
            patch_feats_list = [compute_log_features(encoder, p) for p in patches] # (5, 63)
            p_mat = np.array(patch_feats_list)
            
            # 3. Robust Aggregation (Trimmed Mean)
            # Sort along patches (axis 0) and take mean of middle 3
            p_sorted = np.sort(p_mat, axis=0) # Sorts each feature column independently
            p_trimmed_mean = np.mean(p_sorted[1:4], axis=0) # Ignore 0 and 4 (min/max)
            
            # 4. Delta Encoding (Orthogonalization)
            # Since feats are in log space, difference = log ratio
            delta_feats = p_trimmed_mean - global_feats
            
            # Silent Fail Protection (NaN handling)
            if np.isnan(global_feats).any() or np.isnan(delta_feats).any():
                return np.zeros(126, dtype=np.float32)
                
            return np.concatenate([
                global_feats,
                delta_feats
            ]).astype(np.float32)
            
        except Exception as e:
            # Fallback
            return np.zeros(126, dtype=np.float32)

