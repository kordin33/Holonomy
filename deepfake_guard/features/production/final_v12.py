"""
final_v12.py - PRODUCTION V12 DECOMPOSITION
SOTA Candidate: Global + Patch Baseline Ensemble
Features: 252D (63 Global + 63 Patch_Mean + 63 Disagreement + 63 Gap)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict
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
# ENSEMBLE CONFIG
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

class FinalV12Ensemble:
    def __init__(self):
        pass

    def _compute_baseline_features(self, encoder, image: Image.Image) -> np.ndarray:
        features = []
        # Pre-compute all transformed images to allow batching if optimized further
        # Currently treating loop-by-loop for simplicity
        
        for loop in BASELINE_LOOPS:
            imgs = [image]
            curr = image
            for name in loop:
                curr = TRANSFORMS[name](curr)
                imgs.append(curr)
            
            # Batch encode steps
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            emb = l2_normalize(np.asarray(emb, dtype=np.float32))
            
            steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
            H = chordal_dist(emb[0], emb[-1])
            L = sum(steps)
            
            # Curvature
            D = emb[1:] - emb[:-1]
            angles = []
            if len(D) >= 2:
                angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D)-1)]
            curv_mean = np.mean(angles) if angles else 0.0
            
            features.extend([
                H, L, L/(H+1e-8),
                np.std(steps) if len(steps)>1 else 0.0,
                np.mean(steps), np.max(steps),
                curv_mean
            ])
        return np.array(features, dtype=np.float32)

    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global
        global_feats = self._compute_baseline_features(encoder, image)
        
        # 2. Patches (4 corners + center)
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
        
        patch_feats_list = []
        for p in patches:
            patch_feats_list.append(self._compute_baseline_features(encoder, p))
            
        patch_feats = np.array(patch_feats_list) # (5, 63)
        
        # 3. Aggregations
        patch_mean = np.mean(patch_feats, axis=0)
        patch_max = np.max(patch_feats, axis=0)
        disagreement = np.std(patch_feats, axis=0)
        gap = patch_max - global_feats
        
        # Total: 252D
        return np.concatenate([
            global_feats,
            disagreement,
            gap,
            patch_mean
        ]).astype(np.float32)
