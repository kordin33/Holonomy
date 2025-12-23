"""
holonomy_v21.py - V21: DENSE GRID & CENTROID METRICS
Goal: >0.90 AUC via better geometry and sampling (without H2).

Improvements over V18:
1. **3x3 Grid (9 patches)**: Better spatial coverage than 5-point.
2. **Centroid Metrics**: 'Centroid Drift' and 'Gyration Radius' to capture loop topology.
3. **Mega-Inference**: Optimized one-pass encoding for speed.

Metrics per loop (10D):
- H, L, L/H (Standard)
- Std, Mean, Max Step (Standard)
- Curvature (Standard)
- Centroid Drift (dist(z0, centroid)) [NEW]
- Gyration Radius (std(dist(pts, centroid))) [NEW]
- Tangent Alignment (mean cosine of steps) [NEW]

Total Dims: 90 (Global) + 90 (PatchMean) = 180D.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple
import io


# ============================================================================
# TRANSFORMS & CONFIG
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

# ============================================================================
# MATH UTILS
# ============================================================================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)

def chordal_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))

def cosine_angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))

# ============================================================================
# LOGIC
# ============================================================================

def prepare_images_recursive(image: Image.Image) -> Tuple[List[Image.Image], List[int]]:
    """Flatten all loop images for Mega-Inference."""
    all_imgs = []
    indices = [] # stores lengths
    
    for loop in BASELINE_LOOPS:
        imgs = [image] # Start
        curr = image
        for name in loop:
            curr = TRANSFORMS[name](curr)
            imgs.append(curr)
        all_imgs.extend(imgs)
        indices.append(len(imgs))
        
    return all_imgs, indices

def compute_loop_metrics(embs: np.ndarray) -> List[float]:
    """10D Metrics per loop."""
    # embs shape: (N_steps+1, 768)
    z0 = embs[0]
    steps = [chordal_dist(embs[i], embs[i+1]) for i in range(len(embs)-1)]
    H = chordal_dist(embs[0], embs[-1])
    L = sum(steps)
    
    # Differential vectors
    D = embs[1:] - embs[:-1]
    angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D)-1)] if len(D) >= 2 else []
    curv_mean = np.mean(angles) if angles else 0.0
    
    # NEW: Tangent Alignment (consistency of direction)
    # dot product of normalized steps
    D_norm = l2_normalize(D)
    tangents = [np.dot(D_norm[i], D_norm[i+1]) for i in range(len(D_norm)-1)] if len(D_norm) >= 2 else []
    tangent_align = np.mean(tangents) if tangents else 1.0 
    
    # NEW: Centroid Metrics
    centroid = np.mean(embs, axis=0) # Geometric center of the loop cloud
    centroid = l2_normalize(centroid) # Project back to sphere
    
    # Drift: How far centroid moved from origin z0
    drift = chordal_dist(z0, centroid)
    
    # Gyration: Spread of points around centroid
    radii = [chordal_dist(p, centroid) for p in embs]
    gyration = np.mean(radii)

    return [
        H, L, L/(H+1e-8),
        np.std(steps) if len(steps)>1 else 0.0,
        np.mean(steps), np.max(steps),
        curv_mean,
        drift,          # NEW
        gyration,       # NEW
        tangent_align   # NEW
    ]

class HolonomyV21:
    """
    V21: Grid 3x3 + Centroid Metrics.
    Optimized for GPU saturated inference.
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Prepare Patches (3x3 Grid)
        w, h = image.size
        sw, sh = w // 3, h // 3 # Step size
        ps = min(w, h) // 2     # Patch size (larger than step for overlap)
        
        # Grid centers
        centers = [
            (w//6, h//6), (w//2, h//6), (5*w//6, h//6),
            (w//6, h//2), (w//2, h//2), (5*w//6, h//2),
            (w//6, 5*h//6), (w//2, 5*h//6), (5*w//6, 5*h//6)
        ]
        
        patches = []
        for cx, cy in centers:
            left = max(0, cx - ps//2)
            top = max(0, cy - ps//2)
            right = min(w, left + ps)
            bottom = min(h, top + ps)
            # Adjust if out of bounds
            if right - left < ps: left = max(0, right - ps)
            if bottom - top < ps: top = max(0, bottom - ps)
            
            p = image.crop((left, top, left+ps, top+ps))
            patches.append(p.resize((224, 224), Image.LANCZOS))
            
        # 2. Mega-List Preparation
        mega_batch = []
        structure = [] # Stores ('type', offset, counts_list)
        
        # Global
        g_imgs, g_counts = prepare_images_recursive(image)
        structure.append(('global', len(mega_batch), g_counts))
        mega_batch.extend(g_imgs)
        
        # Patches
        for i, p in enumerate(patches):
            p_imgs, p_counts = prepare_images_recursive(p)
            structure.append(('patch', len(mega_batch), p_counts))
            mega_batch.extend(p_imgs)
            
        # 3. Encoding (One Pass)
        # 900+ images. Batch size 64 handled by encoder.encode_batch internally?
        # Typically encode_batch handles chunking. We trust it.
        all_embs = encoder.encode_batch(mega_batch, batch_size=128, show_progress=False)
        all_embs = l2_normalize(np.asarray(all_embs, dtype=np.float64))
        
        # 4. Feature Extraction/Reconstruction
        global_feats = []
        patch_feats_collection = []
        
        for kind, offset, counts in structure:
            cursor = offset
            feats = []
            for count in counts:
                loop_embs = all_embs[cursor : cursor+count]
                feats.extend(compute_loop_metrics(loop_embs))
                cursor += count
            
            if kind == 'global':
                global_feats = np.array(feats, dtype=np.float32)
            else:
                patch_feats_collection.append(feats)
                
        # Aggregate Patches (Mean)
        patch_feats_collection = np.array(patch_feats_collection, dtype=np.float32)
        patch_mean = np.mean(patch_feats_collection, axis=0) # 90D
        
        # Total 180D
        return np.concatenate([
            global_feats,
            patch_mean
        ]).astype(np.float32)
