"""
holonomy_v18_opt.py - OPTIMIZED PRODUCTION V18 (SOTA 126D)

Strategy: MEGA-BATCH ENCODING for Global + Patch Mean.
Removes H2 overhead.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple, Dict
import io


# ============================================================================
# UTILS & CONFIG
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


def prepare_loop_images(image: Image.Image) -> Tuple[List[Image.Image], List[int]]:
    """Generates all images used in baseline loops and indices."""
    all_images = []
    loop_indices = [] # stores (start_idx, length) for each loop
    
    current_idx = 0
    for loop in BASELINE_LOOPS:
        imgs = [image] # Start with image itself
        curr = image
        for name in loop:
            curr = TRANSFORMS[name](curr)
            imgs.append(curr)
        
        all_images.extend(imgs)
        length = len(imgs)
        loop_indices.append((current_idx, length))
        current_idx += length
        
    return all_images, loop_indices

def compute_metrics_from_embeddings(embs_loop: np.ndarray) -> List[float]:
    steps = [chordal_dist(embs_loop[i], embs_loop[i+1]) for i in range(len(embs_loop)-1)]
    H = chordal_dist(embs_loop[0], embs_loop[-1])
    L = sum(steps)
    
    D = embs_loop[1:] - embs_loop[:-1]
    angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D)-1)] if len(D) >= 2 else []
    curv_mean = np.mean(angles) if angles else 0.0
    
    return [
        H, L, L/(H+1e-8),
        np.std(steps) if len(steps)>1 else 0.0,
        np.mean(steps), np.max(steps),
        curv_mean
    ]


class HolonomyV18Opt:
    """Optimized SOTA V18 Extractor (126D)"""
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Prepare ALL images
        # Global
        global_imgs, global_indices = prepare_loop_images(image)
        
        # Patches
        w, h = image.size
        ps = min(w, h) // 2
        patches = [
            image.crop((0, 0, ps, ps)), image.crop((w-ps, 0, w, ps)),
            image.crop((0, h-ps, ps, h)), image.crop((w-ps, h-ps, w, h)),
            image.crop(((w-ps)//2, (h-ps)//2, (w+ps)//2, (h+ps)//2))
        ]
        patches = [p.resize((224, 224), Image.LANCZOS) for p in patches]
        
        # Mega-Batch
        mega_batch = []
        mega_batch.extend(global_imgs)
        for p in patches:
            p_imgs, _ = prepare_loop_images(p)
            mega_batch.extend(p_imgs) # Indices same as global_indices structure
            
        # 2. Encode
        embs = encoder.encode_batch(mega_batch, batch_size=64, show_progress=False)
        embs = l2_normalize(np.asarray(embs, dtype=np.float64))
        
        # 3. Decode Embeddings
        cursor = 0
        
        # Global
        global_features = []
        for _, length in global_indices:
            loop_embs = embs[cursor : cursor+length]
            global_features.extend(compute_metrics_from_embeddings(loop_embs))
            cursor += length
        global_features = np.array(global_features, dtype=np.float32)
        
        # Patches
        patch_features_collection = []
        for _ in range(5): # 5 patches
            p_feat = []
            for _, length in global_indices: # same structure
                loop_embs = embs[cursor : cursor+length]
                p_feat.extend(compute_metrics_from_embeddings(loop_embs))
                cursor += length
            patch_features_collection.append(p_feat)
        
        patch_features_collection = np.array(patch_features_collection, dtype=np.float32)
        patch_mean = np.mean(patch_features_collection, axis=0)
        
        return np.concatenate([
            global_features,
            patch_mean
        ]).astype(np.float32)
