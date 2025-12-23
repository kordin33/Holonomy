"""
holonomy_decomposition_v16.py - V16: ORTHOGONALIZED FUSION + GATING

Kluczowe nowe techniki:
1. Orthogonalizacja komutatora względem globalu (residual: comm - ŷ(comm|global))
2. Per-loop disagreement (zamiast 63D wszędzie)
3. Interaction features (global * disagreement)
4. Gating mechanism (komutator wzmacnia gdy global niepewny)

Target: przebić 0.8885 (Base + H2_V6_EXP)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple
import io
from sklearn.linear_model import Ridge


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

# Best commutator pairs (from V15 correlations)
COMMUTATOR_PAIRS = [
    ('blur_0.5', 'scale_0.9'),  # angle: -0.45
    ('jpeg_70', 'blur_0.5'),    # angle: -0.47
    ('blur_0.3', 'jpeg_90'),
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


def compute_loop_features(encoder, image: Image.Image, loop: list) -> np.ndarray:
    """7D features per loop."""
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
    
    return np.array([
        H, L, L/(H+1e-8),
        np.std(steps) if len(steps)>1 else 0.0,
        np.mean(steps), np.max(steps),
        curv_mean
    ], dtype=np.float32)


def compute_commutator_features_slim(encoder, image: Image.Image) -> np.ndarray:
    """Slim commutator: only best 3 pairs, only angle + d_comm (6D)."""
    all_imgs = [image]
    
    for name_A, name_B in COMMUTATOR_PAIRS:
        img_A = TRANSFORMS[name_A](image)
        img_B = TRANSFORMS[name_B](image)
        img_AB = TRANSFORMS[name_B](TRANSFORMS[name_A](image))
        img_BA = TRANSFORMS[name_A](TRANSFORMS[name_B](image))
        all_imgs.extend([img_A, img_B, img_AB, img_BA])
    
    embs = encoder.encode_batch(all_imgs, batch_size=16, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float32))
    
    e0 = embs[0]
    features = []
    
    for i in range(len(COMMUTATOR_PAIRS)):
        idx = 1 + i * 4
        eA, eB, eAB, eBA = embs[idx], embs[idx+1], embs[idx+2], embs[idx+3]
        
        d_comm = chordal_dist(eAB, eBA)
        u, v = eA - e0, eB - e0
        angle = cosine_angle(u, v)
        
        features.extend([d_comm, angle])
    
    return np.array(features, dtype=np.float32)


class HolonomyDecompositionV16:
    """
    V16: Orthogonalized Fusion + Per-Loop Disagreement.
    
    Features:
    - Global baseline: 63D
    - Per-loop disagreement: 18D (9 loops × 2 metrics)
    - Commutator (slim): 6D
    - Interaction (global × disagree top): 9D
    
    Total: 96D (much smaller than V15 276D!)
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global per-loop features
        global_per_loop = []
        for loop in BASELINE_LOOPS:
            lf = compute_loop_features(encoder, image, loop)
            global_per_loop.append(lf)
        global_per_loop = np.array(global_per_loop)  # (9, 7)
        global_flat = global_per_loop.flatten()  # 63D
        
        # 2. Patches per-loop
        patches = get_5_patches(image)
        patch_per_loop = []
        for loop in BASELINE_LOOPS:
            patch_feats = [compute_loop_features(encoder, p, loop) for p in patches]
            patch_per_loop.append(np.array(patch_feats))  # (5, 7)
        
        # 3. Per-loop disagreement (18D = 9 loops × 2 metrics)
        per_loop_disagree = []
        for loop_idx in range(9):
            global_loop = global_per_loop[loop_idx]  # (7,)
            patches_loop = patch_per_loop[loop_idx]   # (5, 7)
            
            # Metric 1: max-median per loop (scalar)
            patch_maxs = np.max(patches_loop, axis=0)
            patch_meds = np.median(patches_loop, axis=0)
            disagree1 = np.mean(patch_maxs - patch_meds)
            
            # Metric 2: median|patch - global| (scalar)
            diffs = np.abs(patches_loop - global_loop)
            disagree2 = np.median(diffs)
            
            per_loop_disagree.extend([disagree1, disagree2])
        
        per_loop_disagree = np.array(per_loop_disagree, dtype=np.float32)  # 18D
        
        # 4. Commutator (slim, 6D)
        comm_feats = compute_commutator_features_slim(encoder, image)
        
        # 5. Interaction: top global features × disagree signal
        # Take H values (indices 0, 7, 14, ...) and multiply by mean disagree
        H_values = global_flat[::7][:9]  # 9 H values
        mean_disagree_per_loop = per_loop_disagree[::2]  # 9 disagree1 values
        interaction = H_values * mean_disagree_per_loop  # 9D
        
        # Combine
        return np.concatenate([
            global_flat,         # 63D
            per_loop_disagree,   # 18D
            comm_feats,          # 6D
            interaction          # 9D
        ]).astype(np.float32)  # Total: 96D
