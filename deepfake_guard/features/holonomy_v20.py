"""
holonomy_v20.py - FRONTIER V20: PATCHED H2 HOLONOMY

Objective: Break 0.90/0.92 barrier.
Method:
1. Global Baseline (63D)
2. Patch Mean Baseline (63D)
3. Global H2 CLEAN (16D)
4. Patch Mean H2 CLEAN (16D) - NEW!

Total: 158D.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict
import io


# ============================================================================
# TRANSFORMS & UTILS (Shared)
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

def log_map(z0: np.ndarray, z: np.ndarray) -> np.ndarray:
    dot = np.clip(np.dot(z0, z), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-8:
        return np.zeros_like(z0)
    v = z - dot * z0
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return np.zeros_like(z0)
    return theta * (v / v_norm)

def parallelogram_area(u_A: np.ndarray, u_B: np.ndarray) -> float:
    norm_A = np.linalg.norm(u_A)
    norm_B = np.linalg.norm(u_B)
    dot_AB = np.dot(u_A, u_B)
    return np.sqrt(max(0, norm_A**2 * norm_B**2 - dot_AB**2))


# ============================================================================
# CONFIGS
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

JPEG_MID = [92, 85]
JPEG_HIGH = [84, 70]
BLUR_MID = [0.25, 0.5]
BLUR_HIGH = [0.5, 1.0]


# ============================================================================
# LOGIC
# ============================================================================

def compute_baseline_features(encoder, image: Image.Image) -> np.ndarray:
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
        
        features.extend([H, L, L/(H+1e-8), np.std(steps) if len(steps)>1 else 0.0, np.mean(steps), np.max(steps), curv_mean])
    return np.array(features, dtype=np.float32)


def compute_scale_kappa(encoder, image: Image.Image, jpeg_q: list, blur_s: list) -> dict:
    all_imgs = [image]
    J_imgs = [jpeg_compression(image, q) for q in jpeg_q]
    B_imgs = [gaussian_blur(image, s) for s in blur_s]
    all_imgs.extend(J_imgs); all_imgs.extend(B_imgs)
    JB_imgs = [gaussian_blur(J_imgs[i], blur_s[j]) for i in range(2) for j in range(2)]
    BJ_imgs = [jpeg_compression(B_imgs[j], jpeg_q[i]) for i in range(2) for j in range(2)]
    all_imgs.extend(JB_imgs); all_imgs.extend(BJ_imgs)
    
    embs = encoder.encode_batch(all_imgs, batch_size=32, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float64))
    
    z0 = embs[0]
    z_J = embs[1:3]; z_B = embs[3:5]
    z_JB = embs[5:9].reshape(2, 2, -1); z_BJ = embs[9:13].reshape(2, 2, -1)
    
    u_J = np.array([log_map(z0, z) for z in z_J])
    u_B = np.array([log_map(z0, z) for z in z_B])
    
    kappas = []; w_norms = []
    for i in range(2):
        for j in range(2):
            w = log_map(z0, z_JB[i,j]) - log_map(z0, z_BJ[i,j])
            area = parallelogram_area(u_J[i], u_B[j])
            w_norm = np.linalg.norm(w)
            kappas.append(w_norm / (area + 1e-8))
            w_norms.append(w_norm)
            
    return {
        'kappa_mean': np.mean(kappas), 'kappa_std': np.std(kappas),
        'w_mean': np.mean(w_norms), 'w_std': np.std(w_norms),
        'w_max': np.max(w_norms), 'w_med': np.median(w_norms)
    }

def compute_h2_clean(encoder, image: Image.Image) -> np.ndarray:
    mid = compute_scale_kappa(encoder, image, JPEG_MID, BLUR_MID)
    high = compute_scale_kappa(encoder, image, JPEG_HIGH, BLUR_HIGH)
    
    return np.array([
        (4 * high['kappa_mean'] - mid['kappa_mean']) / 3, # Richardson
        high['kappa_mean'] / (mid['kappa_mean'] + 1e-8),
        high['kappa_mean'] - mid['kappa_mean'],
        (mid['kappa_std'] + high['kappa_std']) / 2,
        abs(high['w_mean'] - mid['w_mean']) / (mid['w_mean'] + 1e-8),
        high['kappa_std'], mid['kappa_std'],
        high['kappa_mean'] / (high['kappa_std'] + 1e-8),
        mid['kappa_mean'] / (mid['kappa_std'] + 1e-8),
        high['w_std'] / (high['w_mean'] + 1e-8),
        mid['w_std'] / (mid['w_mean'] + 1e-8),
        high['w_max'] - high['w_med'],
        mid['w_max'] - mid['w_med'],
        (high['w_max'] - high['w_med']) / (mid['w_max'] - mid['w_med'] + 1e-8),
        (high['kappa_mean'] / (mid['kappa_mean'] + 1e-8)) - 1,
        abs(((4 * high['kappa_mean'] - mid['kappa_mean']) / 3) - high['kappa_mean'])
    ], dtype=np.float32)


class HolonomyV20:
    """FRONTIER V20: Global + PM + H2_Global + H2_PM"""
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global Baseline
        try: g = compute_baseline_features(encoder, image)
        except: g = np.zeros(63, dtype=np.float32)
        
        # 2. H2 Global
        try: h2g = compute_h2_clean(encoder, image)
        except: h2g = np.zeros(16, dtype=np.float32)
        
        # Patches
        try:
            w, h = image.size
            ps = min(w, h) // 2
            patches = [
                image.crop((0, 0, ps, ps)), image.crop((w-ps, 0, w, ps)),
                image.crop((0, h-ps, ps, h)), image.crop((w-ps, h-ps, w, h)),
                image.crop(((w-ps)//2, (h-ps)//2, (w+ps)//2, (h+ps)//2))
            ]
            patches = [p.resize((224, 224), Image.LANCZOS) for p in patches]
            
            # 3. Patch Mean Baseline
            pm_list = [compute_baseline_features(encoder, p) for p in patches]
            pm = np.mean(pm_list, axis=0)
            
            # 4. Patch Mean H2 (NEW)
            h2p_list = [compute_h2_clean(encoder, p) for p in patches]
            h2pm = np.mean(h2p_list, axis=0) # 16D
             
        except:
             pm = np.zeros(63, dtype=np.float32)
             h2pm = np.zeros(16, dtype=np.float32)
        
        return np.concatenate([g, pm, h2g, h2pm]).astype(np.float32)
