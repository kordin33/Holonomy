"""
h2_v7_exp.py - H2 V7 EXP: Richardson Extrapolation + Multi-Scale

Richardson: κ_richardson = (4·κ_high - κ_mid) / 3
To eliminuje O(α³) i daje stabilniejszą κ w reżimie małej pętli.

Cechy: 16D (CLEAN) + 24D (MAX)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple
import io


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB').copy()
    buffer.close()
    return img

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)


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


# Two scales: mid (α) and high (2α)
JPEG_MID = [92, 85]
JPEG_HIGH = [84, 70]
BLUR_MID = [0.25, 0.5]
BLUR_HIGH = [0.5, 1.0]


def compute_scale_kappa(encoder, image: Image.Image, jpeg_q: list, blur_s: list) -> dict:
    """Compute kappa and related features for one scale."""
    all_imgs = [image]
    
    J_imgs = [jpeg_compression(image, q) for q in jpeg_q]
    B_imgs = [gaussian_blur(image, s) for s in blur_s]
    all_imgs.extend(J_imgs)
    all_imgs.extend(B_imgs)
    
    JB_imgs = [gaussian_blur(J_imgs[i], blur_s[j]) for i in range(2) for j in range(2)]
    BJ_imgs = [jpeg_compression(B_imgs[j], jpeg_q[i]) for i in range(2) for j in range(2)]
    all_imgs.extend(JB_imgs)
    all_imgs.extend(BJ_imgs)
    
    embs = encoder.encode_batch(all_imgs, batch_size=16, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float64))
    
    z0 = embs[0]
    z_J = embs[1:3]
    z_B = embs[3:5]
    z_JB = embs[5:9].reshape(2, 2, -1)
    z_BJ = embs[9:13].reshape(2, 2, -1)
    
    u_J = np.array([log_map(z0, z) for z in z_J])
    u_B = np.array([log_map(z0, z) for z in z_B])
    
    kappas = []
    w_norms = []
    areas = []
    
    for i in range(2):
        for j in range(2):
            u_A = u_J[i]
            u_B_j = u_B[j]
            area = parallelogram_area(u_A, u_B_j)
            
            u_AB = log_map(z0, z_JB[i, j])
            u_BA = log_map(z0, z_BJ[i, j])
            w = u_AB - u_BA
            w_norm = np.linalg.norm(w)
            
            kappa = w_norm / (area + 1e-8)
            kappas.append(kappa)
            w_norms.append(w_norm)
            areas.append(area)
    
    return {
        'kappa_mean': np.mean(kappas),
        'kappa_std': np.std(kappas),
        'w_mean': np.mean(w_norms),
        'w_std': np.std(w_norms),
        'w_max': np.max(w_norms),
        'w_med': np.median(w_norms),
        'area_mean': np.mean(areas),
    }


def compute_h2_exp_features(encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute H2 V7 EXP features with Richardson extrapolation.
    Returns: (CLEAN, MAX)
    """
    # Mid and High scale
    mid = compute_scale_kappa(encoder, image, JPEG_MID, BLUR_MID)
    high = compute_scale_kappa(encoder, image, JPEG_HIGH, BLUR_HIGH)
    
    # Richardson extrapolation: κ_rich = (4·κ_high - κ_mid) / 3
    kappa_richardson = (4 * high['kappa_mean'] - mid['kappa_mean']) / 3
    
    # Scale behavior
    scale_ratio = high['kappa_mean'] / (mid['kappa_mean'] + 1e-8)
    scale_diff = high['kappa_mean'] - mid['kappa_mean']
    
    # Consistency
    sym_var = (mid['kappa_std'] + high['kappa_std']) / 2
    w_consistency = abs(high['w_mean'] - mid['w_mean']) / (mid['w_mean'] + 1e-8)
    
    # CLEAN (16D) - shape only
    CLEAN = np.array([
        kappa_richardson,
        scale_ratio,
        scale_diff,
        sym_var,
        w_consistency,
        high['kappa_std'],
        mid['kappa_std'],
        high['kappa_mean'] / (high['kappa_std'] + 1e-8),  # SNR high
        mid['kappa_mean'] / (mid['kappa_std'] + 1e-8),   # SNR mid
        high['w_std'] / (high['w_mean'] + 1e-8),  # CV w high
        mid['w_std'] / (mid['w_mean'] + 1e-8),    # CV w mid
        high['w_max'] - high['w_med'],  # range high
        mid['w_max'] - mid['w_med'],    # range mid
        (high['w_max'] - high['w_med']) / (mid['w_max'] - mid['w_med'] + 1e-8),
        scale_ratio - 1,  # deviation from scale-invariance
        abs(kappa_richardson - high['kappa_mean']),  # Richardson correction magnitude
    ], dtype=np.float32)
    
    # MAX (24D) = CLEAN + amplitude
    MAX = np.concatenate([
        CLEAN,
        [
            mid['kappa_mean'], high['kappa_mean'],
            mid['w_mean'], high['w_mean'],
            mid['w_med'], high['w_med'],
            mid['area_mean'], high['area_mean']
        ]
    ]).astype(np.float32)
    
    return CLEAN, MAX


class H2_V7_EXP:
    """H2 V7 EXP: Richardson Extrapolation. CLEAN 16D, MAX 24D."""
    
    def extract_clean(self, encoder, image: Image.Image) -> np.ndarray:
        clean, _ = compute_h2_exp_features(encoder, image)
        return clean
    
    def extract_max(self, encoder, image: Image.Image) -> np.ndarray:
        _, max_feats = compute_h2_exp_features(encoder, image)
        return max_feats
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        return self.extract_max(encoder, image)
