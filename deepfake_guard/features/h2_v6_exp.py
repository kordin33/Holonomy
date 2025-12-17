"""
h2_v6_exp.py - H2 EXPERIMENTAL: Richardson Extrapolation + Symmetrization

Rozszerzenia eksperymentalne:
A) Out-of-plane już w V6 CORE
B) Symetryzacja komutatora (pary z dwóch stron skali)
C) Richardson extrapolation (mid vs high scale)

Cechy: 12D (8D core + 4D experimental)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List
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
    area_sq = norm_A**2 * norm_B**2 - dot_AB**2
    return np.sqrt(max(0, area_sq))


# ============================================================================
# CONFIG: Extended grid for Richardson
# ============================================================================

# Two scales: mid (α) and high (2α)
JPEG_MID = [92, 85]     # mid scale
JPEG_HIGH = [84, 70]    # high scale (stronger)
BLUR_MID = [0.25, 0.5]
BLUR_HIGH = [0.5, 1.0]


def compute_curvature_exp(encoder, image: Image.Image) -> np.ndarray:
    """
    Compute 12D features with Richardson extrapolation + symmetrization.
    """
    # 1. Prepare images for both scales
    all_imgs = [image]  # z0
    
    # Mid scale
    J_mid = [jpeg_compression(image, q) for q in JPEG_MID]
    B_mid = [gaussian_blur(image, s) for s in BLUR_MID]
    
    # High scale
    J_high = [jpeg_compression(image, q) for q in JPEG_HIGH]
    B_high = [gaussian_blur(image, s) for s in BLUR_HIGH]
    
    all_imgs.extend(J_mid)   # 1-2
    all_imgs.extend(B_mid)   # 3-4
    all_imgs.extend(J_high)  # 5-6
    all_imgs.extend(B_high)  # 7-8
    
    # JB and BJ for mid scale (2×2 = 4)
    JB_mid = [gaussian_blur(J_mid[i], BLUR_MID[j]) for i in range(2) for j in range(2)]
    BJ_mid = [jpeg_compression(B_mid[j], JPEG_MID[i]) for i in range(2) for j in range(2)]
    
    # JB and BJ for high scale (2×2 = 4)
    JB_high = [gaussian_blur(J_high[i], BLUR_HIGH[j]) for i in range(2) for j in range(2)]
    BJ_high = [jpeg_compression(B_high[j], JPEG_HIGH[i]) for i in range(2) for j in range(2)]
    
    all_imgs.extend(JB_mid)   # 9-12
    all_imgs.extend(BJ_mid)   # 13-16
    all_imgs.extend(JB_high)  # 17-20
    all_imgs.extend(BJ_high)  # 21-24
    
    # 2. Encode
    embs = encoder.encode_batch(all_imgs, batch_size=32, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float64))
    
    z0 = embs[0]
    
    def extract_scale_kappa(z_J, z_B, z_JB, z_BJ):
        """Extract mean kappa for a scale."""
        kappas = []
        for i in range(2):
            for j in range(2):
                u_A = log_map(z0, z_J[i])
                u_B = log_map(z0, z_B[j])
                u_AB = log_map(z0, z_JB[i*2 + j])
                u_BA = log_map(z0, z_BJ[i*2 + j])
                
                w = u_AB - u_BA
                area = parallelogram_area(u_A, u_B)
                kappa = np.linalg.norm(w) / (area + 1e-8)
                kappas.append(kappa)
        return np.mean(kappas), np.std(kappas)
    
    # Mid scale
    z_J_mid = embs[1:3]
    z_B_mid = embs[3:5]
    z_JB_mid = embs[9:13]
    z_BJ_mid = embs[13:17]
    
    kappa_mid, kappa_mid_std = extract_scale_kappa(z_J_mid, z_B_mid, z_JB_mid, z_BJ_mid)
    
    # High scale
    z_J_high = embs[5:7]
    z_B_high = embs[7:9]
    z_JB_high = embs[17:21]
    z_BJ_high = embs[21:25]
    
    kappa_high, kappa_high_std = extract_scale_kappa(z_J_high, z_B_high, z_JB_high, z_BJ_high)
    
    # 3. Richardson extrapolation
    # ||w(α)|| ≈ κ·α² + O(α³)
    # κ_richardson = (4·κ_high - κ_mid) / 3  (assuming 2:1 scale ratio)
    kappa_richardson = (4 * kappa_high - kappa_mid) / 3
    
    # 4. Scale consistency (how similar are kappas across scales)
    scale_ratio = kappa_high / (kappa_mid + 1e-8)
    scale_diff = abs(kappa_high - kappa_mid)
    
    # 5. Symmetrization: compare kappa from different orderings
    # Already implicit in our AB vs BA computation
    # Additional: variance across pairs
    sym_variance = (kappa_mid_std + kappa_high_std) / 2
    
    # 6. Core features (simplified from V6)
    # Mean commutator magnitude
    w_norms = []
    for z_JB, z_BJ in [(z_JB_mid, z_BJ_mid), (z_JB_high, z_BJ_high)]:
        for k in range(4):
            u_AB = log_map(z0, z_JB[k])
            u_BA = log_map(z0, z_BJ[k])
            w_norms.append(np.linalg.norm(u_AB - u_BA))
    
    w_mean = np.mean(w_norms)
    w_max = np.max(w_norms)
    
    return np.array([
        kappa_mid,
        kappa_high,
        kappa_richardson,
        scale_ratio,
        scale_diff,
        sym_variance,
        kappa_mid_std,
        kappa_high_std,
        w_mean,
        w_max,
        np.std(w_norms),
        np.median(w_norms)
    ], dtype=np.float32)


class H2_V6_EXP:
    """
    H2 V6 Experimental: Richardson + Symmetrization.
    Features: 12D
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        return compute_curvature_exp(encoder, image)
