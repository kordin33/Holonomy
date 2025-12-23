"""
h2_v7.py - H2 2-Head: CLEAN (Shape) + MAX (Amplitude)

CLEAN (12D) - dla ablacji z Baseline:
- κ0 (WLS curvature)
- λ (nonlinearity)
- sym_var (symmetry variance)
- k_high - k_mid (scale diff)
- w_perp_ratio (out-of-plane)
- stability (jackknife)

MAX (20D) - dla standalone:
- CLEAN (12D)
- E_A, E_B (amplitudes)
- E_mix, comm_norm_mean
- area_mean, w_mean, w_std, w_max
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
    area_sq = norm_A**2 * norm_B**2 - dot_AB**2
    return np.sqrt(max(0, area_sq))


def project_out_of_plane(w: np.ndarray, u_A: np.ndarray, u_B: np.ndarray) -> np.ndarray:
    e1 = u_A / (np.linalg.norm(u_A) + 1e-10)
    u_B_proj = u_B - np.dot(u_B, e1) * e1
    e2 = u_B_proj / (np.linalg.norm(u_B_proj) + 1e-10)
    w_in_span = np.dot(w, e1) * e1 + np.dot(w, e2) * e2
    return w - w_in_span


# 3×3 Grid (low, mid, high)
JPEG_Q = [90, 80, 70]
BLUR_S = [0.3, 0.6, 0.9]


def compute_h2_features(encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute H2 features with 2-head architecture.
    Returns: (CLEAN, MAX)
    """
    # Prepare images
    all_imgs = [image]
    J_imgs = [jpeg_compression(image, q) for q in JPEG_Q]
    B_imgs = [gaussian_blur(image, s) for s in BLUR_S]
    all_imgs.extend(J_imgs)
    all_imgs.extend(B_imgs)
    
    JB_imgs = [gaussian_blur(J_imgs[i], BLUR_S[j]) for i in range(3) for j in range(3)]
    BJ_imgs = [jpeg_compression(B_imgs[j], JPEG_Q[i]) for i in range(3) for j in range(3)]
    all_imgs.extend(JB_imgs)
    all_imgs.extend(BJ_imgs)
    
    # Encode
    embs = encoder.encode_batch(all_imgs, batch_size=32, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float64))
    
    z0 = embs[0]
    z_J = embs[1:4]
    z_B = embs[4:7]
    z_JB = embs[7:16].reshape(3, 3, -1)
    z_BJ = embs[16:25].reshape(3, 3, -1)
    
    # Log-maps
    u_J = np.array([log_map(z0, z) for z in z_J])
    u_B = np.array([log_map(z0, z) for z in z_B])
    
    # For each (i,j): area, w, kappa
    data = []
    for i in range(3):
        for j in range(3):
            u_A = u_J[i]
            u_B_j = u_B[j]
            area = parallelogram_area(u_A, u_B_j)
            
            u_AB = log_map(z0, z_JB[i, j])
            u_BA = log_map(z0, z_BJ[i, j])
            w = u_AB - u_BA
            w_norm = np.linalg.norm(w)
            
            w_perp = project_out_of_plane(w, u_A, u_B_j)
            w_perp_norm = np.linalg.norm(w_perp)
            
            norm_A = np.linalg.norm(u_A)
            norm_B = np.linalg.norm(u_B_j)
            norm_AB = np.linalg.norm(u_AB)
            
            data.append({
                'area': area, 'w_norm': w_norm, 'w_perp': w_perp_norm,
                'norm_A': norm_A, 'norm_B': norm_B, 'norm_AB': norm_AB,
                'i': i, 'j': j
            })
    
    # WLS fit: ||w|| ≈ κ*area + λ*area²
    areas = np.array([d['area'] for d in data])
    w_norms = np.array([d['w_norm'] for d in data])
    
    X = np.column_stack([areas, areas**2])
    weights = areas**2 / (areas.sum()**2 + 1e-10)
    W = np.diag(weights)
    
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ w_norms
        beta = np.linalg.solve(XtWX + 1e-6 * np.eye(2), XtWy)
        kappa, lam = beta[0], beta[1]
        
        y_pred = X @ beta
        ss_res = np.sum(weights * (w_norms - y_pred)**2)
        ss_tot = np.sum(weights * (w_norms - np.average(w_norms, weights=weights))**2)
        R2 = 1 - ss_res / (ss_tot + 1e-10)
        sym_var = np.std(w_norms - y_pred)
    except:
        kappa, lam, R2, sym_var = 0, 0, 0, 0
    
    # k_mid vs k_high (scale behavior)
    kappa_raw = w_norms / (areas + 1e-8)
    k_low = np.mean(kappa_raw[[0, 1, 3]])  # low intensity
    k_mid = np.mean(kappa_raw[[2, 4, 5]])  # mid
    k_high = np.mean(kappa_raw[[6, 7, 8]])  # high
    
    # Out-of-plane
    w_perp_ratios = np.array([d['w_perp'] / (d['w_norm'] + 1e-10) for d in data])
    w_perp_mean = np.mean(w_perp_ratios)
    
    # Stability (jackknife)
    kappa_jk = []
    for k in range(9):
        idx = np.arange(9) != k
        X_k, y_k = X[idx], w_norms[idx]
        try:
            beta_k = np.linalg.lstsq(X_k, y_k, rcond=None)[0]
            kappa_jk.append(beta_k[0])
        except:
            pass
    stability = np.std(kappa_jk) if len(kappa_jk) > 2 else 0
    
    # CLEAN (12D) - shape only
    CLEAN = np.array([
        kappa, lam, R2, sym_var,
        k_high - k_mid, k_high - k_low,
        k_mid / (k_low + 1e-8),
        w_perp_mean,
        stability,
        np.std(kappa_raw),
        np.percentile(kappa_raw, 90) - np.percentile(kappa_raw, 10),
        np.max(kappa_raw) - np.median(kappa_raw)
    ], dtype=np.float32)
    
    # MAX (20D) = CLEAN + amplitude
    E_A = np.mean([d['norm_A'] for d in data])
    E_B = np.mean([d['norm_B'] for d in data])
    E_mix = np.mean([d['norm_AB'] for d in data])
    comm_norm_mean = np.mean(kappa_raw)
    area_mean = np.mean(areas)
    w_mean = np.mean(w_norms)
    w_std = np.std(w_norms)
    w_max = np.max(w_norms)
    
    MAX = np.concatenate([
        CLEAN,
        [E_A, E_B, E_mix, comm_norm_mean, area_mean, w_mean, w_std, w_max]
    ]).astype(np.float32)
    
    return CLEAN, MAX


class H2_V7:
    """H2 V7: 2-Head (CLEAN 12D, MAX 20D)"""
    
    def extract_clean(self, encoder, image: Image.Image) -> np.ndarray:
        clean, _ = compute_h2_features(encoder, image)
        return clean
    
    def extract_max(self, encoder, image: Image.Image) -> np.ndarray:
        _, max_feats = compute_h2_features(encoder, image)
        return max_feats
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Default: returns MAX for standalone testing."""
        return self.extract_max(encoder, image)
