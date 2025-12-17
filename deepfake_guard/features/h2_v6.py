"""
h2_v6.py - H2 FRONTIER: Log-Map Area + WLS Curvature Estimation

Kluczowe zmiany matematyczne:
1. Log-map do przestrzeni stycznej w z0: u_A = log_{z0}(z_A)
2. Pole równoległoboku: area = ||u_A ∧ u_B|| = sqrt(||u_A||²||u_B||² - <u_A,u_B>²)
3. Komutator w stycznej: w = log_{z0}(z_AB) - log_{z0}(z_BA)
4. κ = ||w|| / (area + ε) - poprawna gęstość krzywizny
5. WLS fit: y ≈ κ·x1 + λ·x2 zamiast statystyk (mean/std/p90)
6. Out-of-plane: ||w_⊥|| - składowa ortogonalna do span(u_A, u_B)

Cechy: 8D CLEAN (κ, λ, R², MAD_res, ||w_⊥||/||w||, area_mean, kappa_raw_mean, stability)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple
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


# ============================================================================
# LOG-MAP GEOMETRY
# ============================================================================

def log_map(z0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Log-map from z0 to z on unit sphere.
    Returns tangent vector u at z0 such that exp_{z0}(u) ≈ z.
    """
    dot = np.clip(np.dot(z0, z), -1.0, 1.0)
    theta = np.arccos(dot)
    
    if theta < 1e-8:
        return np.zeros_like(z0)
    
    # Direction in tangent space
    v = z - dot * z0
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return np.zeros_like(z0)
    
    return theta * (v / v_norm)


def parallelogram_area(u_A: np.ndarray, u_B: np.ndarray) -> float:
    """
    Area of parallelogram spanned by u_A and u_B in tangent space.
    ||u_A ∧ u_B|| = sqrt(||u_A||²||u_B||² - <u_A,u_B>²)
    """
    norm_A = np.linalg.norm(u_A)
    norm_B = np.linalg.norm(u_B)
    dot_AB = np.dot(u_A, u_B)
    
    area_sq = norm_A**2 * norm_B**2 - dot_AB**2
    return np.sqrt(max(0, area_sq))


def project_out_of_plane(w: np.ndarray, u_A: np.ndarray, u_B: np.ndarray) -> np.ndarray:
    """
    Project w onto orthogonal complement of span(u_A, u_B).
    Returns w_⊥ = w - Π_{span}(w)
    """
    # Gram-Schmidt on u_A, u_B
    e1 = u_A / (np.linalg.norm(u_A) + 1e-10)
    u_B_proj = u_B - np.dot(u_B, e1) * e1
    e2 = u_B_proj / (np.linalg.norm(u_B_proj) + 1e-10)
    
    # Project w onto span
    w_in_span = np.dot(w, e1) * e1 + np.dot(w, e2) * e2
    
    return w - w_in_span


# ============================================================================
# CONFIG: 3×3 Grid (Adaptive Sweet Spot)
# ============================================================================

JPEG_Q = [90, 80, 70]  # 3 levels
BLUR_S = [0.3, 0.6, 0.9]  # 3 levels


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    return [image.crop((x, y, x+ps, y+ps)).resize((224, 224), Image.LANCZOS) for x, y in positions]


def compute_curvature_v6(encoder, image: Image.Image) -> np.ndarray:
    """
    Compute 8D curvature features using log-map geometry + WLS fit.
    """
    # 1. Prepare all images: z0, J_i (3), B_j (3), JB (9), BJ (9) = 25 images
    all_imgs = [image]
    
    J_imgs = [jpeg_compression(image, q) for q in JPEG_Q]
    B_imgs = [gaussian_blur(image, s) for s in BLUR_S]
    
    all_imgs.extend(J_imgs)  # 1-3
    all_imgs.extend(B_imgs)  # 4-6
    
    # JB and BJ
    JB_imgs = [gaussian_blur(J_imgs[i], BLUR_S[j]) for i in range(3) for j in range(3)]
    BJ_imgs = [jpeg_compression(B_imgs[j], JPEG_Q[i]) for i in range(3) for j in range(3)]
    
    all_imgs.extend(JB_imgs)  # 7-15
    all_imgs.extend(BJ_imgs)  # 16-24
    
    # 2. Encode (one batch)
    embs = encoder.encode_batch(all_imgs, batch_size=32, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float64))
    
    z0 = embs[0]
    z_J = embs[1:4]
    z_B = embs[4:7]
    z_JB = embs[7:16].reshape(3, 3, -1)
    z_BJ = embs[16:25].reshape(3, 3, -1)
    
    # 3. Log-maps
    u_J = np.array([log_map(z0, z) for z in z_J])  # (3, D)
    u_B = np.array([log_map(z0, z) for z in z_B])  # (3, D)
    
    # 4. For each (i,j) pair: compute area, w, kappa
    data_points = []  # (y, x1, x2, w_perp_ratio)
    
    for i in range(3):
        for j in range(3):
            u_A = u_J[i]
            u_B_j = u_B[j]
            
            # Area
            area = parallelogram_area(u_A, u_B_j)
            
            # Commutator in tangent space
            u_AB = log_map(z0, z_JB[i, j])
            u_BA = log_map(z0, z_BJ[i, j])
            w = u_AB - u_BA
            
            w_norm = np.linalg.norm(w)
            
            # Out-of-plane
            w_perp = project_out_of_plane(w, u_A, u_B_j)
            w_perp_norm = np.linalg.norm(w_perp)
            w_perp_ratio = w_perp_norm / (w_norm + 1e-10)
            
            # Model: y ≈ κ·area + λ·area·(||u_A||² + ||u_B||²)
            norm_A = np.linalg.norm(u_A)
            norm_B = np.linalg.norm(u_B_j)
            
            x1 = area
            x2 = area * (norm_A**2 + norm_B**2)
            y = w_norm
            
            data_points.append((y, x1, x2, area, w_perp_ratio))
    
    data = np.array(data_points)  # (9, 5)
    
    # 5. Filter noise-floor (area < threshold)
    areas = data[:, 3]
    tau = np.percentile(areas, 30)
    mask = areas >= tau
    
    if mask.sum() < 3:
        mask = np.ones(9, dtype=bool)
    
    y = data[mask, 0]
    X = data[mask, 1:3]  # (n, 2)
    
    # 6. WLS fit: weights ∝ x1²
    weights = X[:, 0]**2
    weights = weights / (weights.sum() + 1e-10)
    
    W = np.diag(weights)
    
    try:
        # Weighted least squares: (X'WX)^{-1} X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta = np.linalg.solve(XtWX + 1e-8 * np.eye(2), XtWy)
        
        kappa_hat = beta[0]
        lambda_hat = beta[1]
        
        # R² and residuals
        y_pred = X @ beta
        ss_res = np.sum(weights * (y - y_pred)**2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
        R2 = 1 - ss_res / (ss_tot + 1e-10)
        
        MAD_res = np.median(np.abs(y - y_pred))
        
    except:
        kappa_hat, lambda_hat, R2, MAD_res = 0, 0, 0, 0
    
    # 7. Additional features
    kappa_raw = data[:, 0] / (data[:, 3] + 1e-8)  # y / area
    kappa_raw_mean = np.mean(kappa_raw[mask])
    area_mean = np.mean(areas[mask])
    w_perp_ratio_mean = np.mean(data[mask, 4])
    
    # 8. Stability (jackknife variance of kappa)
    kappa_jk = []
    for k in range(mask.sum()):
        idx = np.where(mask)[0]
        leave_out = np.delete(idx, k)
        if len(leave_out) < 2:
            continue
        y_k = data[leave_out, 0]
        X_k = data[leave_out, 1:3]
        try:
            beta_k = np.linalg.lstsq(X_k, y_k, rcond=None)[0]
            kappa_jk.append(beta_k[0])
        except:
            pass
    
    stability = np.std(kappa_jk) if len(kappa_jk) > 1 else 0
    
    return np.array([
        kappa_hat,
        lambda_hat,
        R2,
        MAD_res,
        w_perp_ratio_mean,
        area_mean,
        kappa_raw_mean,
        stability
    ], dtype=np.float32)


class H2_V6:
    """
    H2 V6: Log-Map Area + WLS Curvature.
    Features: 8D CLEAN (global only for speed).
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # Global only for CLEAN version
        return compute_curvature_v6(encoder, image)
