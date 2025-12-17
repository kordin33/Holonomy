"""
optimized_v3.py - PEŁNA IMPLEMENTACJA WSZYSTKICH POPRAWEK

ZASADY BAZOWE:
1. Chordal distance wszędzie: d(a,b) = sqrt(2 - 2*<a,b>)
2. Float64 w krytycznych miejscach (dystanse, logi, regresja)
3. Deterministyczne patche (4 rogi + center)
4. Semantyka: H_end, L_path, rho_straight, slack (nie "closure/loop")

BASELINE V2++: 7 cech per loop (63D total)
H2 V2++: Global fit + Piecewise fit + Geometry + Monotonicity (~13D)
H3 V2++: Dispersion + Patch stats + Patch Synchrony (~40D)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
from scipy.stats import linregress, spearmanr
import io


# ============================================================================
# SHARED HELPERS (Float64 precision!)
# ============================================================================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize to unit sphere."""
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)


def chordal_distance(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Chordal distance on sphere: sqrt(2 - 2*<a,b>). Returns float64."""
    dot = np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0)
    return np.float64(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))


def cosine_angle(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Angle between vectors (for curvature). Returns float64."""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.linalg.norm(a64)
    nb = np.linalg.norm(b64)
    if na < 1e-10 or nb < 1e-10:
        return np.float64(0.0)
    dot = np.clip(np.dot(a64, b64) / (na * nb), -1.0, 1.0)
    return np.float64(np.arccos(dot))


def triangle_features(z0: np.ndarray, zmid: np.ndarray, zend: np.ndarray) -> Tuple:
    """
    Compute triangle geometry features.
    Returns: (s, t, H, L, rho, slack) all in float64.
    """
    s = chordal_distance(z0, zmid)
    t = chordal_distance(zmid, zend)
    H = chordal_distance(z0, zend)
    L = s + t
    rho = H / (L + 1e-10)  # straightness
    slack = L - H          # triangle slack
    return s, t, H, L, rho, slack


def mad(x: np.ndarray) -> np.float64:
    """Median Absolute Deviation."""
    return np.float64(np.median(np.abs(x - np.median(x))))


# ============================================================================
# DEGRADATIONS
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

def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    arr = np.array(image).astype(np.float32) / 255.0
    return Image.fromarray((np.power(arr, gamma) * 255).astype(np.uint8))

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)

def identity(image: Image.Image) -> Image.Image:
    return image.copy()


TRANSFORMS = {
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_85': lambda img: jpeg_compression(img, 85),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_75': lambda img: jpeg_compression(img, 75),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_65': lambda img: jpeg_compression(img, 65),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'jpeg_50': lambda img: jpeg_compression(img, 50),
    'blur_0.3': lambda img: gaussian_blur(img, 0.3),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'blur_1.0': lambda img: gaussian_blur(img, 1.0),
    'scale_0.5': lambda img: downscale_upscale(img, 0.5),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
    'sharpen_1.5': lambda img: sharpen(img, 1.5),
    'sharpen_2.0': lambda img: sharpen(img, 2.0),
    'gamma_1.2': lambda img: adjust_gamma(img, 1.2),
    'contrast_1.3': lambda img: adjust_contrast(img, 1.3),
    'identity': identity,
}


# ============================================================================
# BASELINE V2++ (7 features per loop = 63D)
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


class BaselineV3:
    """
    Baseline V2++ with 7 features per loop:
    [H_end, L_path, rho_straight, std_step, curvature, step_mean, step_max]
    Total: 7 * 9 = 63D
    """
    
    def compute_loop_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute 7 features from one trajectory."""
        E = l2_normalize(embeddings).astype(np.float64)
        
        # Step distances (float64)
        step_dists = np.array([chordal_distance(E[i], E[i+1]) for i in range(len(E)-1)])
        
        # Deltas for curvature
        D = E[1:] - E[:-1]
        
        # H_end (start -> end)
        H_end = chordal_distance(E[0], E[-1])
        
        # L_path (sum of steps)
        L_path = np.sum(step_dists)
        
        # rho_straight = H_end / L_path
        rho_straight = H_end / (L_path + 1e-10)
        
        # std_step
        std_step = np.std(step_dists, dtype=np.float64) if len(step_dists) > 1 else 0.0
        
        # curvature (mean angle between consecutive deltas)
        curvature = 0.0
        if len(D) >= 2:
            angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D) - 1)]
            curvature = np.mean(angles)
        
        # step_mean, step_max
        step_mean = np.mean(step_dists)
        step_max = np.max(step_dists)
        
        return np.array([
            H_end, L_path, rho_straight, std_step, curvature, step_mean, step_max
        ], dtype=np.float32)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        all_features = []
        
        for loop in BASELINE_LOOPS:
            imgs = [image]
            curr = image
            for name in loop:
                curr = TRANSFORMS[name](curr)
                imgs.append(curr)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            emb = np.asarray(emb, dtype=np.float32)
            
            feat = self.compute_loop_features(emb)
            all_features.extend(feat)
        
        return np.array(all_features, dtype=np.float32)


# ============================================================================
# H2 V2++ (Scale Law with full geometry) ~13D
# ============================================================================

class H2_V3:
    """
    H2 Scale Law with:
    - Triangle geometry (s, t, H, L, rho, slack)
    - Global OLS fit (alpha_all, r2_all, residual_std)
    - Piecewise fit (alpha_low, alpha_high, delta_alpha)
    - Geometry stats (rho_low, rho_high, delta_rho, slack_low, slack_high, delta_slack)
    - Monotonicity (spearman_H, spearman_rho)
    - Mini-commutator (d(z_AB, z_BA) for one intensity)
    
    Total: ~15D
    """
    
    def __init__(self):
        self.intensities = [
            (95, 0.2), (90, 0.3), (85, 0.4), (80, 0.5), (75, 0.6),
            (70, 0.7), (65, 0.8), (60, 0.9), (55, 1.0), (50, 1.1), (45, 1.2),
        ]
        self.split_idx = 5  # First 5 = low, rest = high
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        z_0 = l2_normalize(encoder.encode_batch([image], batch_size=1, show_progress=False)[0])
        
        # Collect all triangle features
        all_s, all_t, all_H, all_L, all_rho, all_slack = [], [], [], [], [], []
        
        for jpeg_q, blur_sigma in self.intensities:
            # z_mid (after JPEG)
            img_mid = jpeg_compression(image, jpeg_q)
            z_mid = l2_normalize(encoder.encode_batch([img_mid], batch_size=1, show_progress=False)[0])
            
            # z_end (after JPEG + blur)
            img_end = gaussian_blur(img_mid, blur_sigma)
            z_end = l2_normalize(encoder.encode_batch([img_end], batch_size=1, show_progress=False)[0])
            
            s, t, H, L, rho, slack = triangle_features(z_0, z_mid, z_end)
            
            all_s.append(s)
            all_t.append(t)
            all_H.append(H)
            all_L.append(L)
            all_rho.append(rho)
            all_slack.append(slack)
        
        # Convert to float64 arrays
        all_s = np.array(all_s, dtype=np.float64)
        all_H = np.array(all_H, dtype=np.float64)
        all_rho = np.array(all_rho, dtype=np.float64)
        all_slack = np.array(all_slack, dtype=np.float64)
        
        # Validity filter
        valid = (all_s > 1e-5) & (all_H > 1e-5)
        
        if valid.sum() < 6:
            return np.zeros(15, dtype=np.float32)
        
        # Global OLS fit: log(H) ~ alpha * log(s)
        log_s = np.log(all_s[valid])
        log_H = np.log(all_H[valid])
        
        try:
            result = linregress(log_s, log_H)
            alpha_all = float(result.slope)
            r2_all = float(result.rvalue ** 2)
            residual_std = float((log_H - (result.slope * log_s + result.intercept)).std())
        except:
            alpha_all, r2_all, residual_std = 0.0, 0.0, 1.0
        
        # Piecewise fit (low vs high)
        low_mask = np.zeros(len(all_s), dtype=bool)
        low_mask[:self.split_idx] = True
        low_mask = low_mask & valid
        high_mask = ~np.zeros(len(all_s), dtype=bool)
        high_mask[:self.split_idx] = False
        high_mask = high_mask & valid
        
        alpha_low, alpha_high = 0.0, 0.0
        if low_mask.sum() >= 3:
            try:
                res_low = linregress(np.log(all_s[low_mask]), np.log(all_H[low_mask]))
                alpha_low = float(res_low.slope)
            except:
                pass
        if high_mask.sum() >= 3:
            try:
                res_high = linregress(np.log(all_s[high_mask]), np.log(all_H[high_mask]))
                alpha_high = float(res_high.slope)
            except:
                pass
        
        delta_alpha = alpha_high - alpha_low
        
        # Geometry stats (rho, slack)
        rho_low_mean = float(all_rho[low_mask].mean()) if low_mask.sum() > 0 else 0.0
        rho_high_mean = float(all_rho[high_mask].mean()) if high_mask.sum() > 0 else 0.0
        delta_rho = rho_high_mean - rho_low_mean
        
        slack_low_mean = float(all_slack[low_mask].mean()) if low_mask.sum() > 0 else 0.0
        slack_high_mean = float(all_slack[high_mask].mean()) if high_mask.sum() > 0 else 0.0
        delta_slack = slack_high_mean - slack_low_mean
        
        # Monotonicity (Spearman)
        indices = np.arange(len(all_H))
        try:
            spearman_H, _ = spearmanr(indices[valid], all_H[valid])
        except:
            spearman_H = 0.0
        try:
            spearman_rho, _ = spearmanr(indices[valid], all_rho[valid])
        except:
            spearman_rho = 0.0
        
        # Mini-commutator (middle intensity: jpeg_70, blur_0.7)
        img_J = jpeg_compression(image, 70)
        img_JB = gaussian_blur(img_J, 0.7)
        img_B = gaussian_blur(image, 0.7)
        img_BJ = jpeg_compression(img_B, 70)
        
        embs = encoder.encode_batch([img_JB, img_BJ], batch_size=2, show_progress=False)
        z_JB, z_BJ = l2_normalize(np.asarray(embs))
        commutator_dist = chordal_distance(z_JB, z_BJ)
        
        return np.array([
            alpha_all, r2_all, residual_std,
            alpha_low, alpha_high, delta_alpha,
            rho_low_mean, rho_high_mean, delta_rho,
            slack_low_mean, slack_high_mean, delta_slack,
            float(spearman_H), float(spearman_rho),
            float(commutator_dist),
        ], dtype=np.float32)


# ============================================================================
# H3 V2++ (Dispersion + Patches + Synchrony) ~40D
# ============================================================================

H3_SEQUENCE = [
    lambda img: jpeg_compression(img, 90),
    lambda img: gaussian_blur(img, 0.3),
    lambda img: jpeg_compression(img, 75),
    lambda img: gaussian_blur(img, 0.5),
    lambda img: downscale_upscale(img, 0.9),
    lambda img: jpeg_compression(img, 60),
    lambda img: gaussian_blur(img, 0.7),
]

H3_PATCH_SEQUENCE = [
    lambda img: jpeg_compression(img, 85),
    lambda img: gaussian_blur(img, 0.4),
    lambda img: jpeg_compression(img, 70),
    lambda img: gaussian_blur(img, 0.6),
]


class H3_V3:
    """
    H3 Dispersion with:
    - Global: D_pairwise, trace_var, H_end, L_path, rho_straight, slack
    - Patch aggregations: min/max/range/std/MAD for D_pairwise and rho
    - Patch Synchrony: correlation of step vectors between patches
    
    Total: ~40D
    """
    
    def compute_dispersion(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute dispersion metrics from trajectory. Returns 6 features."""
        E = l2_normalize(embeddings).astype(np.float64)
        K = len(E)
        
        # (1) D_pairwise (mean chordal over all pairs)
        dists = []
        for i in range(K):
            for j in range(i+1, K):
                dists.append(chordal_distance(E[i], E[j]))
        D_pairwise = np.mean(dists) if dists else 0.0
        
        # (2) trace(var)
        var = np.var(E, axis=0, ddof=1)
        trace_var = np.sum(var)
        
        # (3) H_end, L_path, rho_straight, slack
        H_end = chordal_distance(E[0], E[-1])
        step_dists = [chordal_distance(E[i], E[i+1]) for i in range(K-1)]
        L_path = np.sum(step_dists)
        
        if L_path < 1e-5:
            rho_straight = 0.0
            slack = 0.0
        else:
            rho_straight = H_end / (L_path + 1e-10)
            slack = L_path - H_end
        
        return np.array([
            D_pairwise, trace_var, H_end, L_path, rho_straight, slack
        ], dtype=np.float64)
    
    def compute_step_vector(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute step distances as a vector (for synchrony)."""
        E = l2_normalize(embeddings).astype(np.float64)
        return np.array([chordal_distance(E[i], E[i+1]) for i in range(len(E)-1)])
    
    def get_grid_patches(self, image: Image.Image) -> List[Image.Image]:
        """Grid patches: 4 corners + center, resize 224."""
        w, h = image.size
        ps = min(w, h) // 2
        
        positions = [
            (0, 0), (w - ps, 0),
            (0, h - ps), (w - ps, h - ps),
            ((w - ps) // 2, (h - ps) // 2),
        ]
        
        patches = []
        for x, y in positions:
            patch = image.crop((x, y, x + ps, y + ps))
            patch = patch.resize((224, 224), Image.LANCZOS)
            patches.append(patch)
        
        return patches
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # ===== GLOBAL FEATURES =====
        imgs = [image]
        curr = image
        for fn in H3_SEQUENCE:
            curr = fn(curr)
            imgs.append(curr)
        
        emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        global_feat = self.compute_dispersion(np.asarray(emb, dtype=np.float32))
        
        # ===== PATCH FEATURES =====
        patches = self.get_grid_patches(image)
        
        patch_D_pairwise = []
        patch_rho = []
        patch_step_vectors = []
        
        for patch in patches:
            imgs = [patch]
            curr = patch
            for fn in H3_PATCH_SEQUENCE:
                curr = fn(curr)
                imgs.append(curr)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            emb = np.asarray(emb, dtype=np.float32)
            
            disp = self.compute_dispersion(emb)
            patch_D_pairwise.append(disp[0])  # D_pairwise
            patch_rho.append(disp[4])          # rho_straight
            
            step_vec = self.compute_step_vector(emb)
            patch_step_vectors.append(step_vec)
        
        patch_D_pairwise = np.array(patch_D_pairwise)
        patch_rho = np.array(patch_rho)
        
        # Patch aggregations for D_pairwise
        D_min = np.min(patch_D_pairwise)
        D_max = np.max(patch_D_pairwise)
        D_range = D_max - D_min
        D_std = np.std(patch_D_pairwise)
        D_mad = mad(patch_D_pairwise)
        D_global_gap = D_max - global_feat[0]  # max_patch - global
        
        # Patch aggregations for rho
        rho_min = np.min(patch_rho)
        rho_max = np.max(patch_rho)
        rho_range = rho_max - rho_min
        rho_std = np.std(patch_rho)
        rho_mad = mad(patch_rho)
        
        # ===== PATCH SYNCHRONY =====
        # Correlation matrix of step vectors between patches
        step_matrix = np.array(patch_step_vectors)  # (5, num_steps)
        
        if step_matrix.shape[1] >= 2:
            corr_matrix = np.corrcoef(step_matrix)
            # Extract off-diagonal elements
            offdiag = corr_matrix[np.triu_indices(5, k=1)]
            offdiag = offdiag[~np.isnan(offdiag)]  # Remove NaNs
            
            if len(offdiag) > 0:
                sync_mean = np.mean(offdiag)
                sync_std = np.std(offdiag)
                sync_min = np.min(offdiag)
            else:
                sync_mean, sync_std, sync_min = 0.0, 0.0, 0.0
        else:
            sync_mean, sync_std, sync_min = 0.0, 0.0, 0.0
        
        # ===== COMBINE ALL =====
        return np.array([
            # Global (6)
            *global_feat,
            # Patch D_pairwise aggregations (6)
            D_min, D_max, D_range, D_std, D_mad, D_global_gap,
            # Patch rho aggregations (5)
            rho_min, rho_max, rho_range, rho_std, rho_mad,
            # Patch synchrony (3)
            sync_mean, sync_std, sync_min,
        ], dtype=np.float32)


# ============================================================================
# COMBINED V3 (Per-block interface)
# ============================================================================

class CombinedV3:
    """Combines Baseline + H2 + H3 with per-block interface."""
    
    def __init__(self):
        self.baseline = BaselineV3()
        self.h2 = H2_V3()
        self.h3 = H3_V3()
    
    def extract_features(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        return {
            'baseline': self.baseline.extract_features(encoder, image),
            'h2': self.h2.extract_features(encoder, image),
            'h3': self.h3.extract_features(encoder, image),
        }
    
    def extract_all(self, encoder, image: Image.Image) -> np.ndarray:
        feats = self.extract_features(encoder, image)
        return np.concatenate([feats['baseline'], feats['h2'], feats['h3']])
    
    def get_feature_slices(self) -> Dict[str, Tuple[int, int]]:
        """Returns (start, end) indices for each block."""
        return {
            'baseline': (0, 63),
            'h2': (63, 78),
            'h3': (78, 98),
        }
