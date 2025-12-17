"""
optimized_v4.py - H1/H2/H3 V4 (Full Optimization)

KLUCZOWE ZMIANY:
1. Log-map na sferze zamiast prostej różnicy wektorów
2. Patch-H1 (5 patchy) + statystyki "najgorszego patcha"
3. H2 2D fit: log(HJB) ~ a*log(sJ) + b*log(sB) + c
4. Commutator jako krzywa (nie jedna liczba)
5. Dwie wersje: MAX (standalone) i CLEAN (ablacje)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
from scipy.stats import linregress, spearmanr
from scipy.linalg import lstsq
import io


# ============================================================================
# GEOMETRY HELPERS (LOG-MAP NA SFERZE!)
# ============================================================================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)


def chordal_dist(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Chordal distance on sphere."""
    dot = np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0)
    return np.float64(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))


def geodesic_dist(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Geodesic (arc) distance on unit sphere."""
    dot = np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0)
    return np.float64(np.arccos(dot))


def log_map(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
    """
    Log-map: Project z1 onto tangent space at z0.
    Returns vector u in tangent space at z0, with ||u|| = geodesic distance.
    """
    z0 = z0.astype(np.float64)
    z1 = z1.astype(np.float64)
    
    cos_theta = np.clip(np.dot(z0, z1), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if theta < 1e-10:
        return np.zeros_like(z0)
    
    # Project z1 onto tangent plane at z0
    u = z1 - cos_theta * z0
    u_norm = np.linalg.norm(u)
    
    if u_norm < 1e-10:
        return np.zeros_like(z0)
    
    # Scale by geodesic distance
    u = u / u_norm * theta
    return u


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


# ============================================================================
# H1 V4 (Holonomy Spectrum with Log-Map + Patches)
# ============================================================================

H1_LOOPS = [
    ('jpeg_80', lambda img: jpeg_compression(img, 80)),
    ('jpeg_60', lambda img: jpeg_compression(img, 60)),
    ('blur_0.5', lambda img: gaussian_blur(img, 0.5)),
    ('blur_1.0', lambda img: gaussian_blur(img, 1.0)),
    ('scale_0.75', lambda img: downscale_upscale(img, 0.75)),
    ('sharpen_2.0', lambda img: sharpen(img, 2.0)),
]


class H1_V4:
    """
    H1 with Log-Map geometry + Patch analysis.
    Features: energy, spectral stats, patch aggregations.
    """
    
    def get_patches(self, image: Image.Image) -> List[Image.Image]:
        """4 corners + center."""
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
            patches.append(patch.resize((224, 224), Image.LANCZOS))
        return patches
    
    def compute_holonomy_vectors(self, encoder, image: Image.Image) -> np.ndarray:
        """Compute log-map holonomy vectors for all loops."""
        imgs = [image] + [fn(image) for _, fn in H1_LOOPS]
        embs = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        embs = l2_normalize(np.asarray(embs))
        
        z0 = embs[0]
        vectors = []
        for i in range(1, len(embs)):
            u = log_map(z0, embs[i])
            vectors.append(u)
        
        return np.array(vectors)  # (num_loops, dim)
    
    def spectral_features(self, vectors: np.ndarray) -> np.ndarray:
        """Compute spectral features from holonomy vectors."""
        # Energy per loop
        energies = np.linalg.norm(vectors, axis=1)
        
        # Gram matrix (loop x loop)
        G = vectors @ vectors.T
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1]
        
        # Normalize to probability
        eig_sum = np.sum(eigvals) + 1e-10
        p = eigvals / eig_sum
        
        # Stats
        entropy = -np.sum(p * np.log(p + 1e-10))
        gini = 1.0 - np.sum(p ** 2)
        head_tail = (eigvals[0] + 1e-10) / (eigvals[-1] + 1e-10)
        
        E_head = np.sum(eigvals[:2])
        E_tail = np.sum(eigvals[-2:])
        
        return np.array([
            np.mean(energies), np.std(energies), np.max(energies),
            entropy, gini, 
            E_head / (eig_sum + 1e-10), E_tail / (eig_sum + 1e-10),
            head_tail,
        ], dtype=np.float32)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # Global features
        global_vecs = self.compute_holonomy_vectors(encoder, image)
        global_feats = self.spectral_features(global_vecs)
        
        # Patch features
        patches = self.get_patches(image)
        patch_feats_list = []
        
        for patch in patches:
            vecs = self.compute_holonomy_vectors(encoder, patch)
            feats = self.spectral_features(vecs)
            patch_feats_list.append(feats)
        
        patch_feats = np.array(patch_feats_list)  # (5, num_feats)
        
        # Aggregations
        agg_min = np.min(patch_feats, axis=0)
        agg_max = np.max(patch_feats, axis=0)
        agg_std = np.std(patch_feats, axis=0)
        agg_range = agg_max - agg_min
        
        # Global vs max patch gap (for key features: energy_mean, entropy)
        gap_energy = agg_max[0] - global_feats[0]
        gap_entropy = agg_max[3] - global_feats[3]
        
        return np.concatenate([
            global_feats,           # 8D
            agg_min, agg_max,       # 16D
            agg_std, agg_range,     # 16D
            [gap_energy, gap_entropy],  # 2D
        ]).astype(np.float32)


# ============================================================================
# H2 V4 (Scale Law with 2D Fit + Commutator Curve)
# ============================================================================

class H2_V4:
    """
    H2 with 2D fit (JPEG + BLUR separation) and commutator curve.
    """
    
    def __init__(self):
        # Intensities: (jpeg_q, blur_sigma)
        self.intensities = [
            (95, 0.2), (90, 0.3), (85, 0.4), (80, 0.5), (75, 0.6),
            (70, 0.7), (65, 0.8), (60, 0.9), (55, 1.0), (50, 1.1),
        ]
        self.split_idx = 5  # low vs high
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        z0 = l2_normalize(encoder.encode_batch([image], batch_size=1, show_progress=False)[0])
        
        # Collect measurements
        sJ_list, sB_list, HJB_list, comm_list = [], [], [], []
        
        for jpeg_q, blur_s in self.intensities:
            # Separate degradations
            img_J = jpeg_compression(image, jpeg_q)
            img_B = gaussian_blur(image, blur_s)
            img_JB = gaussian_blur(img_J, blur_s)
            img_BJ = jpeg_compression(img_B, jpeg_q)
            
            embs = encoder.encode_batch([img_J, img_B, img_JB, img_BJ], 
                                        batch_size=4, show_progress=False)
            zJ, zB, zJB, zBJ = [l2_normalize(e) for e in embs]
            
            sJ = geodesic_dist(z0, zJ)
            sB = geodesic_dist(z0, zB)
            HJB = geodesic_dist(z0, zJB)
            comm = geodesic_dist(zJB, zBJ)
            
            sJ_list.append(sJ)
            sB_list.append(sB)
            HJB_list.append(HJB)
            comm_list.append(comm)
        
        sJ = np.array(sJ_list, dtype=np.float64)
        sB = np.array(sB_list, dtype=np.float64)
        HJB = np.array(HJB_list, dtype=np.float64)
        comm = np.array(comm_list, dtype=np.float64)
        
        # Validity
        valid = (sJ > 1e-5) & (sB > 1e-5) & (HJB > 1e-5)
        
        if valid.sum() < 6:
            return np.zeros(25, dtype=np.float32)
        
        # 2D Fit: log(HJB) ~ a*log(sJ) + b*log(sB) + c
        X = np.column_stack([np.log(sJ[valid]), np.log(sB[valid]), np.ones(valid.sum())])
        y = np.log(HJB[valid])
        
        try:
            coeffs, residuals, rank, s = lstsq(X, y)
            a, b, c = coeffs
            y_pred = X @ coeffs
            r2 = 1 - np.sum((y - y_pred)**2) / (np.sum((y - np.mean(y))**2) + 1e-10)
            res_std = np.std(y - y_pred)
        except:
            a, b, c, r2, res_std = 0, 0, 0, 0, 1
        
        anisotropy = a - b
        
        # Low vs High splits
        low = np.zeros(len(sJ), dtype=bool)
        low[:self.split_idx] = True
        low = low & valid
        high = ~np.zeros(len(sJ), dtype=bool)
        high[:self.split_idx] = False
        high = high & valid
        
        # Rho and Slack
        L = sJ + sB
        rho = HJB / (L + 1e-10)
        slack_norm = (L - HJB) / (L + 1e-10)
        
        rho_low = np.mean(rho[low]) if low.sum() > 0 else 0
        rho_high = np.mean(rho[high]) if high.sum() > 0 else 0
        slack_low = np.mean(slack_norm[low]) if low.sum() > 0 else 0
        slack_high = np.mean(slack_norm[high]) if high.sum() > 0 else 0
        
        # Commutator curve
        comm_norm = comm / (sJ + sB + 1e-10)
        comm_mean = np.mean(comm_norm[valid])
        comm_max = np.max(comm_norm[valid])
        comm_std = np.std(comm_norm[valid])
        
        # Spearman correlation: comm vs intensity
        indices = np.arange(len(comm))
        try:
            spearman_comm, _ = spearmanr(indices[valid], comm_norm[valid])
        except:
            spearman_comm = 0
        
        # Local slopes in log-log
        log_H = np.log(HJB[valid] + 1e-10)
        log_s = np.log(sJ[valid] + sB[valid] + 1e-10)
        local_slopes = np.diff(log_H) / (np.diff(log_s) + 1e-10)
        
        mean_slope = np.mean(local_slopes) if len(local_slopes) > 0 else 0
        std_slope = np.std(local_slopes) if len(local_slopes) > 0 else 0
        max_jump = np.max(np.abs(np.diff(local_slopes))) if len(local_slopes) > 1 else 0
        
        return np.array([
            a, b, anisotropy, c, r2, res_std,
            rho_low, rho_high, rho_high - rho_low,
            slack_low, slack_high, slack_high - slack_low,
            comm_mean, comm_max, comm_std, float(spearman_comm),
            mean_slope, std_slope, max_jump,
            float(np.mean(sJ[valid])), float(np.mean(sB[valid])),
            float(np.mean(HJB[valid])), float(np.mean(comm[valid])),
            float(np.std(sJ[valid])), float(np.std(sB[valid])),
        ], dtype=np.float32)


# ============================================================================
# H3 V4 (Dispersion with Patch Sync + Clean Features)
# ============================================================================

H3_SEQUENCE = [
    lambda img: jpeg_compression(img, 85),
    lambda img: gaussian_blur(img, 0.4),
    lambda img: jpeg_compression(img, 70),
    lambda img: gaussian_blur(img, 0.6),
]


class H3_V4:
    """
    H3 with Log-Map dispersion + Patch synchrony.
    """
    
    def get_patches(self, image: Image.Image) -> List[Image.Image]:
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
            patches.append(patch.resize((224, 224), Image.LANCZOS))
        return patches
    
    def compute_trajectory(self, encoder, image: Image.Image) -> np.ndarray:
        """Compute log-map trajectory."""
        imgs = [image]
        curr = image
        for fn in H3_SEQUENCE:
            curr = fn(curr)
            imgs.append(curr)
        
        embs = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        embs = l2_normalize(np.asarray(embs))
        
        z0 = embs[0]
        trajectory = [log_map(z0, embs[i]) for i in range(1, len(embs))]
        return np.array(trajectory)  # (steps, dim)
    
    def trajectory_features(self, traj: np.ndarray) -> np.ndarray:
        """Features from one trajectory."""
        steps = np.linalg.norm(traj, axis=1)
        
        # Pairwise distances (geodesic in tangent space ~ Euclidean)
        K = len(traj)
        pairwise = []
        for i in range(K):
            for j in range(i+1, K):
                pairwise.append(np.linalg.norm(traj[i] - traj[j]))
        
        D_pairwise = np.mean(pairwise) if pairwise else 0
        
        # Total displacement
        H_end = steps[-1] if len(steps) > 0 else 0
        L_path = np.sum(steps)
        rho = H_end / (L_path + 1e-10)
        
        return np.array([D_pairwise, H_end, L_path, rho], dtype=np.float64)
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # Global trajectory
        global_traj = self.compute_trajectory(encoder, image)
        global_feats = self.trajectory_features(global_traj)
        
        # Patch trajectories
        patches = self.get_patches(image)
        patch_feats_list = []
        patch_step_vectors = []
        
        for patch in patches:
            traj = self.compute_trajectory(encoder, patch)
            feats = self.trajectory_features(traj)
            patch_feats_list.append(feats)
            patch_step_vectors.append(np.linalg.norm(traj, axis=1))
        
        patch_feats = np.array(patch_feats_list)  # (5, 4)
        
        # Aggregations
        agg_min = np.min(patch_feats, axis=0)
        agg_max = np.max(patch_feats, axis=0)
        agg_std = np.std(patch_feats, axis=0)
        agg_range = agg_max - agg_min
        
        # Global vs max gap
        gap = agg_max - global_feats
        
        # Patch synchrony (correlation of step vectors)
        step_matrix = np.array(patch_step_vectors)  # (5, steps)
        
        if step_matrix.shape[1] >= 2:
            corr_matrix = np.corrcoef(step_matrix)
            offdiag = corr_matrix[np.triu_indices(5, k=1)]
            offdiag = offdiag[~np.isnan(offdiag)]
            
            if len(offdiag) > 0:
                sync_mean = np.mean(offdiag)
                sync_std = np.std(offdiag)
                sync_min = np.min(offdiag)
            else:
                sync_mean, sync_std, sync_min = 0, 0, 0
        else:
            sync_mean, sync_std, sync_min = 0, 0, 0
        
        return np.concatenate([
            global_feats,              # 4D
            agg_min, agg_max,          # 8D
            agg_std, agg_range,        # 8D
            gap,                       # 4D
            [sync_mean, sync_std, sync_min],  # 3D
        ]).astype(np.float32)


# ============================================================================
# COMBINED V4
# ============================================================================

class CombinedV4:
    def __init__(self):
        self.h1 = H1_V4()
        self.h2 = H2_V4()
        self.h3 = H3_V4()
    
    def extract_features(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        return {
            'h1': self.h1.extract_features(encoder, image),
            'h2': self.h2.extract_features(encoder, image),
            'h3': self.h3.extract_features(encoder, image),
        }
    
    def extract_all(self, encoder, image: Image.Image) -> np.ndarray:
        feats = self.extract_features(encoder, image)
        return np.concatenate([feats['h1'], feats['h2'], feats['h3']])
