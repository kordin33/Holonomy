"""
h3_v7.py - H3 V7: 2-Head with Residual Field (CLEAN + MAX)

CLEAN (12D) - residual field relative to global (dla ablacji):
- worst_res, mad_res
- pc1_energy_res, entropy_spec_res
- sync_res_mean, sync_res_min
- flip_rate

MAX (18D) - CLEAN + global roughness (dla standalone)
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


def chordal_dist(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.clip(np.dot(a.astype(np.float64), b.astype(np.float64)), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))


SEQ_PARAMS = {
    'J': [95, 90, 85, 80, 75],
    'B': [0.2, 0.35, 0.5, 0.65, 0.8],
}


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    return [image.crop((x, y, x+ps, y+ps)).resize((224, 224), Image.LANCZOS) for x, y in positions]


def compute_step_vector(encoder, image: Image.Image, seq_name: str) -> np.ndarray:
    """Compute step distances for sequence."""
    params = SEQ_PARAMS[seq_name]
    imgs = [image]
    curr = image
    for p in params:
        if seq_name == 'J':
            curr = jpeg_compression(curr, p)
        else:
            curr = gaussian_blur(curr, p)
        imgs.append(curr)
    
    embs = encoder.encode_batch(imgs, batch_size=8, show_progress=False)
    embs = l2_normalize(np.asarray(embs, dtype=np.float32))
    
    steps = [chordal_dist(embs[i], embs[i+1]) for i in range(len(embs)-1)]
    return np.array(steps, dtype=np.float64)


def compute_roughness(v: np.ndarray) -> np.ndarray:
    if len(v) < 3:
        return np.zeros(3, dtype=np.float32)
    diffs = np.diff(v)
    allan = np.mean(diffs ** 2)
    S2_1 = np.mean((v[1:] - v[:-1]) ** 2)
    S2_2 = np.mean((v[2:] - v[:-2]) ** 2)
    rough_R = S2_2 / (S2_1 + 1e-10)
    jitter_std = np.std(diffs)
    return np.array([rough_R, allan, jitter_std], dtype=np.float32)


def compute_h3_features(encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute H3 features with 2-head architecture.
    Returns: (CLEAN, MAX)
    """
    patches = get_5_patches(image)
    
    all_clean = []
    all_amplitude = []
    
    for seq_name in ['J', 'B']:
        # Global step vector
        v_global = compute_step_vector(encoder, image, seq_name)
        
        # Patch step vectors
        v_patches = [compute_step_vector(encoder, p, seq_name) for p in patches]
        v_patches = np.array(v_patches)  # (5, 5)
        
        # RESIDUAL FIELD: r_patch = v_patch - v_global
        residuals = v_patches - v_global  # (5, 5)
        
        # CLEAN features from residuals
        res_norms = np.linalg.norm(residuals, axis=1)  # (5,)
        worst_res = np.max(res_norms)
        mad_res = np.median(np.abs(res_norms - np.median(res_norms)))
        
        # PCA on residuals
        try:
            cov = np.cov(residuals.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            pc1_energy = eigvals[0] / (eigvals.sum() + 1e-10)
            p = eigvals / (eigvals.sum() + 1e-10)
            entropy_spec = -np.sum(p * np.log(p + 1e-10))
        except:
            pc1_energy, entropy_spec = 0, 0
        
        # Sync on residuals
        corr = np.corrcoef(residuals)
        triu = corr[np.triu_indices(5, k=1)]
        triu = triu[~np.isnan(triu)]
        sync_res_mean = np.mean(triu) if len(triu) > 0 else 0
        sync_res_min = np.min(triu) if len(triu) > 0 else 0
        
        # Flip rate: % patches with corr(v_patch, v_global) < 0
        flip_count = 0
        for v_p in v_patches:
            if np.corrcoef(v_p, v_global)[0, 1] < 0:
                flip_count += 1
        flip_rate = flip_count / 5
        
        all_clean.extend([
            worst_res, mad_res, pc1_energy, entropy_spec,
            sync_res_mean, sync_res_min, flip_rate
        ])
        
        # Amplitude features (global roughness)
        roughness = compute_roughness(v_global)
        all_amplitude.extend(roughness.tolist())
    
    CLEAN = np.array(all_clean, dtype=np.float32)  # 14D
    MAX = np.concatenate([CLEAN, np.array(all_amplitude, dtype=np.float32)])  # 20D
    
    return CLEAN, MAX


class H3_V7:
    """H3 V7: 2-Head (CLEAN 14D, MAX 20D)"""
    
    def extract_clean(self, encoder, image: Image.Image) -> np.ndarray:
        clean, _ = compute_h3_features(encoder, image)
        return clean
    
    def extract_max(self, encoder, image: Image.Image) -> np.ndarray:
        _, max_feats = compute_h3_features(encoder, image)
        return max_feats
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        return self.extract_max(encoder, image)
