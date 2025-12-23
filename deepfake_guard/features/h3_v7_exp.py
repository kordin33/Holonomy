"""
h3_v7_exp.py - H3 V7 EXP: Richardson-style Multi-Scale

Multi-scale coherence: mid vs high intensity sequences.
Richardson-like extrapolation on coherence metrics.

CLEAN (16D), MAX (24D)
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


# Two scales
SEQ_MID = {'J': [95, 92, 89], 'B': [0.2, 0.3, 0.4]}
SEQ_HIGH = {'J': [85, 75, 65], 'B': [0.5, 0.75, 1.0]}


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    return [image.crop((x, y, x+ps, y+ps)).resize((224, 224), Image.LANCZOS) for x, y in positions]


def compute_step_vector_scale(encoder, image: Image.Image, seq_name: str, scale: str) -> np.ndarray:
    """Compute step distances for sequence at given scale."""
    params = SEQ_MID[seq_name] if scale == 'mid' else SEQ_HIGH[seq_name]
    
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


def compute_scale_coherence(encoder, image: Image.Image, seq_name: str, scale: str) -> dict:
    """Compute coherence metrics for one scale."""
    patches = get_5_patches(image)
    
    v_global = compute_step_vector_scale(encoder, image, seq_name, scale)
    v_patches = [compute_step_vector_scale(encoder, p, seq_name, scale) for p in patches]
    v_patches = np.array(v_patches)
    
    # Residuals
    residuals = v_patches - v_global
    res_norms = np.linalg.norm(residuals, axis=1)
    
    # Coherence
    corr_matrix = np.corrcoef(v_patches)
    triu = corr_matrix[np.triu_indices(5, k=1)]
    triu = triu[~np.isnan(triu)]
    
    sync_mean = np.mean(triu) if len(triu) > 0 else 0
    sync_min = np.min(triu) if len(triu) > 0 else 0
    worst_res = np.max(res_norms)
    
    # Global roughness
    diffs = np.diff(v_global)
    allan = np.mean(diffs ** 2) if len(diffs) > 0 else 0
    
    return {
        'sync_mean': sync_mean,
        'sync_min': sync_min,
        'worst_res': worst_res,
        'mad_res': np.median(np.abs(res_norms - np.median(res_norms))),
        'allan': allan,
        'step_mean': np.mean(v_global),
        'step_std': np.std(v_global),
    }


def compute_h3_exp_features(encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute H3 V7 EXP features with Richardson-style multi-scale.
    Returns: (CLEAN, MAX)
    """
    all_clean = []
    all_amplitude = []
    
    for seq_name in ['J', 'B']:
        mid = compute_scale_coherence(encoder, image, seq_name, 'mid')
        high = compute_scale_coherence(encoder, image, seq_name, 'high')
        
        # Richardson-like extrapolation
        sync_richardson = (4 * high['sync_mean'] - mid['sync_mean']) / 3
        
        # Scale behavior
        sync_diff = high['sync_mean'] - mid['sync_mean']
        sync_ratio = high['sync_mean'] / (mid['sync_mean'] + 1e-8)
        worst_diff = high['worst_res'] - mid['worst_res']
        
        # CLEAN (shape-like)
        all_clean.extend([
            sync_richardson,
            sync_diff,
            sync_ratio,
            worst_diff,
            high['sync_min'] - mid['sync_min'],
            high['mad_res'] / (mid['mad_res'] + 1e-8),
            abs(high['sync_mean'] - mid['sync_mean']) / (mid['sync_mean'] + 1e-8),  # consistency
            high['worst_res'] / (mid['worst_res'] + 1e-8),
        ])
        
        # Amplitude
        all_amplitude.extend([
            mid['allan'], high['allan'],
            mid['step_mean'], high['step_mean']
        ])
    
    CLEAN = np.array(all_clean, dtype=np.float32)  # 16D
    MAX = np.concatenate([CLEAN, np.array(all_amplitude, dtype=np.float32)])  # 24D
    
    return CLEAN, MAX


class H3_V7_EXP:
    """H3 V7 EXP: Richardson Multi-Scale. CLEAN 16D, MAX 24D."""
    
    def extract_clean(self, encoder, image: Image.Image) -> np.ndarray:
        clean, _ = compute_h3_exp_features(encoder, image)
        return clean
    
    def extract_max(self, encoder, image: Image.Image) -> np.ndarray:
        _, max_feats = compute_h3_exp_features(encoder, image)
        return max_feats
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        return self.extract_max(encoder, image)
