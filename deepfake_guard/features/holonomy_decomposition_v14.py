"""
holonomy_decomposition_v14.py - FIXED GRID + SPECTRAL + WEIGHTED

Poprawki vs V13:
1. Stała siatka 3×3 (bez permutacji patchy)
2. Square crop z każdej komórki (bez anisotropic warp)
3. Disagreement jako macierz podobieństw P @ P.T
4. Loop Spectral Signature (eigenvalues)
5. Weighted aggregations po energy

Cechy: ~120D
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple
import io


# ============================================================================
# UTILS
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
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a64, b64) / (na * nb), -1.0, 1.0)))


# ============================================================================
# BASELINE LOOPS (same as V12)
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


def get_fixed_grid_patches(image: Image.Image) -> Tuple[List[Image.Image], np.ndarray]:
    """
    Fixed 3×3 grid with SQUARE CROP from each cell.
    Returns patches and their energy weights.
    """
    w, h = image.size
    cw, ch = w // 3, h // 3
    
    patches = []
    energies = []
    
    for row in range(3):
        for col in range(3):
            # Cell bounds
            x0, y0 = col * cw, row * ch
            x1, y1 = x0 + cw, y0 + ch
            
            # Square crop (take smaller dimension)
            cell_w, cell_h = x1 - x0, y1 - y0
            sq_size = min(cell_w, cell_h)
            cx, cy = x0 + cell_w // 2, y0 + cell_h // 2
            sq_x0, sq_y0 = cx - sq_size // 2, cy - sq_size // 2
            sq_x1, sq_y1 = sq_x0 + sq_size, sq_y0 + sq_size
            
            patch = image.crop((sq_x0, sq_y0, sq_x1, sq_y1))
            patch = patch.resize((224, 224), Image.LANCZOS)
            patches.append(patch)
            
            # Energy = Laplacian variance (edge detector)
            gray = np.array(patch.convert('L'), dtype=np.float32)
            lap = np.abs(gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
            energy = np.var(lap)
            energies.append(energy)
    
    energies = np.array(energies, dtype=np.float32)
    return patches, energies


def compute_baseline_features(encoder, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute baseline features + spectral signature per loop.
    Returns: (loop_features (81D), spectral_features (36D))
    """
    loop_feats = []
    spectral_feats = []
    
    for loop in BASELINE_LOOPS:
        imgs = [image]
        curr = image
        for name in loop:
            curr = TRANSFORMS[name](curr)
            imgs.append(curr)
        
        # Batch encode
        emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        emb = l2_normalize(np.asarray(emb, dtype=np.float32))
        
        # Step distances
        steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
        H = chordal_dist(emb[0], emb[-1])
        L = sum(steps)
        
        # Curvature  
        D = emb[1:] - emb[:-1]
        angles = [cosine_angle(D[i], D[i+1]) for i in range(len(D)-1)] if len(D) >= 2 else []
        curv_mean = np.mean(angles) if angles else 0.0
        curv_max = np.max(angles) if angles else 0.0
        
        loop_feats.extend([
            H, L, L/(H+1e-8),
            np.std(steps) if len(steps) > 1 else 0.0,
            np.mean(steps), np.max(steps),
            np.median(steps),
            curv_mean,
            curv_max
        ])  # 9 per loop
        
        # SPECTRAL SIGNATURE: eigenvalues of similarity matrix
        M = np.dot(emb, emb.T)  # (n_steps+1, n_steps+1)
        try:
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)[::-1]
            
            # Features from eigenvalues
            spec_entropy = -np.sum((eigvals / (eigvals.sum() + 1e-10)) * np.log(eigvals / (eigvals.sum() + 1e-10) + 1e-10))
            eff_rank = np.exp(spec_entropy)
            condition = eigvals[0] / (eigvals[-1] + 1e-10)
            off_diag = M[np.triu_indices(M.shape[0], k=1)]
            
            spectral_feats.extend([spec_entropy, eff_rank, condition, np.mean(off_diag)])
        except:
            spectral_feats.extend([0, 1, 1, 0])
    
    return np.array(loop_feats, dtype=np.float32), np.array(spectral_feats, dtype=np.float32)


def compute_disagreement_matrix(patch_feats: np.ndarray) -> np.ndarray:
    """
    Permutation-invariant disagreement from similarity matrix.
    patch_feats: (9, D)
    Returns: 6D features
    """
    P = l2_normalize(patch_feats)  # (9, D)
    S = np.dot(P, P.T)  # (9, 9) similarity matrix
    
    off_diag = S[np.triu_indices(9, k=1)]
    
    return np.array([
        np.mean(off_diag),
        np.min(off_diag),
        np.std(off_diag),
        np.max(off_diag) - np.min(off_diag),
        np.percentile(off_diag, 10),
        np.percentile(off_diag, 90),
    ], dtype=np.float32)


class HolonomyDecompositionV14:
    """
    V14: Fixed Grid + Spectral + Weighted.
    
    Features:
    - Global baseline: 81D
    - Global spectral: 36D
    - Weighted patch mean: 81D
    - Weighted patch range: 81D
    - Disagreement matrix stats: 6D
    
    Total: ~285D
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Global features
        global_loop, global_spectral = compute_baseline_features(encoder, image)
        
        # 2. Fixed grid patches with energy weights
        patches, energies = get_fixed_grid_patches(image)
        
        # Softmax weights (temperature α=1)
        weights = np.exp(energies - energies.max())
        weights = weights / (weights.sum() + 1e-10)
        
        # 3. Patch features (all 9)
        patch_feats = []
        for p in patches:
            pf, _ = compute_baseline_features(encoder, p)
            patch_feats.append(pf)
        patch_feats = np.array(patch_feats)  # (9, 81)
        
        # 4. Weighted aggregations
        weighted_mean = np.sum(patch_feats * weights[:, None], axis=0)  # (81,)
        
        # Range: max - min per feature
        patch_range = np.max(patch_feats, axis=0) - np.min(patch_feats, axis=0)  # (81,)
        
        # 5. Disagreement from similarity matrix
        disagreement = compute_disagreement_matrix(patch_feats)  # 6D
        
        # Combine
        return np.concatenate([
            global_loop,      # 81D
            global_spectral,  # 36D
            weighted_mean,    # 81D
            patch_range,      # 81D
            disagreement      # 6D
        ]).astype(np.float32)  # Total: 285D
