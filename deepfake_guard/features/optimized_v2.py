"""
optimized_v2.py - NAPRAWIONE H2, H3, Baseline według szczegółowego planu

ETAP 1: Metryka Chordal wszędzie (sqrt(2-2*dot))
ETAP 2: Baseline 9 loops z production, curvature z wektorów Δ
ETAP 3: H2 - s = d(z_mid, z0), H = d(z_end, z0), OLS fit
ETAP 4: H3 - Patche obowiązkowo, pairwise mean, długa sekwencja
ETAP 5: Per-block scaling
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict
from scipy.stats import linregress
import io


# ============================================================================
# ETAP 1: METRYKA CHORDAL (sqrt(2 - 2*dot))
# ============================================================================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize vector or array of vectors."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)


def chordal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Chordal distance: d(a,b) = sqrt(2 - 2*<a,b>)
    
    Właściwości:
    - Liniowa dla małych kątów (θ → d ≈ θ)
    - Spełnia nierówność trójkąta
    - Stabilna numerycznie
    """
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))


def cosine_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Kąt między wektorami (dla curvature)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    dot = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(dot))


# ============================================================================
# DEGRADACJE (z Production)
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

def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(factor)

def adjust_gamma(image: Image.Image, gamma: float = 1.5) -> Image.Image:
    arr = np.array(image).astype(np.float32) / 255.0
    return Image.fromarray((np.power(arr, gamma) * 255).astype(np.uint8))

def adjust_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)

def identity(image: Image.Image) -> Image.Image:
    return image.copy()


# Registry (jak w Production)
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
    'gamma_1.5': lambda img: adjust_gamma(img, 1.5),
    'contrast_1.3': lambda img: adjust_contrast(img, 1.3),
    'contrast_1.5': lambda img: adjust_contrast(img, 1.5),
    'identity': identity,
}


# ============================================================================
# ETAP 2: BASELINE (9 loops z Production, curvature z wektorów Δ)
# ============================================================================

# 9 pętli z Production (pełne pokrycie)
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


class BaselineV2:
    """
    Baseline z 9 loops, metryka Chordal, curvature z wektorów Δ.
    
    Cechy per loop:
    - H_raw (chordal distance start->end)
    - path_length (suma chordal distances)
    - tortuosity (path_length / H_raw)
    - std_step (odchylenie standardowe kroków)
    - curvature (średni kąt między kolejnymi Δ_i)
    """
    
    def compute_loop_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Cechy z jednej trajektorii."""
        E = l2_normalize(embeddings)
        
        # Wektory kroków Δ_i = z_{i+1} - z_i (w przestrzeni embeddingów)
        D = E[1:] - E[:-1]
        
        # Step norms w chordal distance
        step_dists = np.array([chordal_distance(E[i], E[i+1]) for i in range(len(E)-1)])
        
        # H_raw (chordal)
        H_raw = chordal_distance(E[0], E[-1])
        
        # Path length
        path_length = float(step_dists.sum())
        
        # Tortuosity
        tortuosity = path_length / (H_raw + 1e-8)
        
        # Std step
        std_step = float(step_dists.std()) if len(step_dists) > 1 else 0.0
        
        # Curvature: średni kąt między kolejnymi wektorami Δ_i
        curvature = 0.0
        if len(D) >= 2:
            angles = []
            for i in range(len(D) - 1):
                angle = cosine_angle(D[i], D[i+1])
                angles.append(angle)
            curvature = float(np.mean(angles))
        
        return np.array([H_raw, path_length, tortuosity, std_step, curvature], dtype=np.float32)
    
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
# ETAP 3: H2 SCALE LAW (s = d(z_mid, z0), H = d(z_end, z0), OLS)
# ============================================================================

class H2_V2:
    """
    H2 Scale Law z naprawioną logiką:
    - s = d(z_mid, z0) - siła degradacji zmierzona jako przemieszczenie po JPEG
    - H = d(z_end, z0) - holonomia po JPEG+blur
    - OLS fit (stabilniejszy na 11 punktach niż Theil-Sen)
    - Metryka Chordal
    """
    
    def __init__(self):
        self.intensities = [
            (95, 0.2), (90, 0.3), (85, 0.4), (80, 0.5), (75, 0.6),
            (70, 0.7), (65, 0.8), (60, 0.9), (55, 1.0), (50, 1.1), (45, 1.2),
        ]
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        z_0 = encoder.encode_batch([image], batch_size=1, show_progress=False)[0]
        z_0 = l2_normalize(z_0)
        
        Hs = []
        Ss = []
        
        for jpeg_q, blur_sigma in self.intensities:
            # z_mid (po JPEG)
            img_mid = jpeg_compression(image, jpeg_q)
            z_mid = encoder.encode_batch([img_mid], batch_size=1, show_progress=False)[0]
            z_mid = l2_normalize(z_mid)
            
            # z_end (po JPEG + blur)
            img_end = gaussian_blur(img_mid, blur_sigma)
            z_end = encoder.encode_batch([img_end], batch_size=1, show_progress=False)[0]
            z_end = l2_normalize(z_end)
            
            # H = d(z_end, z0), s = d(z_mid, z0)
            H = chordal_distance(z_end, z_0)
            s = chordal_distance(z_mid, z_0)
            
            Hs.append(H)
            Ss.append(s)
        
        Hs = np.array(Hs)
        Ss = np.array(Ss)
        
        # Stabilność: filtr valid
        valid = (Hs > 1e-6) & (Ss > 1e-6)
        if valid.sum() < 4:
            return np.zeros(5, dtype=np.float32)
        
        log_H = np.log(Hs[valid])
        log_s = np.log(Ss[valid])
        
        # OLS fit (stabilny na 11 punktach)
        result = linregress(log_s, log_H)
        alpha = float(result.slope)
        r_squared = float(result.rvalue ** 2)
        
        # Residuals
        predicted = alpha * log_s + result.intercept
        residual_std = float((log_H - predicted).std())
        
        # Slope stability
        slope_stability = 1.0 / (result.stderr + 0.01) if result.stderr is not None else 1.0
        
        return np.array([
            alpha,
            r_squared,
            residual_std,
            float(Hs.mean()),
            slope_stability,
        ], dtype=np.float32)


# ============================================================================
# ETAP 4: H3 DISPERSION (Patche!, pairwise mean, długa sekwencja)
# ============================================================================

# Dłuższa sekwencja degradacji (6-8 punktów)
H3_SEQUENCE = [
    lambda img: jpeg_compression(img, 90),
    lambda img: gaussian_blur(img, 0.3),
    lambda img: jpeg_compression(img, 75),
    lambda img: gaussian_blur(img, 0.5),
    lambda img: downscale_upscale(img, 0.9),
    lambda img: jpeg_compression(img, 60),
    lambda img: gaussian_blur(img, 0.7),
]


class H3_V2:
    """
    H3 Dispersion z naprawionymi cechami:
    - Patche obowiązkowo (4 rogi + center)
    - Pairwise mean distance (robustniejsze niż Gram eigenvalues dla K=8)
    - Dłuższa sekwencja (7 kroków)
    - trace(var) surowe + znormalizowane z clampem
    """
    
    def compute_dispersion(self, embeddings: np.ndarray) -> np.ndarray:
        """Metryki dyspersji z trajektorii."""
        E = l2_normalize(embeddings)
        K = len(E)
        
        # (1) Pairwise mean distance (chordal)
        dists = []
        for i in range(K):
            for j in range(i+1, K):
                dists.append(chordal_distance(E[i], E[j]))
        D_pairwise = float(np.mean(dists)) if dists else 0.0
        
        # (2) Closure / path length
        closure = chordal_distance(E[0], E[-1])
        path_length = sum(chordal_distance(E[i], E[i+1]) for i in range(K-1))
        D_path = closure / (path_length + 1e-8)
        
        # (3) trace(var) surowe
        var = np.var(E, axis=0, ddof=1)
        trace_var = float(var.sum())
        
        # (4) trace(var) / ||mean||² (z clampem)
        mean_norm_sq = float(np.linalg.norm(E.mean(axis=0)) ** 2)
        D_cov_norm = trace_var / max(mean_norm_sq, 1e-6)
        D_cov_norm = min(D_cov_norm, 100.0)  # Clamp
        
        # (5) Step stats
        step_dists = [chordal_distance(E[i], E[i+1]) for i in range(K-1)]
        step_mean = float(np.mean(step_dists)) if step_dists else 0.0
        step_std = float(np.std(step_dists)) if len(step_dists) > 1 else 0.0
        
        return np.array([
            D_pairwise,
            D_path,
            trace_var,
            D_cov_norm,
            step_mean,
            step_std,
        ], dtype=np.float32)
    
    def get_grid_patches(self, image: Image.Image) -> List[Image.Image]:
        """Grid patchy: 4 rogi + center, resize 224."""
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
        # Global features
        imgs = [image]
        curr = image
        for fn in H3_SEQUENCE:
            curr = fn(curr)
            imgs.append(curr)
        
        emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        global_feat = self.compute_dispersion(np.asarray(emb, dtype=np.float32))
        
        # Patch features
        patches = self.get_grid_patches(image)
        patch_feats = []
        
        for patch in patches:
            imgs = [patch]
            curr = patch
            for fn in H3_SEQUENCE[:4]:  # Krótsza sekwencja dla patchy (szybciej)
                curr = fn(curr)
                imgs.append(curr)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            patch_feats.append(self.compute_dispersion(np.asarray(emb, dtype=np.float32)))
        
        patch_feats = np.array(patch_feats)
        patch_median = np.median(patch_feats, axis=0)
        patch_p80 = np.percentile(patch_feats, 80, axis=0)
        patch_disagreement = patch_p80 - patch_median  # Patch disagreement!
        
        return np.concatenate([
            global_feat,
            patch_median,
            patch_p80,
            patch_disagreement,
        ]).astype(np.float32)


# ============================================================================
# ETAP 5: COMBINED (per-block scaling)
# ============================================================================

class CombinedV2:
    """Łączy Baseline + H2 + H3 z per-block scaling."""
    
    def __init__(self):
        self.baseline = BaselineV2()
        self.h2 = H2_V2()
        self.h3 = H3_V2()
    
    def extract_features(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        return {
            'baseline': self.baseline.extract_features(encoder, image),
            'h2': self.h2.extract_features(encoder, image),
            'h3': self.h3.extract_features(encoder, image),
        }
    
    def extract_all(self, encoder, image: Image.Image) -> np.ndarray:
        feats = self.extract_features(encoder, image)
        return np.concatenate([feats['baseline'], feats['h2'], feats['h3']])
