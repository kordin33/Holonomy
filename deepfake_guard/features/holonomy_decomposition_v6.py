"""
holonomy_decomposition_v6.py - GRAND UNIFIED HOLONOMY (H2 + H3 + Base + Commutator)

To jest ostateczna wersja łącząca wszystkie najlepsze hipotezy:
1. Baseline Enhanced (9 loops, cosine metrics, shape features)
2. H3 Dispersion (Patche!, Gram eigenvalues, Step stats)
3. H2 Scale Law (Cosine/Chordal distances, robust fit)
4. Commutator Holonomy (A∘B vs B∘A - jedyna stabilna część Decomp)

Eliminuje błędne założenia (tangent space projection) i skupia się na Robust Geometric Features.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
from scipy.stats import theilslopes
import io


# ============================================================================
# SHARED UTILS (GEOMETRY & DEGRADATIONS)
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

def brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)

def contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))

def chordal_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Stabilna metryka na sferze: sqrt(2 - 2*dot)."""
    return float(np.sqrt(max(0, 2.0 - 2.0 * np.dot(a, b))))


# ============================================================================
# PART 1: BASELINE ENHANCED (9 Loops)
# ============================================================================

# Pełny zestaw 9 pętli (brute force coverage)
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

class PartBaseline:
    def process(self, encoder, image) -> np.ndarray:
        features = []
        for loop in BASELINE_LOOPS:
            imgs = [image]
            curr = image
            for name in loop:
                curr = TRANSFORMS[name](curr)
                imgs.append(curr)
            
            emb = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
            emb = l2_normalize(np.asarray(emb))
            
            # Cechy geometryczne (Chordal metric!)
            steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
            H_chordal = chordal_dist(emb[0], emb[-1])
            path_len = sum(steps)
            
            features.extend([
                H_chordal,
                path_len,
                path_len / (H_chordal + 1e-8),  # tortuosity
                np.std(steps) if len(steps)>1 else 0.0,
            ])
            
        return np.array(features, dtype=np.float32)


# ============================================================================
# PART 2: COMMUTATOR (A∘B vs B∘A)
# ============================================================================

class PartCommutator:
    def process(self, encoder, image) -> np.ndarray:
        # A = jpeg, B = blur
        step_A = lambda img: jpeg_compression(img, 75)
        step_B = lambda img: gaussian_blur(img, 0.6)
        
        # Paths
        img_A = step_A(image)
        img_AB = step_B(img_A)
        
        img_B = step_B(image)
        img_BA = step_A(img_B)
        
        embs = encoder.encode_batch([image, img_AB, img_BA], batch_size=3, show_progress=False)
        z0, zAB, zBA = l2_normalize(np.asarray(embs))
        
        # Commutator energy
        comm_dist = chordal_dist(zAB, zBA)
        
        # Closures
        clos_AB = chordal_dist(z0, zAB)
        clos_BA = chordal_dist(z0, zBA)
        
        return np.array([
            comm_dist, 
            abs(clos_AB - clos_BA),
            (clos_AB + clos_BA) / 2
        ], dtype=np.float32)


# ============================================================================
# PART 3: H3 DISPERSION (WITH PATCHES!)
# ============================================================================

class PartH3:
    def process(self, encoder, image) -> np.ndarray:
        # 1. Global dispersion
        global_feat = self._compute_dispersion(encoder, image)
        
        # 2. Patch dispersion (Grid 5)
        w, h = image.size
        cw, ch = w//2, h//2
        patches = [
            image.crop((0,0,cw,ch)),
            image.crop((cw,0,w,ch)),
            image.crop((0,ch,cw,h)),
            image.crop((cw,ch,w,h)),
            image.crop((w//4, h//4, 3*w//4, 3*h//4))
        ]
        # Resize patch to 224
        patches = [p.resize((224,224), Image.LANCZOS) for p in patches]
        
        patch_feats = []
        for p in patches:
            patch_feats.append(self._compute_dispersion(encoder, p))
        
        patch_feats = np.array(patch_feats)
        patch_median = np.median(patch_feats, axis=0)
        patch_p80 = np.percentile(patch_feats, 80, axis=0)
        
        return np.concatenate([global_feat, patch_median, patch_p80])
        
    def _compute_dispersion(self, encoder, img) -> np.ndarray:
        # Sekwencja H3
        seq = [
            lambda x: jpeg_compression(x, 85),
            lambda x: gaussian_blur(x, 0.4),
            lambda x: downscale_upscale(x, 0.9)
        ]
        
        imgs = [img]
        curr = img
        for fn in seq:
            curr = fn(curr)
            imgs.append(curr)
            
        embs = encoder.encode_batch(imgs, batch_size=len(imgs), show_progress=False)
        embs = l2_normalize(np.asarray(embs))
        
        # Mean pairwise distance (robust noise measure)
        dists = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                dists.append(chordal_dist(embs[i], embs[j]))
        
        D_pairwise = np.mean(dists)
        
        # Path closure vs length
        closure = chordal_dist(embs[0], embs[-1])
        length = sum(chordal_dist(embs[i], embs[i+1]) for i in range(len(embs)-1))
        D_path = closure / (length + 1e-8)
        
        return np.array([D_pairwise, D_path], dtype=np.float32)


# ============================================================================
# PART 4: H2 SCALE LAW (ROBUST)
# ============================================================================

class PartH2:
    def process(self, encoder, image) -> np.ndarray:
        intensities = [
            (95, 0.2), (90, 0.3), (85, 0.4), (80, 0.5), (75, 0.6),
            (70, 0.7), (65, 0.8), (60, 0.9), (55, 1.0), (50, 1.1), (45, 1.2)
        ]
        
        z0 = l2_normalize(encoder.encode_batch([image], batch_size=1, show_progress=False)[0])
        
        Hs = []
        Ss = []
        
        for jq, bs in intensities:
            im = jpeg_compression(image, jq)
            im = gaussian_blur(im, bs)
            z = l2_normalize(encoder.encode_batch([im], batch_size=1, show_progress=False)[0])
            
            Hs.append(chordal_dist(z, z0))
            Ss.append(100 - jq) # Strength parameter directly
            
        Hs = np.array(Hs)
        Ss = np.array(Ss)
        
        # Robust fit log-log
        valid = (Hs > 1e-6)
        if valid.sum() < 4:
            return np.zeros(3, dtype=np.float32)
            
        log_h = np.log(Hs[valid])
        log_s = np.log(Ss[valid])
        
        slope, _, _, _ = theilslopes(log_h, log_s)
        
        return np.array([
            slope,
            np.mean(Hs),
            np.std(Hs)
        ], dtype=np.float32)


# ============================================================================
# GRAND UNIFIED CLASS
# ============================================================================

class HolonomyDecompositionV6:
    def __init__(self):
        self.base = PartBaseline()
        self.comm = PartCommutator()
        self.h3 = PartH3()
        self.h2 = PartH2()
        
    def extract_features(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        f_base = self.base.process(encoder, image)
        f_comm = self.comm.process(encoder, image)
        f_h3 = self.h3.process(encoder, image)
        f_h2 = self.h2.process(encoder, image)
        
        return {
            'baseline': f_base,
            'commutator': f_comm,
            'h3': f_h3,
            'h2': f_h2,
            'all': np.concatenate([f_base, f_comm, f_h3, f_h2])
        }
