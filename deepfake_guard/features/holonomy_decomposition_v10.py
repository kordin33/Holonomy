"""
holonomy_decomposition_v10.py - DENSE COMMUTATOR FIELD & CURVATURE FLOW

Cel: 0.90 AUC (Standalone)
Strategia: "Deep Geometry"
1. Baseline Enhanced (Core 0.87)
2. Dense Commutator Field (16 patchy): Mapa niespójności przemienności Jpeg<->Blur.
   Deepfake'i generowane dyfuzyjnie nie mają stałej struktury topologicznej - komutator wariuje lokalnie.
3. Curvature Flow: Analiza ewolucji krzywizny wzdłuż trajektorii.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
import io


# ============================================================================
# UTILS (CHORDAL & FLOAT64)
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
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = np.linalg.norm(a64)
    nb = np.linalg.norm(b64)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    dot = np.clip(np.dot(a64, b64) / (na * nb), -1.0, 1.0)
    return float(np.arccos(dot))


# ============================================================================
# PART 1: BASELINE ENHANCED (Core 0.87)
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

class PartBaselineEnhanced:
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
            
            # Geometry
            steps = [chordal_dist(emb[i], emb[i+1]) for i in range(len(emb)-1)]
            H_chordal = chordal_dist(emb[0], emb[-1])
            path_len = sum(steps)
            
            # Curvature flow
            deltas = emb[1:] - emb[:-1]
            angles = []
            if len(deltas) >= 2:
                angles = [cosine_angle(deltas[i], deltas[i+1]) for i in range(len(deltas)-1)]
            
            curv_mean = np.mean(angles) if angles else 0.0
            curv_max = np.max(angles) if angles else 0.0
            
            features.extend([
                H_chordal,
                path_len,
                path_len / (H_chordal + 1e-8),
                np.std(steps) if len(steps)>1 else 0.0,
                curv_mean,
                curv_max # New: max curvature spike
            ])
        return np.array(features, dtype=np.float32)


# ============================================================================
# PART 2: DENSE COMMUTATOR FIELD (Grid 4x4)
# ============================================================================

class PartDenseCommutator:
    def __init__(self):
        # Commutator Pair: JPEG vs BLUR (strongest signal in V7)
        self.opA = lambda img: jpeg_compression(img, 70)
        self.opB = lambda img: gaussian_blur(img, 0.6)
        
    def _get_16_patches(self, image):
        w, h = image.size
        cw, ch = w//4, h//4
        patches = []
        for i in range(4):
            for j in range(4):
                p = image.crop((i*cw, j*ch, (i+1)*cw, (j+1)*ch))
                # Resize dla stabilności CLIPa
                patches.append(p.resize((224,224), Image.LANCZOS))
        return patches

    def process(self, encoder, image) -> np.ndarray:
        patches = self._get_16_patches(image)
        
        # Batch processing: 16 patches * 3 images (Base, AB, BA) = 48 images
        # CLIP can handle batch 48.
        
        batch_imgs = []
        for p in patches:
            pA = self.opA(p)
            pAB = self.opB(pA)
            pB = self.opB(p)
            pBA = self.opA(pB)
            batch_imgs.extend([pAB, pBA]) # Only endpoints needed for comm dist
            
        embs = encoder.encode_batch(batch_imgs, batch_size=32, show_progress=False) # Batch 32 safe
        embs = l2_normalize(np.asarray(embs))
        
        # Calculate commutators for each patch
        comm_dists = []
        for i in range(16):
            zAB = embs[2*i]
            zBA = embs[2*i+1]
            comm_dists.append(chordal_dist(zAB, zBA))
            
        comm_dists = np.array(comm_dists)
        
        # Spatial Statistics of the "Commutator Field"
        mean_comm = np.mean(comm_dists)
        std_comm = np.std(comm_dists)
        max_comm = np.max(comm_dists)
        range_comm = max_comm - np.min(comm_dists)
        
        # Entropy of the field (rough estimation)
        # histogram bins=5
        hist, _ = np.histogram(comm_dists, bins=5, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        return np.array([
            mean_comm, std_comm, max_comm, range_comm, entropy
        ], dtype=np.float32)


# ============================================================================
# PART 3: HOLONOMY DECOMP V10 (Unified)
# ============================================================================

class HolonomyDecompositionV10:
    def __init__(self):
        self.base = PartBaselineEnhanced()
        self.dense = PartDenseCommutator()
        
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Baseline (9*6 = 54 features)
        f_base = self.base.process(encoder, image)
        
        # 2. Dense Commutator (5 features)
        f_dense = self.dense.process(encoder, image)
        
        return np.concatenate([f_base, f_dense])
