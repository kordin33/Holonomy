"""
holonomy_v22.py - V22: SPECIALIZED FREQUENCY PROBES
Goal: Break 0.92 barrier by disentangling degradation types.

Strategy:
Instead of mixing transformations, we create 3 specialized loop sets:
1. NOISE_LOOPS: Focus on High-Freq (Sharpen, Gaussian Noise, Blur).
   - Detects diffusion noise artifacts.
2. GRID_LOOPS: Focus on Blocking (JPEG, WebP).
   - Detects GAN checkerboard/grid artifacts.
3. SCALE_LOOPS: Focus on Interpolation (Resize Up/Down).
   - Detects resampling anomalies.

Architecture:
- Global Probe (3 x 63D = 189D)
- No patches initially (to test probe strength).
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import random

# ============================================================================
# TRANSFORMS
# ============================================================================

def jpeg(img: Image.Image, q: int) -> Image.Image:
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=q)
    out.seek(0); return Image.open(out).convert('RGB')

def blur(img: Image.Image, r: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=r))

def sharpen(img: Image.Image, f: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(f)

def noise(img: Image.Image, std: float) -> Image.Image:
    # Add Gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def scale(img: Image.Image, f: float) -> Image.Image:
    w, h = img.size
    return img.resize((int(w*f), int(h*f)), Image.LANCZOS).resize((w, h), Image.LANCZOS)

TRANSFORMS = {
    'j90': lambda x: jpeg(x, 90), 'j70': lambda x: jpeg(x, 70), 'j50': lambda x: jpeg(x, 50),
    'b0.5': lambda x: blur(x, 0.5), 'b1.0': lambda x: blur(x, 1.0),
    's1.5': lambda x: sharpen(x, 1.5), 's2.0': lambda x: sharpen(x, 2.0),
    'n10': lambda x: noise(x, 10.0), 'n20': lambda x: noise(x, 20.0),
    'sc0.5': lambda x: scale(x, 0.5), 'sc0.8': lambda x: scale(x, 0.8), 'sc1.2': lambda x: scale(x, 1.2)
}

# ============================================================================
# SPECIALIZED LOOP SETS
# ============================================================================

# 1. NOISE PROBE (High Frequency / Texture)
LOOPS_NOISE = [
    ['s1.5', 'b0.5', 's1.5'], 
    ['n10', 'b0.5', 'n10'],
    ['s2.0', 'b1.0', 's1.5'],
    ['n20', 'b0.5', 's1.5', 'n10'],
    ['b0.5', 's2.0', 'b0.5', 's1.5']
]

# 2. GRID PROBE (Compression / Blocking)
LOOPS_GRID = [
    ['j90', 'j70', 'j90'],
    ['j70', 'j50', 'j70'],
    ['j90', 'j50', 'j90', 'j50'],
    ['j90', 's1.5', 'j70'], # Sharpness interacts with JPEG grid
    ['j50', 'b0.5', 'j50']
]

# 3. SCALE PROBE (Interpolation / Aliasing)
LOOPS_SCALE = [
    ['sc0.8', 'sc0.5', 'sc0.8'],
    ['sc0.5', 's1.5', 'sc0.5'],
    ['sc0.8', 'j90', 'sc0.8'],
    ['sc1.2', 'sc0.8', 'sc1.2'], # Super-resolution artifacts check
    ['sc0.5', 'sc0.5', 'sc0.5']
]

ALL_PROBES = {
    'noise': LOOPS_NOISE,
    'grid': LOOPS_GRID,
    'scale': LOOPS_SCALE
}

# ============================================================================
# METRICS
# ============================================================================

def l2_normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-10)

def chordal_dist(a, b):
    return float(np.sqrt(max(0, 2 - 2 * np.clip(np.dot(a, b), -1, 1))))

def compute_metrics(embs):
    # Standard 7 metrics
    steps = [chordal_dist(embs[i], embs[i+1]) for i in range(len(embs)-1)]
    H = chordal_dist(embs[0], embs[-1])
    L = sum(steps)
    return [
        H, L, L/(H+1e-8), 
        np.std(steps) if len(steps)>1 else 0,
        np.mean(steps), np.max(steps),
        # Simple curvature approximation (H/L ratio is already kind of curvature)
        1.0 - (H/L) if L>0 else 0
    ]

class HolonomyV22:
    """V22: Specialized Frequency Probes."""
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        all_imgs = []
        counts = []
        
        # Collect Images
        for probe_name, loops in ALL_PROBES.items():
            for loop in loops:
                imgs = [image]
                curr = image
                for t in loop:
                    curr = TRANSFORMS[t](curr)
                    imgs.append(curr)
                all_imgs.extend(imgs)
                counts.append(len(imgs))
                
        # Encode Batch
        embs = encoder.encode_batch(all_imgs, batch_size=64, show_progress=False)
        embs = np.array([l2_normalize(e) for e in embs])
        
        # Compute Features
        features = []
        cursor = 0
        for count in counts:
            loop_embs = embs[cursor:cursor+count]
            features.extend(compute_metrics(loop_embs))
            cursor += count
            
        return np.array(features, dtype=np.float32)
