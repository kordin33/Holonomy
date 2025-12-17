"""
degradation_commutator_v3_fixed.py - POPRAWIONA wersja po analizie

NAPRAWY:
1. H_res (residual) zamiast H_norm (ratio) - nie zabija sygnału!
2. Shape features: curvature, tortuosity, std_step (nie tylko scale)
3. Pipeline ze StandardScaler w testach
4. Minimalny zestaw: H_raw, tortuosity, curvature, H_res (4 cechy × 9 pętli = 36D)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple, Dict
import io


# ============================================================================
# DEGRADATION OPERATORS - DETERMINISTYCZNE
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
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    downscaled = image.resize((new_w, new_h), Image.LANCZOS)
    upscaled = downscaled.resize((w, h), Image.LANCZOS)
    return upscaled


def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def adjust_gamma(image: Image.Image, gamma: float = 1.5) -> Image.Image:
    img_array = np.array(image).astype(np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    return Image.fromarray((corrected * 255).astype(np.uint8))


def adjust_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def identity(image: Image.Image) -> Image.Image:
    return image.copy()


# ============================================================================
# REGISTRY - DETERMINISTYCZNE
# ============================================================================

TRANSFORMATIONS = {
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
# TRAJECTORY SHAPE FEATURES (POPRAWIONE!)
# ============================================================================

def compute_trajectory_shape_features(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Oblicza SHAPE features z trajektorii (nie tylko scale!).
    
    Cechy:
    - H_raw: ||z_end - z_0|| (główna!)
    - path_length: Σ||Δ_i||
    - mean_step: mean(||Δ_i||)
    - std_step: std(||Δ_i||) - mierzy regularność
    - max_step: max(||Δ_i||)
    - curvature: mean(1 - cos(Δ_i, Δ_{i+1})) - ile skręca
    - tortuosity: path_length / (H_raw + ε) - stabilniejsze niż H_norm!
    """
    E = np.asarray(embeddings)
    
    # Wektory kroków Δ_i = z_{i+1} - z_i
    D = E[1:] - E[:-1]
    step_norms = np.linalg.norm(D, axis=1)
    
    # Scale features (z zabezpieczeniem przed pustymi)
    if len(step_norms) == 0:
        return {
            'H_raw': 0.0, 'path_length': 0.0, 'mean_step': 0.0,
            'std_step': 0.0, 'max_step': 0.0, 'curvature': 0.0, 'tortuosity': 0.0
        }
    
    path_length = float(step_norms.sum())
    mean_step = float(step_norms.mean())
    std_step = float(step_norms.std()) if len(step_norms) > 1 else 0.0
    max_step = float(step_norms.max())
    
    # Curvature: ile skręca trajektoria
    curvature = 0.0
    if len(D) >= 2:
        # cos(Δ_i, Δ_{i+1})
        norms_prev = np.linalg.norm(D[:-1], axis=1)
        norms_next = np.linalg.norm(D[1:], axis=1)
        
        # Zabezpieczenie przed dzieleniem przez zero
        valid = (norms_prev > 1e-8) & (norms_next > 1e-8)
        
        if valid.any():
            numerator = np.sum(D[:-1][valid] * D[1:][valid], axis=1)
            denominator = norms_prev[valid] * norms_next[valid]
            cos_angles = np.clip(numerator / denominator, -1, 1)
            curvature = float(np.mean(1 - cos_angles))
    
    # H_raw
    H_raw = float(np.linalg.norm(E[-1] - E[0]))
    
    # Tortuosity: path_length / H_raw
    tortuosity = path_length / (H_raw + 1e-8)
    
    # Zabezpieczenie przed NaN/Inf
    result = {
        'H_raw': H_raw,
        'path_length': path_length,
        'mean_step': mean_step,
        'std_step': std_step,
        'max_step': max_step,
        'curvature': curvature,
        'tortuosity': tortuosity,
    }
    
    # Zamień NaN/Inf na 0
    for key in result:
        if not np.isfinite(result[key]):
            result[key] = 0.0
    
    return result


def compute_holonomy_with_shape(
    encoder,
    image: Image.Image,
    loop: List[str]
) -> Dict[str, float]:
    """
    Oblicza trajectory features dla pętli.
    """
    # Wszystkie obrazy w pętli
    images = [image]
    current_img = image
    
    for transform_name in loop:
        transform = TRANSFORMATIONS[transform_name]
        current_img = transform(current_img)
        images.append(current_img)
    
    # Batch encode
    embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
    
    # Shape features
    return compute_trajectory_shape_features(embeddings)


# ============================================================================
# LOOPS
# ============================================================================

LOOPS = [
    # Best from baseline
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],      # Loop 5 (best!)
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],
    ['scale_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.5'],
    ['jpeg_80', 'blur_0.3', 'jpeg_60', 'blur_0.5'],
    ['jpeg_50', 'scale_0.75', 'blur_1.0', 'jpeg_80'],
    ['jpeg_90', 'blur_0.3', 'scale_0.9', 'jpeg_80'],
    ['blur_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.7'],
    ['jpeg_90', 'scale_0.75', 'jpeg_50', 'scale_0.75'],
    ['sharpen_1.5', 'jpeg_80', 'scale_0.75'],
]


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_minimal_features(
    encoder,
    image: Image.Image
) -> Dict[str, np.ndarray]:
    """
    Ekstraktuje MINIMALNY zestaw cech który ma szansę działać:
    
    Dla każdej pętli:
    - H_raw (główna!)
    - tortuosity (stabilniejsze niż H_norm)
    - curvature (shape, nie scale)
    - std_step (regularność)
    
    = 4 cechy × 9 pętli = 36D
    
    Plus osobno:
    - only_H_raw: tylko H_raw (9D) - baseline
    - shape_features: tortuosity + curvature + std_step (27D)
    """
    only_H_raw = []
    shape_features = []
    all_features = []
    
    for loop in LOOPS:
        stats = compute_holonomy_with_shape(encoder, image, loop)
        
        # Baseline
        only_H_raw.append(stats['H_raw'])
        
        # Shape (nie scale!)
        shape_features.extend([
            stats['tortuosity'],
            stats['curvature'],
            stats['std_step']
        ])
        
        # Combined minimal (4 cechy per loop)
        all_features.extend([
            stats['H_raw'],
            stats['tortuosity'],
            stats['curvature'],
            stats['std_step']
        ])
    
    return {
        'H_raw': np.array(only_H_raw, dtype=np.float32),           # 9D
        'shape': np.array(shape_features, dtype=np.float32),       # 27D
        'minimal': np.array(all_features, dtype=np.float32),       # 36D
    }


def extract_batch_minimal_features(
    encoder,
    images: List[Image.Image],
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """Batch extraction."""
    from tqdm.auto import tqdm
    
    results = {'H_raw': [], 'shape': [], 'minimal': []}
    
    iterator = tqdm(images, desc="Extracting") if show_progress else images
    
    for img in iterator:
        feats = extract_minimal_features(encoder, img)
        for key in results.keys():
            results[key].append(feats[key])
    
    for key in results.keys():
        results[key] = np.array(results[key], dtype=np.float32)
    
    return results
