"""
degradation_commutator_optimized_v2.py - NAPRAWIONA wersja

FIXES:
1. ✅ H_raw jako główna cecha (nie zastępujemy przez H_norm!)
2. ✅ Patchwise na H_raw (nie H_norm)
3. ✅ Usunięte losowe operacje z commutatorów (crop/noise)
4. ✅ Lokalny RNG zamiast globalnego seed
5. ✅ Combined zawiera H_raw jako bazę
6. ✅ JPEG z .copy() i .close()
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple, Callable, Dict
import io


# ============================================================================
# DEGRADATION OPERATORS - DETERMINISTYCZNE
# ============================================================================

def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """JPEG compression - deterministyczna."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB').copy()
    buffer.close()
    return img


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur - deterministyczna."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def downscale_upscale(image: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale + upscale - deterministyczna."""
    w, h = image.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    downscaled = image.resize((new_w, new_h), Image.LANCZOS)
    upscaled = downscaled.resize((w, h), Image.LANCZOS)
    return upscaled


def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """Wyostrzenie - deterministyczne."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def adjust_gamma(image: Image.Image, gamma: float = 1.5) -> Image.Image:
    """Gamma correction - deterministyczna."""
    img_array = np.array(image).astype(np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    return Image.fromarray((corrected * 255).astype(np.uint8))


def adjust_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Kontrast - deterministyczny."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def identity(image: Image.Image) -> Image.Image:
    """Identity - deterministyczna."""
    return image.copy()


# ============================================================================
# REGISTRY - TYLKO DETERMINISTYCZNE!
# ============================================================================

TRANSFORMATIONS = {
    # JPEG
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_85': lambda img: jpeg_compression(img, 85),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_75': lambda img: jpeg_compression(img, 75),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_65': lambda img: jpeg_compression(img, 65),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'jpeg_50': lambda img: jpeg_compression(img, 50),
    
    # Blur
    'blur_0.3': lambda img: gaussian_blur(img, 0.3),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'blur_1.0': lambda img: gaussian_blur(img, 1.0),
    
    # Scale
    'scale_0.5': lambda img: downscale_upscale(img, 0.5),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
    
    # Enhancements
    'sharpen_1.5': lambda img: sharpen(img, 1.5),
    'sharpen_2.0': lambda img: sharpen(img, 2.0),
    'gamma_1.2': lambda img: adjust_gamma(img, 1.2),
    'gamma_1.5': lambda img: adjust_gamma(img, 1.5),
    'contrast_1.3': lambda img: adjust_contrast(img, 1.3),
    'contrast_1.5': lambda img: adjust_contrast(img, 1.5),
    
    'identity': identity,
}


# ============================================================================
# COMMUTATOR - DETERMINISTYCZNE PARY
# ============================================================================

# Real-world PARY - BEZ crop/noise!
REAL_WORLD_COMMUTATOR_PAIRS = [
    ('sharpen_1.5', 'jpeg_80'),
    ('jpeg_70', 'sharpen_2.0'),
    ('gamma_1.2', 'jpeg_70'),
    ('jpeg_60', 'gamma_1.5'),
    ('scale_0.75', 'jpeg_70'),
    ('jpeg_80', 'scale_0.9'),
    ('blur_0.5', 'jpeg_80'),
    ('jpeg_70', 'contrast_1.3'),
    ('contrast_1.5', 'jpeg_60'),
    ('jpeg_80', 'blur_0.7'),
]


def compute_commutator_energy_batch(
    encoder,
    image: Image.Image,
    transform_a: Callable,
    transform_b: Callable
) -> float:
    """Commutator energy - batch processing."""
    img_ab = transform_a(transform_b(image))
    img_ba = transform_b(transform_a(image))
    
    embeddings = encoder.encode_batch([img_ab, img_ba], batch_size=2, show_progress=False)
    
    e_ab = embeddings[0]
    e_ba = embeddings[1]
    
    energy = np.linalg.norm(e_ab - e_ba)
    
    return energy


def extract_commutator_features(
    encoder,
    image: Image.Image,
    pairs: List[Tuple[str, str]] = None
) -> np.ndarray:
    """Ekstraktuje commutator energy dla deterministycznych par."""
    if pairs is None:
        pairs = REAL_WORLD_COMMUTATOR_PAIRS
    
    features = []
    for a_name, b_name in pairs:
        transform_a = TRANSFORMATIONS[a_name]
        transform_b = TRANSFORMATIONS[b_name]
        
        energy = compute_commutator_energy_batch(encoder, image, transform_a, transform_b)
        features.append(energy)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# HOLONOMY - H_RAW + trajectory features
# ============================================================================

def compute_holonomy_with_trajectory(
    encoder,
    image: Image.Image,
    loop: List[str]
) -> Dict[str, float]:
    """
    Oblicza H_raw + trajectory features (path_length, max_dev, etc.)
    
    Returns dict:
        - H_raw: ||z_end - z_0||
        - path_length: Σ ||z_{i+1} - z_i||
        - max_deviation: max_i ||z_i - z_0||
        - H_norm: H_raw / path_length (jako dodatkowa cecha!)
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
    
    e_0 = embeddings[0]
    e_n = embeddings[-1]
    
    # H_raw (główna cecha!)
    H_raw = np.linalg.norm(e_n - e_0)
    
    # Path length
    path_length = sum(
        np.linalg.norm(embeddings[i] - embeddings[i-1])
        for i in range(1, len(embeddings))
    )
    
    # Max deviation
    max_deviation = max(
        np.linalg.norm(e - e_0)
        for e in embeddings
    )
    
    # H_norm jako dodatkowa cecha (nie zastępuje H_raw!)
    H_norm = H_raw / (path_length + 1e-8)
    
    return {
        'H_raw': H_raw,
        'path_length': path_length,
        'max_deviation': max_deviation,
        'H_norm': H_norm
    }


# ============================================================================
# PATCHWISE - NA H_RAW!
# ============================================================================

def extract_patch_deterministic(
    image: Image.Image, 
    patch_idx: int, 
    n_patches: int, 
    patch_size: int,
    rng: np.random.Generator
) -> Image.Image:
    """Wyciąga patch deterministycznie z lokalnym RNG."""
    w, h = image.size
    
    if w < patch_size or h < patch_size:
        return image.resize((patch_size, patch_size), Image.LANCZOS)
    
    x = rng.integers(0, w - patch_size + 1)
    y = rng.integers(0, h - patch_size + 1)
    
    return image.crop((x, y, x + patch_size, y + patch_size))


def compute_patchwise_holonomy(
    encoder,
    image: Image.Image,
    loop: List[str],
    n_patches: int = 5,
    patch_size: int = 112,
    seed: int = 42
) -> Dict[str, float]:
    """
    Holonomia patchowo - NA H_RAW!
    
    Używa lokalnego RNG (nie globalnego seed!).
    """
    # Lokalny RNG
    rng = np.random.default_rng(seed)
    
    patch_holonomies_raw = []
    
    for patch_idx in range(n_patches):
        patch = extract_patch_deterministic(image, patch_idx, n_patches, patch_size, rng)
        
        # Holonomia dla tego patcha - UŻYWAMY H_RAW!
        stats = compute_holonomy_with_trajectory(encoder, patch, loop)
        patch_holonomies_raw.append(stats['H_raw'])
    
    patch_holonomies_raw = np.array(patch_holonomies_raw)
    
    return {
        'median': np.median(patch_holonomies_raw),
        'p80': np.percentile(patch_holonomies_raw, 80),
        'std': np.std(patch_holonomies_raw),
        'mean': np.mean(patch_holonomies_raw)
    }


# ============================================================================
# LOOPS - te same co baseline (działają!)
# ============================================================================

OPTIMIZED_LOOPS = [
    # Best from baseline
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],      # Loop 5 (best!)
    
    # Real-world scenarios
    ['sharpen_1.5', 'jpeg_80', 'scale_0.75'],
    ['jpeg_60', 'gamma_1.2', 'jpeg_80'],
    ['scale_0.75', 'jpeg_70', 'contrast_1.3'],
    
    # Aggressive
    ['jpeg_50', 'scale_0.5', 'blur_1.0', 'jpeg_70'],
    ['sharpen_2.0', 'jpeg_60', 'scale_0.9'],
    
    # Gentle
    ['jpeg_90', 'blur_0.3', 'scale_0.9', 'jpeg_80'],
    ['gamma_1.2', 'jpeg_85', 'contrast_1.3'],
    
    # Mixed
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],
]


# ============================================================================
# COMBINED - POPRAWNIE!
# ============================================================================

def extract_all_optimized_features_v2(
    encoder,
    image: Image.Image
) -> Dict[str, np.ndarray]:
    """
    NAPRAWIONA wersja - H_raw jako baza!
    
    Returns:
    - 'baseline_holonomy': H_raw dla każdej pętli (9D)
    - 'trajectory_features': path_length, max_dev, H_norm (9×3 = 27D)
    - 'patchwise_holonomy': median/p80/std/mean dla top-3 loops (3×4 = 12D)
    - 'commutator': deterministyczne pary (10D)
    - 'combined': H_raw + trajectory + patchwise + commutator (58D)
    """
    features_dict = {}
    
    # 1. Baseline holonomy (H_raw) - GŁÓWNA CECHA!
    baseline_hol = []
    trajectory_feats = []
    
    for loop in OPTIMIZED_LOOPS:
        stats = compute_holonomy_with_trajectory(encoder, image, loop)
        
        baseline_hol.append(stats['H_raw'])
        trajectory_feats.extend([stats['path_length'], stats['max_deviation'], stats['H_norm']])
    
    features_dict['baseline_holonomy'] = np.array(baseline_hol, dtype=np.float32)
    features_dict['trajectory_features'] = np.array(trajectory_feats, dtype=np.float32)
    
    # 2. Patchwise holonomy (NA H_RAW!) - tylko top-3 loops
    patchwise_feats = []
    for loop in OPTIMIZED_LOOPS[:3]:
        stats = compute_patchwise_holonomy(encoder, image, loop, n_patches=5, patch_size=112)
        patchwise_feats.extend([stats['median'], stats['p80'], stats['std'], stats['mean']])
    
    features_dict['patchwise_holonomy'] = np.array(patchwise_feats, dtype=np.float32)
    
    # 3. Commutator (deterministyczny!)
    features_dict['commutator'] = extract_commutator_features(encoder, image)
    
    # 4. Combined - ZACZYNA SIĘ OD H_RAW!
    combined = np.concatenate([
        features_dict['baseline_holonomy'],      # 9D - GŁÓWNA CECHA!
        features_dict['trajectory_features'],    # 27D
        features_dict['patchwise_holonomy'],     # 12D
        features_dict['commutator']              # 10D
    ])
    features_dict['combined'] = combined
    
    return features_dict
