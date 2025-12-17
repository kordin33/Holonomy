"""
degradation_commutator.py - Commutator Energy & Loop Holonomy

Hipoteza #2a: Sygnatura nieprzemienności degradacji (Commutator Energy)
    AI ma większą "nieprzemienność" w przestrzeni cech dla transformacji 
    (kompresja/resize/blur) niż real obrazy.
    
    Definicja:
        C_{a,b}(x) = e(a(b(x))) - e(b(a(x)))
        E_{a,b}(x) = ||C_{a,b}(x)||_2
        
    Format-agnostyczne - mierzysz reakcję, nie format!

Hipoteza #2b: Holonomia orbity (Loop Signature)
    Real ma mniejszą holonomię dla pętli degradacji (spójność odpowiedzi),
    AI ma większą (defekty struktury mikrotekstur).
    
    Definicja (dyskretna pętla):
        z_0 = e(x)
        z_1 = e(t_1(x))
        z_2 = e(t_2(t_1(x)))
        ...
        z_n = e(t_n(...(x)))
        
        H(x) = ||z_n - z_0||_2
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple, Callable, Dict
import io


# ============================================================================
# DEGRADATION OPERATORS
# ============================================================================

def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """JPEG compression with specified quality."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with specified sigma."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def downscale_upscale(image: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale then upscale back to original size."""
    w, h = image.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    
    # Downscale
    downscaled = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Upscale back
    upscaled = downscaled.resize((w, h), Image.LANCZOS)
    
    return upscaled


def add_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    """Add Gaussian noise to image."""
    img_array = np.array(image).astype(np.float32) / 255.0
    
    noise = np.random.normal(0, std, img_array.shape)
    noisy = img_array + noise
    noisy = np.clip(noisy, 0, 1)
    
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    return Image.fromarray(noisy_uint8)


def identity(image: Image.Image) -> Image.Image:
    """Identity transformation (no-op)."""
    return image.copy()


# ============================================================================
# TRANSFORMATION REGISTRY
# ============================================================================

# Predefined transformations
TRANSFORMATIONS = {
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'jpeg_50': lambda img: jpeg_compression(img, 50),
    
    'blur_0.3': lambda img: gaussian_blur(img, 0.3),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'blur_1.0': lambda img: gaussian_blur(img, 1.0),
    
    'scale_0.5': lambda img: downscale_upscale(img, 0.5),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
    
    'noise_0.01': lambda img: add_gaussian_noise(img, 0.01),
    'noise_0.02': lambda img: add_gaussian_noise(img, 0.02),
    
    'identity': identity,
}


# ============================================================================
# COMMUTATOR ENERGY (Hipoteza #2a)
# ============================================================================

def compute_commutator_energy(
    encoder,
    image: Image.Image,
    transform_a: Callable,
    transform_b: Callable
) -> float:
    """
    Oblicza E_{a,b}(x) = ||e(a(b(x))) - e(b(a(x)))||_2
    
    Args:
        encoder: Encoder object
        image: PIL Image
        transform_a: Transformacja a
        transform_b: Transformacja b
    
    Returns:
        Commutator energy (scalar)
    """
    # a(b(x))
    img_ab = transform_a(transform_b(image))
    
    # b(a(x))
    img_ba = transform_b(transform_a(image))
    
    # Embeddings
    e_ab = encoder.encode_batch([img_ab], batch_size=1, show_progress=False)
    e_ba = encoder.encode_batch([img_ba], batch_size=1, show_progress=False)
    
    # L2 distance
    energy = np.linalg.norm(e_ab - e_ba)
    
    return energy


def compute_commutator_energy_batch(
    encoder,
    image: Image.Image,
    transform_a: Callable,
    transform_b: Callable
) -> float:
    """
    SZYBSZA WERSJA: Batch processing dla a(b(x)) i b(a(x)).
    """
    # Przygotuj obrazy
    img_ab = transform_a(transform_b(image))
    img_ba = transform_b(transform_a(image))
    
    # Batch encode
    embeddings = encoder.encode_batch([img_ab, img_ba], batch_size=2, show_progress=False)
    
    e_ab = embeddings[0]
    e_ba = embeddings[1]
    
    # L2 distance
    energy = np.linalg.norm(e_ab - e_ba)
    
    return energy


# Predefined transformation pairs for commutator
COMMUTATOR_PAIRS = [
    ('jpeg_60', 'blur_0.7'),
    ('jpeg_80', 'scale_0.75'),
    ('blur_0.5', 'scale_0.75'),
    ('jpeg_50', 'blur_1.0'),
    ('jpeg_70', 'scale_0.9'),
    ('blur_0.3', 'jpeg_80'),
    ('scale_0.5', 'jpeg_60'),
    ('blur_0.7', 'scale_0.9'),
    ('jpeg_90', 'blur_0.5'),
    ('scale_0.75', 'blur_0.7'),
]


def extract_commutator_features(
    encoder,
    image: Image.Image,
    pairs: List[Tuple[str, str]] = None
) -> np.ndarray:
    """
    Ekstraktuje cechy commutator energy dla wielu par transformacji.
    
    Args:
        encoder: Encoder object
        image: PIL Image
        pairs: Lista par (transform_a_name, transform_b_name)
    
    Returns:
        Feature vector (N_pairs,)
    """
    if pairs is None:
        pairs = COMMUTATOR_PAIRS
    
    features = []
    
    for a_name, b_name in pairs:
        transform_a = TRANSFORMATIONS[a_name]
        transform_b = TRANSFORMATIONS[b_name]
        
        energy = compute_commutator_energy_batch(encoder, image, transform_a, transform_b)
        features.append(energy)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# LOOP HOLONOMY (Hipoteza #2b)
# ============================================================================

def compute_loop_holonomy(
    encoder,
    image: Image.Image,
    loop: List[str]
) -> float:
    """
    Oblicza holonomię pętli: H(x) = ||z_n - z_0||_2
    
    Args:
        encoder: Encoder object
        image: PIL Image
        loop: Lista nazw transformacji (w kolejności)
    
    Returns:
        Holonomy (scalar)
    """
    # z_0 = e(x)
    e_0 = encoder.encode_batch([image], batch_size=1, show_progress=False)
    
    # Zastosuj transformacje sekwencyjnie
    current_img = image
    for transform_name in loop:
        transform = TRANSFORMATIONS[transform_name]
        current_img = transform(current_img)
    
    # z_n = e(t_n(...t_1(x)))
    e_n = encoder.encode_batch([current_img], batch_size=1, show_progress=False)
    
    # Holonomy
    holonomy = np.linalg.norm(e_n - e_0)
    
    return holonomy


def compute_loop_holonomy_batch(
    encoder,
    image: Image.Image,
    loop: List[str]
) -> Tuple[float, List[np.ndarray]]:
    """
    SZYBSZA WERSJA: Batch processing dla całej pętli.
    
    Returns:
        (holonomy, embeddings_list)
    """
    # Przygotuj wszystkie obrazy w pętli
    images = [image]  # z_0
    current_img = image
    
    for transform_name in loop:
        transform = TRANSFORMATIONS[transform_name]
        current_img = transform(current_img)
        images.append(current_img)
    
    # Batch encode wszystkich
    embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
    
    e_0 = embeddings[0]
    e_n = embeddings[-1]
    
    # Holonomy
    holonomy = np.linalg.norm(e_n - e_0)
    
    return holonomy, embeddings


# Predefined loops
HOLONOMY_LOOPS = [
    # Loop 1: JPEG → scale → JPEG → scale back
    ['jpeg_90', 'scale_0.75', 'jpeg_50', 'scale_0.75'],
    
    # Loop 2: blur → JPEG → blur → identity
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],
    
    # Loop 3: scale down/up → JPEG → blur
    ['scale_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.5'],
    
    # Loop 4: JPEG cascade with blur
    ['jpeg_80', 'blur_0.3', 'jpeg_60', 'blur_0.5'],
    
    # Loop 5: scale → blur → JPEG → scale
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],
    
    # Loop 6: aggressive compression loop
    ['jpeg_50', 'scale_0.75', 'blur_1.0', 'jpeg_80'],
    
    # Loop 7: gentle degradation loop
    ['jpeg_90', 'blur_0.3', 'scale_0.9', 'jpeg_80'],
    
    # Loop 8: mixed loop
    ['blur_0.5', 'scale_0.75', 'jpeg_60', 'blur_0.7'],
]


def extract_holonomy_features(
    encoder,
    image: Image.Image,
    loops: List[List[str]] = None
) -> np.ndarray:
    """
    Ekstraktuje cechy holonomii dla wielu pętli.
    
    Args:
        encoder: Encoder object
        image: PIL Image
        loops: Lista pętli (każda pętla = lista nazw transformacji)
    
    Returns:
        Feature vector (N_loops,)
    """
    if loops is None:
        loops = HOLONOMY_LOOPS
    
    features = []
    
    for loop in loops:
        holonomy, _ = compute_loop_holonomy_batch(encoder, image, loop)
        features.append(holonomy)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# COMBINED FEATURES
# ============================================================================

def extract_degradation_invariance_features(
    encoder,
    image: Image.Image,
    pairs: List[Tuple[str, str]] = None,
    loops: List[List[str]] = None
) -> np.ndarray:
    """
    Ekstraktuje WSZYSTKIE cechy: commutator + holonomy.
    
    Returns:
        Feature vector:
            [commutator_1, ..., commutator_M, holonomy_1, ..., holonomy_N]
            
        Domyślnie: 10 commutator + 8 holonomy = 18 wymiarów
    """
    # Commutator features
    comm_features = extract_commutator_features(encoder, image, pairs)
    
    # Holonomy features
    hol_features = extract_holonomy_features(encoder, image, loops)
    
    # Combine
    combined = np.concatenate([comm_features, hol_features])
    
    return combined


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def extract_batch_degradation_invariance(
    encoder,
    images: List[Image.Image],
    pairs: List[Tuple[str, str]] = None,
    loops: List[List[str]] = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Ekstraktuje cechy dla batch'a obrazów.
    
    Returns:
        Array (N, 18) - domyślnie 10 commutator + 8 holonomy
    """
    from tqdm.auto import tqdm
    
    features_list = []
    
    iterator = tqdm(images, desc="Degradation Invariance") if show_progress else images
    
    for img in iterator:
        features = extract_degradation_invariance_features(encoder, img, pairs, loops)
        features_list.append(features)
    
    return np.array(features_list, dtype=np.float32)


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def visualize_commutator_example(
    encoder,
    image: Image.Image,
    transform_a_name: str,
    transform_b_name: str,
    title: str = "Commutator Example"
):
    """
    Wizualizuje przykład kommutatora: a(b(x)) vs b(a(x)).
    """
    import matplotlib.pyplot as plt
    
    transform_a = TRANSFORMATIONS[transform_a_name]
    transform_b = TRANSFORMATIONS[transform_b_name]
    
    # Obrazy
    img_ab = transform_a(transform_b(image))
    img_ba = transform_b(transform_a(image))
    
    # Embeddings
    e_ab = encoder.encode_batch([img_ab], batch_size=1, show_progress=False)
    e_ba = encoder.encode_batch([img_ba], batch_size=1, show_progress=False)
    
    energy = np.linalg.norm(e_ab - e_ba)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img_ab)
    axes[1].set_title(f"{transform_a_name}({transform_b_name}(x))", fontsize=11)
    axes[1].axis('off')
    
    axes[2].imshow(img_ba)
    axes[2].set_title(f"{transform_b_name}({transform_a_name}(x))", fontsize=11)
    axes[2].axis('off')
    
    plt.suptitle(f"{title}\nCommutator Energy: {energy:.4f}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_loop_trajectory(
    encoder,
    image: Image.Image,
    loop: List[str],
    title: str = "Loop Trajectory"
):
    """
    Wizualizuje trajektorię pętli w embedding space (PCA 2D).
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Get all embeddings
    holonomy, embeddings_list = compute_loop_holonomy_batch(encoder, image, loop)
    
    # PCA do 2D
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings_list)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Trajectory
    ax.plot(proj[:, 0], proj[:, 1], 'o-', linewidth=2, markersize=8, alpha=0.7)
    
    # Start and end points
    ax.scatter(proj[0, 0], proj[0, 1], c='green', s=200, marker='*', 
              label='Start', zorder=10, edgecolors='black', linewidth=1.5)
    ax.scatter(proj[-1, 0], proj[-1, 1], c='red', s=200, marker='X', 
              label='End', zorder=10, edgecolors='black', linewidth=1.5)
    
    # Annotate steps
    for i, (x, y) in enumerate(proj):
        label = loop[i-1] if i > 0 else "x"
        ax.annotate(f"{i}: {label}", (x, y), fontsize=8, 
                   xytext=(5, 5), textcoords='offset points')
    
    # Holonomy line
    ax.plot([proj[0, 0], proj[-1, 0]], [proj[0, 1], proj[-1, 1]], 
           'r--', linewidth=2, alpha=0.5, label=f'Holonomy={holonomy:.4f}')
    
    ax.set_title(f"{title}\nLoop: {' → '.join(loop)}", 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PCA Dim 1")
    ax.set_ylabel("PCA Dim 2")
    
    plt.tight_layout()
    
    return fig
