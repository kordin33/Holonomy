"""
quantization_scaling.py - Embedding Noise Scaling Exponent

Hipoteza: Naturalne obrazy mają stabilne "natural image statistics". 
Kontrolowana kwantyzacja z ditheringiem wywołuje w embeddingu wzrost szumu
zgodny z prawem skali. Generaty mogą odbiegać, bo ich mikrotekstury 
mają inną strukturę.

Definicja:
    x_Δ = Q_Δ(x)  - obraz po kwantyzacji z krokiem Δ (z ditheringiem)
    S(Δ) = ||e(x_Δ) - e(x)||₂  - odległość embeddingów
    log S(Δ) ≈ α log Δ + c  - prawo potęgowe
    
Cechy: α (wykładnik), R² (jakość dopasowania), residual_std
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from scipy.stats import linregress


# ============================================================================
# QUANTIZATION WITH DITHERING
# ============================================================================

def floyd_steinberg_dither(img_array: np.ndarray, levels: int) -> np.ndarray:
    """
    Floyd-Steinberg dithering - rozkłada błąd kwantyzacji na sąsiednie piksele.
    
    Args:
        img_array: (H, W, C) float array [0, 1]
        levels: liczba poziomów kwantyzacji (np. 256 / delta)
    
    Returns:
        Dithered quantized array [0, 1]
    """
    img = img_array.copy()
    h, w, c = img.shape
    
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                old_pixel = img[y, x, ch]
                new_pixel = np.round(old_pixel * (levels - 1)) / (levels - 1)
                img[y, x, ch] = new_pixel
                
                quant_error = old_pixel - new_pixel
                
                # Rozprowadź błąd (Floyd-Steinberg kernel)
                if x + 1 < w:
                    img[y, x + 1, ch] += quant_error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img[y + 1, x - 1, ch] += quant_error * 3 / 16
                    img[y + 1, x, ch] += quant_error * 5 / 16
                    if x + 1 < w:
                        img[y + 1, x + 1, ch] += quant_error * 1 / 16
    
    return np.clip(img, 0, 1)


def quantize_with_dithering(image: Image.Image, delta: int) -> Image.Image:
    """
    Kwantyzuje obraz z krokiem Δ używając Floyd-Steinberg dithering.
    
    Args:
        image: PIL Image
        delta: krok kwantyzacji (np. 4, 8, 16, 32, 64)
    
    Returns:
        Zkwantyzowany obraz (PIL Image)
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    
    levels = max(2, 256 // delta)  # liczba poziomów
    
    # Dithering
    quantized = floyd_steinberg_dither(img_array, levels)
    
    # Konwersja z powrotem
    quantized_uint8 = (quantized * 255).astype(np.uint8)
    
    return Image.fromarray(quantized_uint8)


# ============================================================================
# EMBEDDING DISTANCE MEASUREMENT
# ============================================================================

def measure_embedding_distance(
    encoder,
    original_image: Image.Image,
    delta: int
) -> float:
    """
    Mierzy S(Δ) = ||e(x_Δ) - e(x)||₂
    
    Args:
        encoder: Encoder object (z get_encoder)
        original_image: Oryginalny obraz
        delta: Krok kwantyzacji
    
    Returns:
        Odległość L2 między embeddingami
    """
    # Embedding oryginalnego obrazu
    e_orig = encoder.encode_batch([original_image], batch_size=1, show_progress=False)
    
    # Kwantyzacja
    x_delta = quantize_with_dithering(original_image, delta)
    
    # Embedding po kwantyzacji
    e_quant = encoder.encode_batch([x_delta], batch_size=1, show_progress=False)
    
    # Odległość L2
    distance = np.linalg.norm(e_quant - e_orig)
    
    return distance


def measure_embedding_distances_batch(
    encoder,
    original_image: Image.Image,
    deltas: List[int]
) -> np.ndarray:
    """
    SZYBSZA WERSJA: Mierzy S(Δ) dla wszystkich Δ w jednym batchu.
    
    Args:
        encoder: Encoder object
        original_image: Oryginalny obraz
        deltas: Lista kroków kwantyzacji
    
    Returns:
        Array S(Δ) dla każdego Δ
    """
    # Przygotuj batch: [original, quant_1, quant_2, ...]
    batch_images = [original_image]
    for delta in deltas:
        x_delta = quantize_with_dithering(original_image, delta)
        batch_images.append(x_delta)
    
    # Jeden batch encode
    embeddings = encoder.encode_batch(batch_images, batch_size=len(batch_images), show_progress=False)
    
    e_orig = embeddings[0:1]  # First embedding
    e_quants = embeddings[1:]  # Rest
    
    # Odległości L2
    distances = np.linalg.norm(e_quants - e_orig, axis=1)
    
    return distances


# ============================================================================
# SCALING EXPONENT EXTRACTION
# ============================================================================

def extract_scaling_exponent(
    encoder,
    image: Image.Image,
    deltas: List[int] = None
) -> Dict[str, float]:
    """
    Ekstraktuje wykładnik α z prawa skali: log S(Δ) ≈ α log Δ + c
    
    Args:
        encoder: Encoder object
        image: PIL Image
        deltas: Lista kroków kwantyzacji (domyślnie [4, 8, 16, 32, 64])
    
    Returns:
        Dict z cechami:
            - alpha: wykładnik skali
            - r_squared: R² dopasowania
            - residual_std: std residuów
            - intercept: wyraz wolny c
            - mean_S: średnia wartość S(Δ)
    """
    if deltas is None:
        deltas = [4, 8, 16, 32, 64]
    
    # Mierzenie S(Δ) dla wszystkich Δ w jednym batchu (SZYBSZE!)
    S_values = measure_embedding_distances_batch(encoder, image, deltas)
    
    # Log-log regression
    log_deltas = np.log(deltas)
    log_S = np.log(S_values + 1e-10)  # dodajemy epsilon żeby uniknąć log(0)
    
    # Linear regression w log-log space
    slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_S)
    
    # Residuals
    log_S_pred = slope * log_deltas + intercept
    residuals = log_S - log_S_pred
    residual_std = np.std(residuals)
    
    return {
        'alpha': slope,
        'r_squared': r_value ** 2,
        'residual_std': residual_std,
        'intercept': intercept,
        'mean_S': np.mean(S_values),
        'std_S': np.std(S_values),
        'p_value': p_value,
        'std_err': std_err
    }


def extract_quantization_scaling_features(
    encoder,
    image: Image.Image,
    deltas: List[int] = None
) -> np.ndarray:
    """
    Główna funkcja ekstraktująca cechy skalowania kwantyzacji.
    
    Args:
        encoder: Encoder object
        image: PIL Image
        deltas: Lista kroków kwantyzacji
    
    Returns:
        Feature vector (8 wymiarów):
            [0] alpha - wykładnik skali
            [1] r_squared - jakość dopasowania
            [2] residual_std - odchylenie residuów
            [3] intercept - wyraz wolny
            [4] mean_S - średnia odległość
            [5] std_S - std odległości
            [6] p_value - p-value regresji
            [7] std_err - błąd standardowy
    """
    stats = extract_scaling_exponent(encoder, image, deltas)
    
    features = np.array([
        stats['alpha'],
        stats['r_squared'],
        stats['residual_std'],
        stats['intercept'],
        stats['mean_S'],
        stats['std_S'],
        stats['p_value'],
        stats['std_err']
    ], dtype=np.float32)
    
    return features


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def extract_batch_quantization_scaling(
    encoder,
    images: List[Image.Image],
    deltas: List[int] = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Ekstraktuje cechy skalowania dla batch'a obrazów.
    
    Args:
        encoder: Encoder object
        images: Lista obrazów PIL
        deltas: Lista kroków kwantyzacji
        show_progress: Czy pokazać progress bar
    
    Returns:
        Array (N, 8) z cechami
    """
    from tqdm.auto import tqdm
    
    features_list = []
    
    iterator = tqdm(images, desc="Quantization Scaling") if show_progress else images
    
    for img in iterator:
        features = extract_quantization_scaling_features(encoder, img, deltas)
        features_list.append(features)
    
    return np.array(features_list, dtype=np.float32)


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_scaling_law(
    encoder,
    image: Image.Image,
    deltas: List[int] = None,
    title: str = "Quantization Scaling Law"
):
    """
    Wizualizuje prawo skali dla pojedynczego obrazu.
    
    Args:
        encoder: Encoder object
        image: PIL Image
        deltas: Lista kroków kwantyzacji
        title: Tytuł wykresu
    """
    import matplotlib.pyplot as plt
    
    if deltas is None:
        deltas = [4, 8, 16, 32, 64]
    
    # Pomiar S(Δ)
    S_values = []
    for delta in deltas:
        S_delta = measure_embedding_distance(encoder, image, delta)
        S_values.append(S_delta)
    
    S_values = np.array(S_values)
    
    # Dopasowanie
    log_deltas = np.log(deltas)
    log_S = np.log(S_values + 1e-10)
    slope, intercept, r_value, _, _ = linregress(log_deltas, log_S)
    
    # Wykres
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1.plot(deltas, S_values, 'o-', label='Measured S(Δ)')
    ax1.set_xlabel('Δ (quantization step)')
    ax1.set_ylabel('S(Δ) = ||e(x_Δ) - e(x)||₂')
    ax1.set_title('Embedding Distance vs Quantization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log-log scale
    ax2.plot(deltas, S_values, 'o', label='Measured', markersize=8)
    
    # Fitted line
    deltas_fit = np.linspace(min(deltas), max(deltas), 100)
    S_fit = np.exp(slope * np.log(deltas_fit) + intercept)
    ax2.plot(deltas_fit, S_fit, 'r--', 
             label=f'Fit: S ∝ Δ^{slope:.3f}\nR² = {r_value**2:.4f}')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('log Δ')
    ax2.set_ylabel('log S(Δ)')
    ax2.set_title('Log-Log Scale')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
