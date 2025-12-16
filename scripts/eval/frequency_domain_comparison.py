"""
frequency_domain_comparison.py - Porównanie FFT vs DCT vs Wavelet

Test różnych metod ekstrakcji cech w domenie częstotliwości:
1. FFT (Fourier Transform)
2. DCT (Discrete Cosine Transform - JPEG artifacts)
3. Wavelet (Haar, multi-scale)

Benchmark: RGB + każda metoda
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import fftpack
import pywt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pacmap
import io
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("./data/genimage")
OUTPUT_DIR = Path("./results/frequency_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JPEG_QUALITY = 85
MAX_PER_CLASS = 500
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# FREQUENCY DOMAIN FEATURE EXTRACTORS
# ============================================================================

def extract_fft_features(image: Image.Image, n_features: int = 64) -> np.ndarray:
    """FFT-based features."""
    img = np.array(image.convert('L'), dtype=np.float32) / 255.0
    
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    features = []
    
    # Energy in frequency bands
    total_energy = np.sum(magnitude ** 2)
    features.append(np.log1p(total_energy))
    
    # Radial frequency distribution
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    
    n_bands = 16
    for i in range(n_bands):
        r_min = max_r * i / n_bands
        r_max = max_r * (i + 1) / n_bands
        band_mask = (r >= r_min) & (r < r_max)
        if band_mask.sum() > 0:
            band_energy = np.sum(magnitude[band_mask] ** 2)
            features.append(np.log1p(band_energy))
    
    # High/Low frequency ratio
    low_mask = r < max_r * 0.25
    high_mask = r >= max_r * 0.5
    
    low_energy = np.sum(magnitude[low_mask] ** 2)
    high_energy = np.sum(magnitude[high_mask] ** 2)
    features.append(np.log1p(high_energy) / (np.log1p(low_energy) + 1e-8))
    
    # Angular distribution (4 quadrants)
    for i in range(4):
        angle_min = i * np.pi / 2
        angle_max = (i + 1) * np.pi / 2
        angle = np.arctan2(y - cy, x - cx)
        quad_mask = (angle >= angle_min) & (angle < angle_max)
        if quad_mask.sum() > 0:
            quad_energy = np.sum(magnitude[quad_mask] ** 2)
            features.append(np.log1p(quad_energy))
    
    # Normalize
    features = np.array(features[:n_features], dtype=np.float32)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


def extract_dct_features(image: Image.Image, n_features: int = 64) -> np.ndarray:
    """DCT-based features (JPEG compression artifacts detection)."""
    img = np.array(image.convert('L'), dtype=np.float32)
    
    # Full image DCT
    dct = fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')
    
    h, w = dct.shape
    features = []
    
    # DC component
    features.append(dct[0, 0])
    
    # Zigzag scan of DCT coefficients (like JPEG)
    # Extract coefficients in zigzag order
    zigzag_indices = []
    for s in range(min(h, w)):
        if s % 2 == 0:
            for i in range(s + 1):
                if i < h and s - i < w:
                    zigzag_indices.append((i, s - i))
        else:
            for i in range(s + 1):
                if s - i < h and i < w:
                    zigzag_indices.append((s - i, i))
    
    # Take first 50 zigzag coefficients
    for idx in zigzag_indices[:50]:
        if len(features) < n_features:
            features.append(abs(dct[idx]))
    
    # Energy in different DCT regions
    # Top-left (low freq), bottom-right (high freq)
    block_size = h // 4
    for i in range(4):
        for j in range(4):
            y_start = i * block_size
            y_end = (i + 1) * block_size if i < 3 else h
            x_start = j * block_size
            x_end = (j + 1) * block_size if j < 3 else w
            
            block = dct[y_start:y_end, x_start:x_end]
            if len(features) < n_features:
                features.append(np.log1p(np.sum(block ** 2)))
    
    # Normalize
    features = np.array(features[:n_features], dtype=np.float32)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


def extract_wavelet_features(image: Image.Image, n_features: int = 64) -> np.ndarray:
    """Wavelet-based features (multi-scale analysis)."""
    img = np.array(image.convert('L'), dtype=np.float32) / 255.0
    
    # Multi-level 2D wavelet decomposition
    coeffs = pywt.wavedec2(img, 'haar', level=3)
    
    features = []
    
    # Approximation coefficients (low-pass)
    cA3 = coeffs[0]
    features.append(np.mean(cA3))
    features.append(np.std(cA3))
    features.append(np.median(cA3))
    
    # Detail coefficients for each level
    for level_coeffs in coeffs[1:]:
        cH, cV, cD = level_coeffs  # Horizontal, Vertical, Diagonal
        
        for detail in [cH, cV, cD]:
            # Statistical features
            features.append(np.mean(detail))
            features.append(np.std(detail))
            features.append(np.max(np.abs(detail)))
            
            # Energy
            energy = np.sum(detail ** 2)
            features.append(np.log1p(energy))
            
            # Sparsity (measure of compression artifacts)
            threshold = np.std(detail) * 3
            sparsity = (np.abs(detail) < threshold).sum() / detail.size
            features.append(sparsity)
    
    # Normalize
    features = np.array(features[:n_features], dtype=np.float32)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


# ============================================================================
# DATA LOADING
# ============================================================================

def normalize_to_jpeg(img: Image.Image, quality: int = 85) -> Image.Image:
    """Convert to JPEG to normalize compression."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def load_data():
    print("="*50)
    print("LOADING DATA")
    print("="*50)
    
    sources = {
        "Real": (DATA_DIR / "real_pool", 1),
        "Midjourney": (DATA_DIR / "mj_pool", 0),
        "StableDiffusion": (DATA_DIR / "sd_pool", 0),
    }
    
    images, labels, methods = [], [], []
    
    for method_name, (folder, label) in sources.items():
        files = list(folder.glob("*.png")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg"))
        np.random.shuffle(files)
        files = files[:MAX_PER_CLASS]
        
        print(f"\nLoading {method_name}: {len(files)}")
        
        for p in tqdm(files, desc=method_name):
            try:
                img = Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
                img_normalized = normalize_to_jpeg(img, quality=JPEG_QUALITY)
                
                images.append(img_normalized)
                labels.append(label)
                methods.append(method_name)
            except:
                continue
    
    return images, np.array(labels), np.array(methods)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 FREQUENCY DOMAIN COMPARISON")
    print("   FFT vs DCT vs Wavelet")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    images, labels, methods = load_data()
    
    # Split
    train_idx, test_idx = train_test_split(
        np.arange(len(images)), test_size=0.3, random_state=RANDOM_STATE, stratify=methods
    )
    
    train_images = [images[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    methods_test = methods[test_idx]
    
    print(f"\nTrain: {len(train_images)}, Test: {len(test_images)}")
    
    # Extract RGB embeddings
    print("\n" + "="*50)
    print("EXTRACTING RGB EMBEDDINGS (ViT-L/14)")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    rgb_train = encoder.encode_batch(train_images, batch_size=32, show_progress=True)
    rgb_test = encoder.encode_batch(test_images, batch_size=32, show_progress=True)
    
    # Extract frequency features
    print("\n" + "="*50)
    print("EXTRACTING FREQUENCY FEATURES")
    print("="*50)
    
    feature_extractors = {
        "FFT": extract_fft_features,
        "DCT": extract_dct_features,
        "Wavelet": extract_wavelet_features,
    }
    
    freq_features = {}
    
    for name, extractor in feature_extractors.items():
        print(f"\n{name}...")
        train_feat = np.array([extractor(img) for img in tqdm(train_images, desc=f"{name} Train")])
        test_feat = np.array([extractor(img) for img in tqdm(test_images, desc=f"{name} Test")])
        freq_features[name] = (train_feat, test_feat)
        print(f"  Shape: {train_feat.shape}")
    
    # Benchmark
    print("\n" + "="*70)
    print("BENCHMARK: RGB + Frequency Features")
    print("="*70)
    
    results = {}
    
    # Baseline: RGB only
    print("\nRGB (baseline):")
    svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
    svm.fit(rgb_train, y_train)
    y_pred = svm.predict(rgb_test)
    acc = accuracy_score(y_test, y_pred)
    
    per_method = {}
    for method in np.unique(methods_test):
        mask = methods_test == method
        per_method[method] = accuracy_score(y_test[mask], y_pred[mask])
    
    results["RGB"] = {"acc": acc, "per_method": per_method}
    print(f"  Overall: {acc:.2%}")
    for m, a in per_method.items():
        print(f"  {m}: {a:.2%}")
    
    # RGB + Frequency features
    for freq_name, (train_feat, test_feat) in freq_features.items():
        print(f"\nRGB + {freq_name}:")
        
        X_train = np.hstack([rgb_train, train_feat])
        X_test = np.hstack([rgb_test, test_feat])
        
        svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        per_method = {}
        for method in np.unique(methods_test):
            mask = methods_test == method
            per_method[method] = accuracy_score(y_test[mask], y_pred[mask])
        
        results[f"RGB+{freq_name}"] = {"acc": acc, "per_method": per_method}
        print(f"  Overall: {acc:.2%}")
        for m, a in per_method.items():
            print(f"  {m}: {a:.2%}")
    
    # Summary table
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    print("\n{:<15} {:<10} {:<12} {:<12} {:<15}".format(
        "Method", "Overall", "Real", "Midjourney", "StableDiff"
    ))
    print("-"*70)
    
    for name, res in results.items():
        pm = res["per_method"]
        print("{:<15} {:<10.2%} {:<12.2%} {:<12.2%} {:<15.2%}".format(
            name,
            res["acc"],
            pm.get("Real", 0),
            pm.get("Midjourney", 0),
            pm.get("StableDiffusion", 0)
        ))
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]["acc"])
    print(f"\n🏆 BEST: {best[0]} = {best[1]['acc']:.2%}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods_names = list(results.keys())
    overall_accs = [results[m]["acc"] for m in methods_names]
    real_accs = [results[m]["per_method"].get("Real", 0) for m in methods_names]
    
    x = np.arange(len(methods_names))
    width = 0.35
    
    ax.bar(x - width/2, overall_accs, width, label='Overall', color='#3498db')
    ax.bar(x + width/2, real_accs, width, label='Real Only', color='#2ecc71')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy')
    ax.set_title('Frequency Domain Comparison: FFT vs DCT vs Wavelet', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_bar.png", dpi=150)
    print(f"\n✓ Saved: comparison_bar.png")
    plt.close()
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
