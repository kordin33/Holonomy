"""
genimage_normalize_and_benchmark.py

1. Konwertuje wszystkie obrazy do JPEG z tym samym quality (eliminuje bias formatu)
2. Benchmark z FFT i bez
3. PaCMAP wizualizacja
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
OUTPUT_DIR = Path("./results/genimage_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JPEG_QUALITY = 85  # Normalizacja do tego samego quality
MAX_PER_CLASS = 500  # Balansujemy do najmniejszej klasy (MJ=500)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# FFT FEATURES
# ============================================================================

def extract_fft_improved(image: Image.Image) -> np.ndarray:
    """Extract FFT features focusing on high-frequency artifacts."""
    img = np.array(image.convert('L'), dtype=np.float32) / 255.0
    
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    features = []
    
    total_energy = np.sum(magnitude ** 2)
    features.append(np.log1p(total_energy))
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    
    low_mask = r < max_r * 0.25
    mid_mask = (r >= max_r * 0.25) & (r < max_r * 0.5)
    high_mask = r >= max_r * 0.5
    
    low_energy = np.sum(magnitude[low_mask] ** 2)
    mid_energy = np.sum(magnitude[mid_mask] ** 2)
    high_energy = np.sum(magnitude[high_mask] ** 2)
    
    features.append(np.log1p(high_energy) / (np.log1p(low_energy) + 1e-8))
    features.append(np.log1p(mid_energy) / (np.log1p(low_energy) + 1e-8))
    features.append(np.log1p(high_energy) / (np.log1p(total_energy) + 1e-8))
    
    for i in range(8):
        r_min = max_r * i / 8
        r_max = max_r * (i + 1) / 8
        band_mask = (r >= r_min) & (r < r_max)
        band_energy = np.sum(magnitude[band_mask] ** 2)
        features.append(np.log1p(band_energy))
    
    q1 = magnitude[:cy, :cx].sum()
    q2 = magnitude[:cy, cx:].sum()
    q3 = magnitude[cy:, :cx].sum()
    q4 = magnitude[cy:, cx:].sum()
    features.append(np.std([q1, q2, q3, q4]) / (np.mean([q1, q2, q3, q4]) + 1e-8))
    
    log_mag = np.log1p(magnitude)
    peaks = (log_mag > log_mag.mean() + 3 * log_mag.std()).sum()
    features.append(peaks / (h * w))
    
    dc = magnitude[cy, cx]
    features.append(dc / (total_energy + 1e-8))
    
    features = np.array(features, dtype=np.float32)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


def normalize_to_jpeg(img: Image.Image, quality: int = 85) -> Image.Image:
    """Convert image to JPEG format in memory to normalize compression artifacts."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    print("="*50)
    print("LOADING & NORMALIZING DATA")
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
        
        print(f"\nLoading {method_name}: {len(files)} images")
        
        for p in tqdm(files, desc=method_name):
            try:
                img = Image.open(p).convert("RGB")
                
                # Resize to common resolution
                img = img.resize((512, 512), Image.LANCZOS)
                
                # Normalize to JPEG (eliminates format bias!)
                img_normalized = normalize_to_jpeg(img, quality=JPEG_QUALITY)
                
                images.append(img_normalized)
                labels.append(label)
                methods.append(method_name)
            except Exception as e:
                continue
    
    print(f"\nTotal: {len(images)} images (all normalized to JPEG q={JPEG_QUALITY})")
    
    return images, np.array(labels), np.array(methods)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 GENIMAGE BENCHMARK (Format-Normalized)")
    print("   Real vs Midjourney vs Stable Diffusion")
    print("   All images normalized to JPEG to eliminate format bias")
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
    
    # Extract embeddings
    print("\n" + "="*50)
    print("EXTRACTING RGB EMBEDDINGS (ViT-L/14)")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    rgb_train = encoder.encode_batch(train_images, batch_size=32, show_progress=True)
    rgb_test = encoder.encode_batch(test_images, batch_size=32, show_progress=True)
    
    print(f"RGB shape: {rgb_train.shape}")
    
    # Extract FFT
    print("\n" + "="*50)
    print("EXTRACTING FFT FEATURES")
    print("="*50)
    
    fft_train = np.array([extract_fft_improved(img) for img in tqdm(train_images, desc="FFT Train")])
    fft_test = np.array([extract_fft_improved(img) for img in tqdm(test_images, desc="FFT Test")])
    
    print(f"FFT shape: {fft_train.shape}")
    
    # Combine
    rgb_fft_train = np.hstack([rgb_train, fft_train])
    rgb_fft_test = np.hstack([rgb_test, fft_test])
    
    # Benchmark
    print("\n" + "="*50)
    print("BENCHMARK: SVM RGB vs RGB+FFT")
    print("="*50)
    
    results = {}
    
    for name, (X_tr, X_te) in [("RGB", (rgb_train, rgb_test)), 
                                ("RGB+FFT", (rgb_fft_train, rgb_fft_test))]:
        print(f"\n{name}:")
        
        svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_tr, y_train)
        y_pred = svm.predict(X_te)
        
        acc = accuracy_score(y_test, y_pred)
        
        # Per-method
        per_method = {}
        for method in np.unique(methods_test):
            mask = methods_test == method
            per_method[method] = accuracy_score(y_test[mask], y_pred[mask])
        
        print(f"  Overall: {acc:.2%}")
        for m, a in per_method.items():
            print(f"  {m}: {a:.2%}")
        
        results[name] = {"acc": acc, "per_method": per_method}
    
    delta = results["RGB+FFT"]["acc"] - results["RGB"]["acc"]
    print(f"\n🎯 FFT Impact: {delta:+.2%}")
    
    # PaCMAP
    print("\n" + "="*50)
    print("GENERATING PACMAP VISUALIZATION")
    print("="*50)
    
    print("Computing PaCMAP for RGB...")
    pacmap_rgb = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj_rgb = pacmap_rgb.fit_transform(rgb_test)
    
    print("Computing PaCMAP for RGB+FFT...")
    pacmap_fft = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj_fft = pacmap_fft.fit_transform(rgb_fft_test)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {"Real": "#2ecc71", "Midjourney": "#e74c3c", "StableDiffusion": "#3498db"}
    
    for ax, proj, title in zip(axes, [proj_rgb, proj_fft], 
                                ["RGB Embeddings", "RGB + FFT"]):
        for method in colors.keys():
            mask = methods_test == method
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1],
                          c=colors[method], label=method,
                          alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("GenImage: PaCMAP (Format-Normalized)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pacmap_rgb_vs_fft.png", dpi=200)
    print(f"✓ Saved: pacmap_rgb_vs_fft.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS (Format-Normalized)")
    print("="*70)
    
    for name, res in results.items():
        print(f"\n{name}: {res['acc']:.2%}")
        for m, a in res['per_method'].items():
            print(f"  {m}: {a:.2%}")
    
    if delta > 0:
        print(f"\n✅ FFT HELPS! +{delta:.2%}")
    elif delta < 0:
        print(f"\n⚠️ FFT hurts: {delta:.2%}")
    else:
        print(f"\n➖ FFT neutral")
    
    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
