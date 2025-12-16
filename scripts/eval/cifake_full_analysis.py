"""
cifake_full_analysis.py - Pełna analiza CIFAKE

1. Naprawa leakage (usunięcie overlapping obrazów)
2. Porównanie SVM z FFT i bez FFT
3. PaCMAP wizualizacja dla RGB i RGB+FFT
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import hashlib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pacmap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/cifake_full_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN = 5000
MAX_TEST = 2000
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
    
    # Total energy
    total_energy = np.sum(magnitude ** 2)
    features.append(np.log1p(total_energy))
    
    # Frequency band ratios
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
    
    # Radial bands (8 bands)
    for i in range(8):
        r_min = max_r * i / 8
        r_max = max_r * (i + 1) / 8
        band_mask = (r >= r_min) & (r < r_max)
        band_energy = np.sum(magnitude[band_mask] ** 2)
        features.append(np.log1p(band_energy))
    
    # Quadrant asymmetry
    q1 = magnitude[:cy, :cx].sum()
    q2 = magnitude[:cy, cx:].sum()
    q3 = magnitude[cy:, :cx].sum()
    q4 = magnitude[cy:, cx:].sum()
    features.append(np.std([q1, q2, q3, q4]) / (np.mean([q1, q2, q3, q4]) + 1e-8))
    
    # Peak count
    log_mag = np.log1p(magnitude)
    peaks = (log_mag > log_mag.mean() + 3 * log_mag.std()).sum()
    features.append(peaks / (h * w))
    
    # DC ratio
    dc = magnitude[cy, cx]
    features.append(dc / (total_energy + 1e-8))
    
    features = np.array(features, dtype=np.float32)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


# ============================================================================
# DATA LOADING WITH LEAKAGE FIX
# ============================================================================

def get_hash(path):
    try:
        img = Image.open(path)
        return hashlib.md5(np.array(img).tobytes()).hexdigest()
    except:
        return None


def load_data_fixed():
    """Load data with leakage removal."""
    print("="*50)
    print("LOADING DATA WITH LEAKAGE FIX")
    print("="*50)
    
    # First, hash all train files
    print("\n1. Hashing train files...")
    train_hashes = set()
    train_files = {"REAL": [], "FAKE": []}
    
    for cls in ["REAL", "FAKE"]:
        files = list((DATA_DIR/"train"/cls).glob("*.jpg"))[:MAX_TRAIN]
        for p in tqdm(files, desc=f"Train {cls}"):
            h = get_hash(p)
            if h and h not in train_hashes:
                train_hashes.add(h)
                train_files[cls].append(p)
    
    print(f"   Train REAL: {len(train_files['REAL'])}")
    print(f"   Train FAKE: {len(train_files['FAKE'])}")
    
    # Load test files, excluding any in train
    print("\n2. Loading test files (excluding overlaps)...")
    test_files = {"REAL": [], "FAKE": []}
    overlaps = 0
    
    for cls in ["REAL", "FAKE"]:
        files = list((DATA_DIR/"test"/cls).glob("*.jpg"))[:MAX_TEST]
        for p in tqdm(files, desc=f"Test {cls}"):
            h = get_hash(p)
            if h and h not in train_hashes:
                test_files[cls].append(p)
            elif h in train_hashes:
                overlaps += 1
    
    print(f"   Test REAL: {len(test_files['REAL'])}")
    print(f"   Test FAKE: {len(test_files['FAKE'])}")
    print(f"   Removed overlaps: {overlaps}")
    
    # Load images
    print("\n3. Loading images...")
    
    train_images, train_labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        for p in tqdm(train_files[cls], desc=f"Load Train {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            train_images.append(img)
            train_labels.append(label)
    
    test_images, test_labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        for p in tqdm(test_files[cls], desc=f"Load Test {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            test_images.append(img)
            test_labels.append(label)
    
    return (train_images, np.array(train_labels)), (test_images, np.array(test_labels))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 CIFAKE FULL ANALYSIS")
    print("   Leakage Fix | FFT Comparison | PaCMAP Visualization")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data with fix
    (train_images, y_train), (test_images, y_test) = load_data_fixed()
    print(f"\nFinal: Train={len(train_images)}, Test={len(test_images)}")
    
    # Extract RGB embeddings
    print("\n" + "="*50)
    print("EXTRACTING RGB EMBEDDINGS (ViT-L/14)")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    rgb_train = encoder.encode_batch(train_images, batch_size=64, show_progress=True)
    rgb_test = encoder.encode_batch(test_images, batch_size=64, show_progress=True)
    
    print(f"RGB shape: {rgb_train.shape}")
    
    # Extract FFT features
    print("\n" + "="*50)
    print("EXTRACTING FFT FEATURES")
    print("="*50)
    
    fft_train = np.array([extract_fft_improved(img) for img in tqdm(train_images, desc="FFT Train")])
    fft_test = np.array([extract_fft_improved(img) for img in tqdm(test_images, desc="FFT Test")])
    
    print(f"FFT shape: {fft_train.shape}")
    
    # Combine
    rgb_fft_train = np.hstack([rgb_train, fft_train])
    rgb_fft_test = np.hstack([rgb_test, fft_test])
    
    print(f"RGB+FFT shape: {rgb_fft_train.shape}")
    
    # ========================================================================
    # BENCHMARK: SVM with/without FFT
    # ========================================================================
    
    print("\n" + "="*50)
    print("BENCHMARK: SVM RGB vs SVM RGB+FFT")
    print("="*50)
    
    results = {}
    
    for name, (X_tr, X_te) in [("RGB", (rgb_train, rgb_test)), 
                                ("RGB+FFT", (rgb_fft_train, rgb_fft_test))]:
        print(f"\n{name}:")
        
        svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_tr, y_train)
        y_pred = svm.predict(X_te)
        
        acc = accuracy_score(y_test, y_pred)
        real_acc = accuracy_score(y_test[y_test==1], y_pred[y_test==1])
        fake_acc = accuracy_score(y_test[y_test==0], y_pred[y_test==0])
        
        print(f"  Overall: {acc:.2%}")
        print(f"  Real: {real_acc:.2%}")
        print(f"  Fake: {fake_acc:.2%}")
        
        results[name] = {"acc": acc, "real": real_acc, "fake": fake_acc}
    
    # FFT improvement
    delta = results["RGB+FFT"]["acc"] - results["RGB"]["acc"]
    print(f"\nFFT improvement: {delta:+.2%}")
    
    # ========================================================================
    # PACMAP VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*50)
    print("GENERATING PACMAP VISUALIZATIONS")
    print("="*50)
    
    # Combine train+test for visualization (use subset)
    viz_size = 3000
    indices = np.random.choice(len(test_images), min(viz_size, len(test_images)), replace=False)
    
    viz_rgb = rgb_test[indices]
    viz_fft = rgb_fft_test[indices]
    viz_labels = y_test[indices]
    
    # PaCMAP RGB
    print("Computing PaCMAP for RGB...")
    pacmap_rgb = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj_rgb = pacmap_rgb.fit_transform(viz_rgb)
    
    # PaCMAP RGB+FFT
    print("Computing PaCMAP for RGB+FFT...")
    pacmap_fft = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj_fft = pacmap_fft.fit_transform(viz_fft)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {0: "#e74c3c", 1: "#2ecc71"}
    labels_map = {0: "Fake (SD)", 1: "Real (CIFAR)"}
    
    for ax, proj, title in zip(axes, [proj_rgb, proj_fft], 
                                ["RGB Embeddings (768 dim)", "RGB + FFT (783 dim)"]):
        for label in [0, 1]:
            mask = viz_labels == label
            ax.scatter(proj[mask, 0], proj[mask, 1],
                      c=colors[label], label=labels_map[label],
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PaCMAP Dim 1")
        ax.set_ylabel("PaCMAP Dim 2")
    
    plt.suptitle("CIFAKE: PaCMAP Comparison - RGB vs RGB+FFT", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pacmap_rgb_vs_fft.png", dpi=200)
    print(f"✓ Saved: pacmap_rgb_vs_fft.png")
    plt.close()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS (after leakage fix)")
    print("="*70)
    
    print("\n{:<12} {:<10} {:<10} {:<10}".format("Features", "Overall", "Real", "Fake"))
    print("-"*42)
    for name, res in results.items():
        print("{:<12} {:<10.2%} {:<10.2%} {:<10.2%}".format(
            name, res["acc"], res["real"], res["fake"]))
    
    print(f"\n🎯 FFT Impact: {delta:+.2%}")
    
    if delta > 0:
        print("✅ FFT POMAGA na tym datasecie!")
    elif delta < 0:
        print("⚠️ FFT pogarsza wyniki - obrazy 32x32 mogą mieć za mało informacji freq.")
    else:
        print("➖ FFT nie ma wpływu")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print(f"\nWyniki zapisane w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
