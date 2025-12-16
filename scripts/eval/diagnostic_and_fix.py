"""
diagnostic_and_fix.py - Diagnoza i naprawa problemów

1. Sprawdzenie data leakage
2. Lepsza implementacja FFT (high-frequency artifacts)
3. Więcej klasyfikatorów: RF, XGBoost, MLP, GradientBoosting
4. Analiza per-method accuracy
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = Path("./results/diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 400
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# IMPROVED FFT - High Frequency Artifacts Detection
# ============================================================================

def extract_fft_improved(image: Image.Image) -> np.ndarray:
    """
    Improved FFT feature extraction focusing on:
    1. High-frequency energy ratio (gdzie GAN artifacts się pojawiają)
    2. Radial spectrum w różnych pasmach
    3. Azimuthal variance (checkerboard detection)
    """
    img = np.array(image.convert('L'), dtype=np.float32) / 255.0
    
    # 2D FFT
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    features = []
    
    # 1. Total energy
    total_energy = np.sum(magnitude ** 2)
    features.append(np.log1p(total_energy))
    
    # 2. High/Low frequency ratio (key for GAN detection!)
    # Low freq: center 25%
    # High freq: outer 75%
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)
    
    low_mask = r < max_r * 0.25
    mid_mask = (r >= max_r * 0.25) & (r < max_r * 0.5)
    high_mask = r >= max_r * 0.5
    
    low_energy = np.sum(magnitude[low_mask] ** 2)
    mid_energy = np.sum(magnitude[mid_mask] ** 2)
    high_energy = np.sum(magnitude[high_mask] ** 2)
    
    # Ratios
    features.append(np.log1p(high_energy) / (np.log1p(low_energy) + 1e-8))
    features.append(np.log1p(mid_energy) / (np.log1p(low_energy) + 1e-8))
    features.append(np.log1p(high_energy) / (np.log1p(total_energy) + 1e-8))
    
    # 3. Radial bands (8 bands)
    for i in range(8):
        r_min = max_r * i / 8
        r_max = max_r * (i + 1) / 8
        band_mask = (r >= r_min) & (r < r_max)
        band_energy = np.sum(magnitude[band_mask] ** 2)
        features.append(np.log1p(band_energy))
    
    # 4. Quadrant asymmetry (GAN artifacts often asymmetric)
    q1 = magnitude[:cy, :cx].sum()
    q2 = magnitude[:cy, cx:].sum()
    q3 = magnitude[cy:, :cx].sum()
    q4 = magnitude[cy:, cx:].sum()
    
    features.append(np.std([q1, q2, q3, q4]) / (np.mean([q1, q2, q3, q4]) + 1e-8))
    
    # 5. Peak detection (GAN often has periodic peaks)
    log_mag = np.log1p(magnitude)
    mean_mag = log_mag.mean()
    std_mag = log_mag.std()
    peaks = (log_mag > mean_mag + 3 * std_mag).sum()
    features.append(peaks / (h * w))
    
    # 6. DC component ratio
    dc = magnitude[cy, cx]
    features.append(dc / (total_energy + 1e-8))
    
    features = np.array(features, dtype=np.float32)
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features


# ============================================================================
# DATA LOADING WITH HASH CHECK
# ============================================================================

def load_data_with_hashes():
    """Load data and compute hashes to detect duplicates."""
    DATA_ROOT = Path("./data")
    images, labels, methods, hashes = [], [], [], []
    
    def get_image_hash(img):
        return hashlib.md5(np.array(img).tobytes()).hexdigest()
    
    # Real
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    if real_path.exists():
        files = list(real_path.glob("*.jpg"))[:MAX_PER_CLASS]
        print(f"Loading Real: {len(files)}")
        for p in tqdm(files, desc="Real"):
            try:
                img = Image.open(p).convert("RGB").resize((224, 224))
                images.append(img)
                labels.append(1)
                methods.append("Real")
                hashes.append(get_image_hash(img))
            except:
                continue
    
    # Fakes
    fake_sources = {
        "Inpainting": DATA_ROOT / "DeepFakeFace/_temp_inpainting",
        "Insight": DATA_ROOT / "DeepFakeFace/_temp_insight",
        "Text2Img": DATA_ROOT / "DeepFakeFace/_temp_text2img",
        "Wiki": DATA_ROOT / "DeepFakeFace/_temp_wiki",
    }
    
    for method_name, path in fake_sources.items():
        if path.exists():
            files = list(path.rglob("*.jpg"))[:MAX_PER_CLASS]
            print(f"Loading {method_name}: {len(files)}")
            for p in tqdm(files, desc=method_name):
                try:
                    img = Image.open(p).convert("RGB").resize((224, 224))
                    images.append(img)
                    labels.append(0)
                    methods.append(method_name)
                    hashes.append(get_image_hash(img))
                except:
                    continue
    
    return images, np.array(labels), np.array(methods), hashes


# ============================================================================
# MAIN DIAGNOSTIC
# ============================================================================

def main():
    print("="*70)
    print("🔍 DIAGNOSTIC & FIX")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Load data with hashes
    print("\n" + "="*50)
    print("1. LOADING DATA & CHECKING FOR DUPLICATES")
    print("="*50)
    
    images, labels, methods, hashes = load_data_with_hashes()
    print(f"\nTotal: {len(images)} images")
    
    # Check duplicates
    unique_hashes = set(hashes)
    duplicates = len(hashes) - len(unique_hashes)
    print(f"Unique images: {len(unique_hashes)}")
    print(f"Duplicates: {duplicates}")
    
    if duplicates > 0:
        print("⚠️ WARNING: Duplicates found!")
    else:
        print("✅ No duplicates in dataset")
    
    # 2. Extract embeddings
    print("\n" + "="*50)
    print("2. EXTRACTING EMBEDDINGS")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    rgb_embeddings = encoder.encode_batch(images, batch_size=32, show_progress=True)
    print(f"RGB shape: {rgb_embeddings.shape}")
    
    # 3. Extract IMPROVED FFT
    print("\n" + "="*50)
    print("3. EXTRACTING IMPROVED FFT FEATURES")
    print("="*50)
    
    fft_features = np.array([extract_fft_improved(img) for img in tqdm(images, desc="FFT")])
    print(f"FFT shape: {fft_features.shape}")
    print(f"FFT features: {fft_features.shape[1]} dimensions")
    
    # Combine
    rgb_fft = np.hstack([rgb_embeddings, fft_features])
    print(f"RGB+FFT shape: {rgb_fft.shape}")
    
    # 4. Stratified split with leakage check
    print("\n" + "="*50)
    print("4. TRAIN/TEST SPLIT WITH LEAKAGE CHECK")
    print("="*50)
    
    train_idx, test_idx = train_test_split(
        np.arange(len(images)), test_size=0.3, random_state=RANDOM_STATE, stratify=methods
    )
    
    # Check for hash overlap
    train_hashes = set([hashes[i] for i in train_idx])
    test_hashes = set([hashes[i] for i in test_idx])
    overlap = train_hashes & test_hashes
    
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Hash overlap between train/test: {len(overlap)}")
    
    if len(overlap) > 0:
        print("🚨 DATA LEAKAGE DETECTED!")
    else:
        print("✅ No data leakage")
    
    # 5. Test ALL classifiers
    print("\n" + "="*50)
    print("5. TESTING CLASSIFIERS")
    print("="*50)
    
    classifiers = {
        "k-NN (k=10)": KNeighborsClassifier(n_neighbors=10, metric='cosine', n_jobs=-1),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boost": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=RANDOM_STATE),
        "Logistic Reg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }
    
    results = []
    
    for feature_name, embeddings in [("RGB", rgb_embeddings), ("RGB+FFT", rgb_fft)]:
        print(f"\n--- {feature_name} ({embeddings.shape[1]} dim) ---")
        
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        methods_test = methods[test_idx]
        
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            overall_acc = accuracy_score(y_test, y_pred)
            
            # Per-method accuracy
            per_method = {}
            for method in np.unique(methods_test):
                mask = methods_test == method
                per_method[method] = accuracy_score(y_test[mask], y_pred[mask])
            
            results.append({
                "features": feature_name,
                "classifier": clf_name,
                "overall": overall_acc,
                **per_method
            })
            
            print(f"  {clf_name}: {overall_acc:.2%} | Text2Img: {per_method.get('Text2Img', 0):.2%}")
    
    # 6. Summary
    print("\n" + "="*70)
    print("📊 FULL RESULTS TABLE")
    print("="*70)
    
    print("\n{:<12} {:<18} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Features", "Classifier", "Overall", "Real", "Inpaint", "Text2Img", "Wiki"
    ))
    print("-"*80)
    
    for r in results:
        print("{:<12} {:<18} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%}".format(
            r["features"],
            r["classifier"],
            r["overall"],
            r.get("Real", 0),
            r.get("Inpainting", 0),
            r.get("Text2Img", 0),
            r.get("Wiki", 0),
        ))
    
    # Best for Text2Img
    text2img_best = max(results, key=lambda x: x.get("Text2Img", 0))
    overall_best = max(results, key=lambda x: x["overall"])
    
    print("\n" + "="*70)
    print(f"🏆 BEST OVERALL: {overall_best['features']} + {overall_best['classifier']} = {overall_best['overall']:.2%}")
    print(f"🎯 BEST TEXT2IMG: {text2img_best['features']} + {text2img_best['classifier']} = {text2img_best.get('Text2Img', 0):.2%}")
    print("="*70)
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
