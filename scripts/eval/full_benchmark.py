"""
full_benchmark.py - Pełny benchmark wszystkich kombinacji

Kombinacje:
- Encodery: ViT-B/32, ViT-L/14
- Features: RGB, RGB+FFT
- Klasyfikatory: k-NN, SVM

= 2 x 2 x 2 = 8 kombinacji

Wyniki zapisywane do results/benchmark_full/
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.utils.cuda_utils import setup_cuda_optimizations, clear_cuda_cache

# ============================================================================
# CONFIG
# ============================================================================

DATA_ROOT = Path("./data")
RESULTS_DIR = Path("./results/benchmark_full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES_PER_CLASS = 400
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# FFT FEATURE EXTRACTION
# ============================================================================

def extract_fft_features(image: Image.Image, feature_dim: int = 64) -> np.ndarray:
    """Ekstrakcja cech z domeny częstotliwości."""
    img_gray = np.array(image.convert('L'), dtype=np.float32)
    
    fft = np.fft.fft2(img_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)
    
    h, w = log_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    max_r = min(center_x, center_y)
    radial_profile = np.zeros(max_r)
    
    for i in range(max_r):
        mask = r == i
        if mask.sum() > 0:
            radial_profile[i] = log_magnitude[mask].mean()
    
    if len(radial_profile) > feature_dim:
        indices = np.linspace(0, len(radial_profile)-1, feature_dim).astype(int)
        features = radial_profile[indices]
    else:
        features = np.zeros(feature_dim)
        features[:len(radial_profile)] = radial_profile
    
    features = (features - features.mean()) / (features.std() + 1e-8)
    return features.astype(np.float32)


def extract_fft_batch(images: list, feature_dim: int = 64) -> np.ndarray:
    """Batch FFT extraction."""
    features = []
    for img in tqdm(images, desc="FFT extraction"):
        features.append(extract_fft_features(img, feature_dim))
    return np.array(features)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load all data from all methods."""
    images, labels, methods = [], [], []
    
    # Real
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    if real_path.exists():
        files = list(real_path.glob("*.jpg"))[:MAX_IMAGES_PER_CLASS]
        print(f"Loading Real: {len(files)}")
        for p in tqdm(files, desc="Real"):
            try:
                images.append(Image.open(p).convert("RGB").resize((224, 224)))
                labels.append(1)  # 1 = real
                methods.append("real")
            except:
                continue
    
    # Fakes
    fake_sources = {
        "inpainting": DATA_ROOT / "DeepFakeFace/_temp_inpainting",
        "insight": DATA_ROOT / "DeepFakeFace/_temp_insight",
        "text2img": DATA_ROOT / "DeepFakeFace/_temp_text2img",
        "wiki": DATA_ROOT / "DeepFakeFace/_temp_wiki",
    }
    
    for method_name, path in fake_sources.items():
        if path.exists():
            files = list(path.rglob("*.jpg"))[:MAX_IMAGES_PER_CLASS]
            print(f"Loading {method_name}: {len(files)}")
            for p in tqdm(files, desc=method_name):
                try:
                    images.append(Image.open(p).convert("RGB").resize((224, 224)))
                    labels.append(0)  # 0 = fake
                    methods.append(method_name)
                except:
                    continue
    
    return images, np.array(labels), np.array(methods)


# ============================================================================
# BENCHMARK
# ============================================================================

def run_full_benchmark():
    print("="*70)
    print("🔬 FULL BENCHMARK: ViT-B/32 vs ViT-L/14 x RGB vs RGB+FFT x k-NN vs SVM")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # CUDA optimization
    if device == "cuda":
        setup_cuda_optimizations(use_tf32=True, use_cudnn_benchmark=True, verbose=True)
    
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    images, labels, methods = load_all_data()
    print(f"\nTotal: {len(images)} images")
    
    # Split
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=RANDOM_STATE, stratify=methods
    )
    
    train_images = [images[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    methods_test = methods[test_idx]
    
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")
    
    # Pre-compute FFT for all images
    print("\n" + "="*50)
    print("PRE-COMPUTING FFT FEATURES")
    print("="*50)
    
    fft_train = extract_fft_batch(train_images)
    fft_test = extract_fft_batch(test_images)
    
    # Results storage
    all_results = []
    
    # Encoders
    encoders = [
        ("ViT-B/32", "clip", "ViT-B/32"),
        ("ViT-L/14", "clip", "ViT-L/14"),
    ]
    
    # Classifiers
    classifiers = {
        "kNN": lambda: KNeighborsClassifier(n_neighbors=10, metric='cosine', n_jobs=-1),
        "SVM": lambda: SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
    }
    
    for encoder_name, encoder_type, variant in encoders:
        print("\n" + "="*70)
        print(f"ENCODER: {encoder_name}")
        print("="*70)
        
        encoder = get_encoder(encoder_type, variant, device)
        
        # Extract RGB embeddings
        print("\nExtracting RGB embeddings...")
        rgb_train = encoder.encode_batch(train_images, batch_size=32, show_progress=True)
        rgb_test = encoder.encode_batch(test_images, batch_size=32, show_progress=True)
        
        # Create feature sets
        feature_sets = {
            "RGB": (rgb_train, rgb_test),
            "RGB+FFT": (np.hstack([rgb_train, fft_train]), np.hstack([rgb_test, fft_test])),
        }
        
        for feature_name, (X_train, X_test) in feature_sets.items():
            print(f"\n--- Features: {feature_name} (dim={X_train.shape[1]}) ---")
            
            for clf_name, clf_factory in classifiers.items():
                print(f"\n  Classifier: {clf_name}")
                
                clf = clf_factory()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                # Overall metrics
                overall_acc = accuracy_score(y_test, y_pred)
                overall_f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                print(f"    Overall Accuracy: {overall_acc:.4f}")
                print(f"    Overall F1: {overall_f1:.4f}")
                
                # Per-method accuracy
                per_method = {}
                for method in np.unique(methods_test):
                    mask = methods_test == method
                    if mask.sum() > 0:
                        method_acc = accuracy_score(y_test[mask], y_pred[mask])
                        per_method[method] = round(method_acc, 4)
                        print(f"    {method}: {method_acc:.4f}")
                
                # Store result
                result = {
                    "encoder": encoder_name,
                    "features": feature_name,
                    "classifier": clf_name,
                    "feature_dim": X_train.shape[1],
                    "overall_accuracy": round(overall_acc, 4),
                    "overall_f1": round(overall_f1, 4),
                    "confusion_matrix": cm.tolist(),
                    "per_method": per_method,
                }
                all_results.append(result)
        
        # Cleanup
        del encoder
        clear_cuda_cache()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")
    
    # Summary table
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n{:<12} {:<10} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Encoder", "Features", "Clf", "Overall", "Real", "Inpaint", "Text2Img", "Wiki"
    ))
    print("-"*80)
    
    for r in all_results:
        pm = r["per_method"]
        print("{:<12} {:<10} {:<8} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%}".format(
            r["encoder"],
            r["features"],
            r["classifier"],
            r["overall_accuracy"],
            pm.get("real", 0),
            pm.get("inpainting", 0),
            pm.get("text2img", 0),
            pm.get("wiki", 0),
        ))
    
    # Find best config
    best = max(all_results, key=lambda x: x["overall_accuracy"])
    print("\n" + "="*80)
    print(f"🏆 BEST CONFIG: {best['encoder']} + {best['features']} + {best['classifier']}")
    print(f"   Overall Accuracy: {best['overall_accuracy']:.2%}")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    run_full_benchmark()
