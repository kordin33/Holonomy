"""
benchmark_suite.py - Kompleksowy system benchmarkowy dla DeepfakeGuard

Porównuje:
- Encodery: CLIP ViT-B/32, CLIP ViT-L/14, DINOv2
- Features: RGB, RGB+FFT
- Klasyfikatory: k-NN, Logistic Regression, SVM, Random Forest

Trenuje na PEŁNEJ matrycy (wszystkie metody), testuje per-method.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_ROOT = Path("./data")
RESULTS_DIR = Path("./results/benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES_PER_CLASS = 500  # Dla szybszego testowania, zwiększ do 1000+ dla produkcji
RANDOM_STATE = 42

# Encodery do testowania
ENCODERS = {
    "clip_vit_b32": {"name": "clip", "variant": "ViT-B/32"},
    "clip_vit_l14": {"name": "clip", "variant": "ViT-L/14"},
    "dinov2": {"name": "dinov2", "variant": None},
}

# Klasyfikatory
CLASSIFIERS = {
    "knn": lambda: KNeighborsClassifier(n_neighbors=10, metric='cosine'),
    "logistic": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "svm": lambda: SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

# ============================================================================
# FFT FEATURE EXTRACTION
# ============================================================================

def extract_fft_features(image: Image.Image, feature_dim: int = 64) -> np.ndarray:
    """
    Ekstrakcja cech z domeny częstotliwości (FFT).
    
    Deepfake często mają charakterystyczne artefakty w wysokich częstotliwościach.
    """
    # Convert to grayscale numpy
    img_gray = np.array(image.convert('L'), dtype=np.float32)
    
    # 2D FFT
    fft = np.fft.fft2(img_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Log magnitude spectrum
    log_magnitude = np.log1p(magnitude)
    
    # Azimuthal average (radial power spectrum)
    h, w = log_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create radial bins
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    max_r = min(center_x, center_y)
    radial_profile = np.zeros(max_r)
    
    for i in range(max_r):
        mask = r == i
        if mask.sum() > 0:
            radial_profile[i] = log_magnitude[mask].mean()
    
    # Resample to fixed dimension
    if len(radial_profile) > feature_dim:
        # Downsample
        indices = np.linspace(0, len(radial_profile)-1, feature_dim).astype(int)
        features = radial_profile[indices]
    else:
        # Pad
        features = np.zeros(feature_dim)
        features[:len(radial_profile)] = radial_profile
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features.astype(np.float32)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(max_per_class: int = 500):
    """
    Ładuj dane z wszystkich metod generowania.
    
    Returns:
        images: List[PIL.Image]
        labels: List[str] - "real" lub "fake"
        methods: List[str] - metoda generowania lub "real"
    """
    images = []
    labels = []
    methods = []
    
    # Real images
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    if real_path.exists():
        real_files = list(real_path.glob("*.jpg")) + list(real_path.glob("*.png"))
        np.random.shuffle(real_files)
        
        print(f"Loading Real images ({min(len(real_files), max_per_class)})...")
        for p in tqdm(real_files[:max_per_class]):
            try:
                img = Image.open(p).convert("RGB").resize((224, 224))
                images.append(img)
                labels.append("real")
                methods.append("real")
            except:
                continue
    
    # Fake images from different methods
    fake_sources = {
        "inpainting": DATA_ROOT / "DeepFakeFace/_temp_inpainting",
        "insight": DATA_ROOT / "DeepFakeFace/_temp_insight",
        "text2img": DATA_ROOT / "DeepFakeFace/_temp_text2img",
        "wiki": DATA_ROOT / "DeepFakeFace/_temp_wiki",
    }
    
    for method_name, method_path in fake_sources.items():
        if method_path.exists():
            fake_files = list(method_path.rglob("*.jpg")) + list(method_path.rglob("*.png"))
            np.random.shuffle(fake_files)
            
            print(f"Loading {method_name} fakes ({min(len(fake_files), max_per_class)})...")
            for p in tqdm(fake_files[:max_per_class]):
                try:
                    img = Image.open(p).convert("RGB").resize((224, 224))
                    images.append(img)
                    labels.append("fake")
                    methods.append(method_name)
                except:
                    continue
    
    return images, labels, methods


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(images, encoder, use_fft: bool = False):
    """
    Ekstrahuj embeddingi z obrazów.
    
    Args:
        images: Lista PIL Images
        encoder: Encoder (CLIP lub DINOv2)
        use_fft: Czy dołączyć cechy FFT
        
    Returns:
        embeddings: np.ndarray [N, D]
    """
    print(f"Extracting embeddings (FFT={use_fft})...")
    
    # RGB embeddings
    rgb_embeddings = encoder.encode_batch(images, show_progress=True)
    
    if not use_fft:
        return rgb_embeddings
    
    # FFT features
    print("Extracting FFT features...")
    fft_features = []
    for img in tqdm(images):
        fft_feat = extract_fft_features(img, feature_dim=64)
        fft_features.append(fft_feat)
    
    fft_features = np.array(fft_features)
    
    # Concatenate [RGB | FFT]
    combined = np.hstack([rgb_embeddings, fft_features])
    
    return combined


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark():
    """Główna funkcja benchmarkowa."""
    
    print("="*70)
    print("🔬 DEEPFAKEGUARD BENCHMARK SUITE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    images, labels, methods = load_dataset(max_per_class=MAX_IMAGES_PER_CLASS)
    
    print(f"\nTotal: {len(images)} images")
    print(f"Labels: {np.unique(labels, return_counts=True)}")
    print(f"Methods: {np.unique(methods, return_counts=True)}")
    
    # Convert labels to binary
    y = np.array([1 if l == "real" else 0 for l in labels])
    methods_arr = np.array(methods)
    
    # Train/test split (stratified by method)
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=RANDOM_STATE, stratify=methods_arr
    )
    
    train_images = [images[i] for i in train_idx]
    test_images = [images[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    methods_test = methods_arr[test_idx]
    
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")
    
    # Results storage
    all_results = {}
    
    # 2. Run benchmarks for each encoder
    for encoder_key, encoder_config in ENCODERS.items():
        print("\n" + "="*70)
        print(f"ENCODER: {encoder_key.upper()}")
        print("="*70)
        
        try:
            encoder = get_encoder(
                encoder_name=encoder_config["name"],
                model_variant=encoder_config["variant"] or "vitb14",
                device=device,
            )
        except Exception as e:
            print(f"Failed to load {encoder_key}: {e}")
            continue
        
        # Test with and without FFT
        for use_fft in [False, True]:
            feature_key = f"{encoder_key}_{'fft' if use_fft else 'rgb'}"
            print(f"\n--- Features: {feature_key} ---")
            
            # Extract embeddings
            X_train = extract_embeddings(train_images, encoder, use_fft=use_fft)
            X_test = extract_embeddings(test_images, encoder, use_fft=use_fft)
            
            print(f"Feature dimension: {X_train.shape[1]}")
            
            # Store embeddings for visualization
            all_results[feature_key] = {
                "X_test": X_test,
                "y_test": y_test,
                "methods_test": methods_test,
                "classifiers": {},
            }
            
            # 3. Test each classifier
            for clf_name, clf_factory in CLASSIFIERS.items():
                print(f"\n  Classifier: {clf_name}")
                
                clf = clf_factory()
                clf.fit(X_train, y_train)
                
                # Overall prediction
                y_pred = clf.predict(X_test)
                
                overall_acc = accuracy_score(y_test, y_pred)
                overall_f1 = f1_score(y_test, y_pred)
                
                print(f"    Overall: Acc={overall_acc:.4f}, F1={overall_f1:.4f}")
                
                # Per-method accuracy
                method_results = {}
                for method in np.unique(methods_test):
                    mask = methods_test == method
                    if mask.sum() > 0:
                        method_acc = accuracy_score(y_test[mask], y_pred[mask])
                        method_results[method] = method_acc
                        print(f"    {method}: {method_acc:.4f}")
                
                all_results[feature_key]["classifiers"][clf_name] = {
                    "overall_accuracy": overall_acc,
                    "overall_f1": overall_f1,
                    "per_method": method_results,
                }
        
        # Clear GPU memory
        del encoder
        torch.cuda.empty_cache()
    
    # 4. Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Prepare JSON-serializable results
    json_results = {}
    for feature_key, data in all_results.items():
        json_results[feature_key] = {
            "feature_dim": data["X_test"].shape[1] if "X_test" in data else 0,
            "classifiers": data.get("classifiers", {}),
        }
    
    results_path = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Results saved to {results_path}")
    
    # 5. Print summary table
    print("\n" + "="*70)
    print("📊 SUMMARY TABLE")
    print("="*70)
    
    print("\n{:<25} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Config", "Classifier", "Overall", "Real", "Inpaint", "Text2Img", "Wiki"
    ))
    print("-"*100)
    
    for feature_key, data in all_results.items():
        for clf_name, clf_results in data.get("classifiers", {}).items():
            per_method = clf_results.get("per_method", {})
            print("{:<25} {:<15} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%} {:<10.2%}".format(
                feature_key,
                clf_name,
                clf_results.get("overall_accuracy", 0),
                per_method.get("real", 0),
                per_method.get("inpainting", 0),
                per_method.get("text2img", 0),
                per_method.get("wiki", 0),
            ))
    
    return all_results


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    run_benchmark()
