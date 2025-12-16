"""
hypothesis_tester.py - Framework do testowania hipotez matematycznych

Umożliwia testowanie różnych hipotez matematycznych dotyczących detekcji deepfake'ów.
Każda hipoteza jest implementowana jako feature extractor i może być łączona z baseline (RGB).

Hipotezy do przetestowania:
1. Quantization Scaling - prawo skali wrażliwości na kwantyzację
2. [Następne hipotezy będą tutaj dodawane]

Workflow:
- Każda hipoteza = osobny feature extractor
- Można testować: RGB, RGB+H1, RGB+H2, RGB+H1+H2, etc.
- Po dodaniu wszystkich hipotez - benchmark wszystkich kombinacji
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.quantization_scaling import extract_batch_quantization_scaling
from deepfake_guard.features.degradation_commutator import extract_batch_degradation_invariance


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/hypothesis_testing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN = 5000
MAX_TEST = 2000
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# DATA LOADING (z leakage fix)
# ============================================================================

import hashlib

def get_hash(path):
    try:
        img = Image.open(path)
        return hashlib.md5(np.array(img).tobytes()).hexdigest()
    except:
        return None


def load_data_fixed():
    """Load CIFAKE data with leakage removal."""
    print("=" * 50)
    print("LOADING DATA WITH LEAKAGE FIX")
    print("=" * 50)
    
    # Hash train files
    print("\n1. Hashing train files...")
    train_hashes = set()
    train_files = {"REAL": [], "FAKE": []}
    
    for cls in ["REAL", "FAKE"]:
        files = list((DATA_DIR / "train" / cls).glob("*.jpg"))[:MAX_TRAIN]
        for p in tqdm(files, desc=f"Train {cls}"):
            h = get_hash(p)
            if h and h not in train_hashes:
                train_hashes.add(h)
                train_files[cls].append(p)
    
    print(f"   Train REAL: {len(train_files['REAL'])}")
    print(f"   Train FAKE: {len(train_files['FAKE'])}")
    
    # Load test files (excluding overlaps)
    print("\n2. Loading test files (excluding overlaps)...")
    test_files = {"REAL": [], "FAKE": []}
    overlaps = 0
    
    for cls in ["REAL", "FAKE"]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:MAX_TEST]
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
# HYPOTHESIS FEATURE EXTRACTORS
# ============================================================================

class HypothesisExtractor:
    """Base class dla extractorów hipotez."""
    
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
    
    def extract(self, encoder, images: List[Image.Image], show_progress: bool = True) -> np.ndarray:
        """
        Ekstraktuje cechy dla batch'a obrazów.
        
        Returns:
            Array (N, dim)
        """
        raise NotImplementedError


class QuantizationScalingExtractor(HypothesisExtractor):
    """Hipoteza #1: Prawo skali wrażliwości na kwantyzację."""
    
    def __init__(self, deltas: List[int] = None):
        super().__init__("Quantization_Scaling", 8)
        self.deltas = deltas or [4, 8, 16, 32, 64]
    
    def extract(self, encoder, images: List[Image.Image], show_progress: bool = True) -> np.ndarray:
        return extract_batch_quantization_scaling(encoder, images, self.deltas, show_progress)


class DegradationInvarianceExtractor(HypothesisExtractor):
    """Hipoteza #2: Commutator Energy + Loop Holonomy."""
    
    def __init__(self):
        super().__init__("Degradation_Invariance", 18)  # 10 commutator + 8 holonomy
    
    def extract(self, encoder, images: List[Image.Image], show_progress: bool = True) -> np.ndarray:
        return extract_batch_degradation_invariance(encoder, images, show_progress=show_progress)


# Rejestr wszystkich hipotez
HYPOTHESIS_REGISTRY = {
    'quant_scaling': QuantizationScalingExtractor(),
    'degrad_invariance': DegradationInvarianceExtractor(),
    # Kolejne hipotezy będą dodawane tutaj
}


# ============================================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_features_with_hypotheses(
    encoder,
    images: List[Image.Image],
    hypothesis_names: List[str] = None,
    include_rgb: bool = True
) -> Dict[str, np.ndarray]:
    """
    Ekstraktuje cechy dla różnych kombinacji hipotez.
    
    Args:
        encoder: Encoder object
        images: Lista obrazów
        hypothesis_names: Lista nazw hipotez do użycia (None = wszystkie)
        include_rgb: Czy dołączyć RGB embeddings
    
    Returns:
        Dict: {
            'rgb': array (N, 768),
            'quant_scaling': array (N, 8),
            'rgb+quant_scaling': array (N, 776),
            ...
        }
    """
    features = {}
    
    # RGB embeddings (baseline)
    if include_rgb:
        print("\n" + "=" * 50)
        print("EXTRACTING RGB EMBEDDINGS (ViT-L/14)")
        print("=" * 50)
        rgb = encoder.encode_batch(images, batch_size=64, show_progress=True)
        features['rgb'] = rgb
        print(f"RGB shape: {rgb.shape}")
    
    # Hipotezy
    if hypothesis_names is None:
        hypothesis_names = list(HYPOTHESIS_REGISTRY.keys())
    
    for hyp_name in hypothesis_names:
        if hyp_name not in HYPOTHESIS_REGISTRY:
            print(f"⚠️  Hypothesis '{hyp_name}' not found in registry, skipping...")
            continue
        
        extractor = HYPOTHESIS_REGISTRY[hyp_name]
        
        print("\n" + "=" * 50)
        print(f"EXTRACTING: {extractor.name}")
        print("=" * 50)
        
        hyp_features = extractor.extract(encoder, images, show_progress=True)
        features[hyp_name] = hyp_features
        print(f"{extractor.name} shape: {hyp_features.shape}")
        
        # Kombinacja RGB + Hipoteza
        if include_rgb:
            combined_key = f"rgb+{hyp_name}"
            combined = np.hstack([rgb, hyp_features])
            features[combined_key] = combined
            print(f"RGB+{extractor.name} shape: {combined.shape}")
    
    return features


# ============================================================================
# BENCHMARK (będzie uruchomiony później)
# ============================================================================

def benchmark_hypotheses(
    train_features: Dict[str, np.ndarray],
    test_features: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Benchmarkuje wszystkie kombinacje hipotez używając SVM.
    
    Returns:
        Dict: {
            'rgb': {'acc': 0.95, 'real': 0.94, 'fake': 0.96},
            'rgb+quant_scaling': {...},
            ...
        }
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    print("\n" + "=" * 70)
    print("🔬 BENCHMARKING ALL HYPOTHESES")
    print("=" * 70)
    
    results = {}
    
    for name in sorted(train_features.keys()):
        print(f"\n{name}:")
        print(f"  Feature dim: {train_features[name].shape[1]}")
        
        X_train = train_features[name]
        X_test = test_features[name]
        
        # SVM
        svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        real_acc = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
        fake_acc = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
        
        print(f"  Overall: {acc:.2%}")
        print(f"  Real: {real_acc:.2%}")
        print(f"  Fake: {fake_acc:.2%}")
        
        results[name] = {
            'acc': acc,
            'real': real_acc,
            'fake': fake_acc
        }
    
    return results


# ============================================================================
# VISUALIZATION (PaCMAP)
# ============================================================================

def visualize_hypotheses(
    test_features: Dict[str, np.ndarray],
    y_test: np.ndarray,
    max_samples: int = 3000
):
    """
    Tworzy wizualizacje PaCMAP dla wszystkich hipotez.
    """
    import matplotlib.pyplot as plt
    import pacmap
    
    print("\n" + "=" * 50)
    print("GENERATING PACMAP VISUALIZATIONS")
    print("=" * 50)
    
    # Subset dla wizualizacji
    indices = np.random.choice(len(y_test), min(max_samples, len(y_test)), replace=False)
    labels_subset = y_test[indices]
    
    colors = {0: "#e74c3c", 1: "#2ecc71"}
    labels_map = {0: "Fake (SD)", 1: "Real (CIFAR)"}
    
    # Dla każdej kombinacji cech
    for name, features in test_features.items():
        print(f"\nComputing PaCMAP for {name}...")
        
        features_subset = features[indices]
        
        # PaCMAP
        pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
        proj = pacmap_model.fit_transform(features_subset)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label in [0, 1]:
            mask = labels_subset == label
            ax.scatter(proj[mask, 0], proj[mask, 1],
                      c=colors[label], label=labels_map[label],
                      alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        
        ax.set_title(f"PaCMAP: {name}\n({features.shape[1]} dimensions)", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PaCMAP Dim 1")
        ax.set_ylabel("PaCMAP Dim 2")
        
        plt.tight_layout()
        
        # Save
        safe_name = name.replace('+', '_')
        plt.savefig(OUTPUT_DIR / f"pacmap_{safe_name}.png", dpi=200)
        print(f"✓ Saved: pacmap_{safe_name}.png")
        plt.close()


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """
    Główna funkcja - ekstraktuje cechy dla wszystkich hipotez.
    NIE uruchamia benchmarków - czekamy na wszystkie hipotezy.
    """
    print("=" * 70)
    print("🧪 HYPOTHESIS TESTING FRAMEWORK")
    print("   Extracting features for all hypotheses...")
    print("   (Benchmarks will run after all hypotheses are added)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    (train_images, y_train), (test_images, y_test) = load_data_fixed()
    print(f"\nFinal: Train={len(train_images)}, Test={len(test_images)}")
    
    # Initialize encoder (tylko raz!)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract features for TRAIN set
    print("\n" + "=" * 70)
    print("📊 EXTRACTING FEATURES - TRAIN SET")
    print("=" * 70)
    train_features = extract_features_with_hypotheses(
        encoder, 
        train_images,
        hypothesis_names=['quant_scaling', 'degrad_invariance'],  # dodaj kolejne hipotezy tutaj
        include_rgb=True
    )
    
    # Extract features for TEST set
    print("\n" + "=" * 70)
    print("📊 EXTRACTING FEATURES - TEST SET")
    print("=" * 70)
    test_features = extract_features_with_hypotheses(
        encoder,
        test_images,
        hypothesis_names=['quant_scaling', 'degrad_invariance'],  # dodaj kolejne hipotezy tutaj
        include_rgb=True
    )
    
    # Save extracted features
    print("\n" + "=" * 50)
    print("💾 SAVING EXTRACTED FEATURES")
    print("=" * 50)
    
    np.savez_compressed(
        OUTPUT_DIR / "train_features.npz",
        y_train=y_train,
        **train_features
    )
    
    np.savez_compressed(
        OUTPUT_DIR / "test_features.npz",
        y_test=y_test,
        **test_features
    )
    
    print(f"✓ Saved to {OUTPUT_DIR}")
    
    print("\n" + "=" * 70)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print("\nExtracted feature sets:")
    for name, feat in train_features.items():
        print(f"  - {name}: {feat.shape}")
    
    print("\n⏳ Waiting for more hypotheses...")
    print("   Run benchmarks with: run_benchmarks()")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()


def run_benchmarks():
    """
    Uruchamia benchmarki i wizualizacje dla wszystkich hipotez.
    WYWOŁAJ TO PO DODANIU WSZYSTKICH HIPOTEZ!
    """
    print("=" * 70)
    print("🚀 RUNNING BENCHMARKS FOR ALL HYPOTHESES")
    print("=" * 70)
    
    # Load saved features
    print("\nLoading saved features...")
    train_data = np.load(OUTPUT_DIR / "train_features.npz")
    test_data = np.load(OUTPUT_DIR / "test_features.npz")
    
    y_train = train_data['y_train']
    y_test = test_data['y_test']
    
    train_features = {k: train_data[k] for k in train_data.files if k != 'y_train'}
    test_features = {k: test_data[k] for k in test_data.files if k != 'y_test'}
    
    # Benchmark
    results = benchmark_hypotheses(train_features, test_features, y_train, y_test)
    
    # Visualize
    visualize_hypotheses(test_features, y_test)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n{:<25} {:<10} {:<10} {:<10}".format("Features", "Overall", "Real", "Fake"))
    print("-" * 55)
    for name, res in sorted(results.items()):
        print("{:<25} {:<10.2%} {:<10.2%} {:<10.2%}".format(
            name, res["acc"], res["real"], res["fake"]))
    
    # Improvements
    if 'rgb' in results:
        print("\n" + "=" * 70)
        print("📈 IMPROVEMENTS OVER BASELINE (RGB)")
        print("=" * 70)
        
        baseline_acc = results['rgb']['acc']
        
        for name, res in sorted(results.items()):
            if name != 'rgb':
                delta = res['acc'] - baseline_acc
                symbol = "✅" if delta > 0 else "⚠️" if delta < 0 else "➖"
                print(f"{symbol} {name}: {delta:+.2%}")
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
