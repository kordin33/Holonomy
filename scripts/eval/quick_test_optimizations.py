"""
quick_test_optimizations.py - SZYBKI test wszystkich optymalizacji

JEDEN encoder, ekstraktuje WSZYSTKIE cechy naraz, potem testuje każdą.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.degradation_commutator_optimized import extract_all_optimized_features_v2

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/optimization_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 150  # Zmniejszam do 150 dla szybkości
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:n_per_class]
        for p in tqdm(files, desc=f"Loading {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    return images, np.array(labels)


def test_features(features, labels, name):
    """Szybki test SVM."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE, probability=True)
    svm.fit(X_train, y_train)
    
    y_prob = svm.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, svm.predict(X_test))
    
    return {'name': name, 'auc': auc, 'acc': acc, 'shape': features.shape}


def main():
    print("="*70)
    print("🚀 QUICK OPTIMIZATION TEST")
    print("   One encoder, all features extracted together!")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data(SAMPLE_SIZE)
    print(f"Total: {len(images)} images")
    
    # Encoder
    print("\nInitializing CLIP ViT-L/14...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract ALL features ONCE
    print("\n" + "="*70)
    print("EXTRACTING ALL FEATURES (one pass)")
    print("="*70)
    
    all_features = {
        'baseline_holonomy': [],
        'trajectory_features': [],
        'patchwise_holonomy': [],
        'commutator': [],
        'combined': []
    }
    
    for img in tqdm(images, desc="Extracting"):
        feats = extract_all_optimized_features_v2(encoder, img)
        
        for key in all_features.keys():
            all_features[key].append(feats[key])
    
    # Convert to arrays
    for key in all_features.keys():
        all_features[key] = np.array(all_features[key], dtype=np.float32)
        print(f"  {key}: {all_features[key].shape}")
    
    # Test each
    print("\n" + "="*70)
    print("TESTING EACH FEATURE SET")
    print("="*70)
    
    results = []
    
    for feat_name, display_name in [
        ('baseline_holonomy', 'Baseline (H_raw)'),
        ('trajectory_features', 'Trajectory'),
        ('patchwise_holonomy', 'Patchwise'),
        ('commutator', 'Commutator'),
        ('combined', 'Combined')
    ]:
        print(f"\n🔬 Testing: {display_name}")
        res = test_features(all_features[feat_name], labels, display_name)
        results.append(res)
        
        symbol = "✅" if res['auc'] > 0.74 else "⚠️" if res['auc'] > 0.65 else "❌"
        print(f"  {symbol} AUC: {res['auc']:.4f}, Acc: {res['acc']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<25} {'AUC':<8} {'Acc':<8} {'Shape':<12}")
    print("-"*70)
    
    baseline_auc = results[0]['auc']
    
    for res in results:
        delta = res['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "❌"
        
        print(f"{res['name']:<25} {res['auc']:.4f}  {res['acc']:.4f}  {str(res['shape']):<12} {symbol} {delta:+.4f}")
    
    # Best
    best = max(results, key=lambda x: x['auc'])
    print(f"\n🏆 BEST: {best['name']} (AUC={best['auc']:.4f})")
    
    # Save
    print("\n💾 Saving results...")
    np.savez_compressed(
        OUTPUT_DIR / "quick_test_results.npz",
        labels=labels,
        **all_features
    )
    
    import json
    with open(OUTPUT_DIR / "quick_test_summary.json", 'w') as f:
        results_json = []
        for r in results:
            r_copy = r.copy()
            r_copy['shape'] = list(r['shape'])
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Saved to {OUTPUT_DIR}/")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
