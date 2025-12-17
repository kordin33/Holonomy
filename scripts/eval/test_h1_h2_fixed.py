"""
test_h1_h2_fixed.py - Test NAPRAWIONYCH H1 i H2

NAPRAWY:
0. Baseline comparison z tego samego runu (nie stała!)
1. H1: PCA na holonomy vectors, whiten=True, stabilne agregaty
2. H2: area = zmierzona siła degradacji, 11 punktów
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.holonomy_h1_h2_fixed import (
    H1_HolonomySpectrum_Fixed,
    H2_AreaScaleLaw_Fixed,
)
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/h1_h2_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 150
RANDOM_STATE = 42


def load_data_cifake(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:n_per_class]
        for p in tqdm(files, desc=f"Loading {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    return images, np.array(labels)


def test_with_pipeline(features, labels, name):
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])
    
    param_grid = {
        'svm__C': [0.1, 0.3, 1, 3, 10],
        'svm__gamma': ['scale', 0.001, 0.01, 0.1]
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, grid.best_estimator_.predict(X_test))
    
    return {
        'name': name,
        'auc': auc,
        'acc': acc,
        'shape': features.shape,
        'best_params': grid.best_params_
    }


def main():
    print("="*70)
    print("🔬 TEST NAPRAWIONYCH H1 i H2")
    print("   Dataset: CIFAKE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data_cifake(SAMPLE_SIZE)
    
    # Split for PCA fitting (train only!)
    train_idx, test_idx = train_test_split(
        range(len(labels)), test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    train_images = [images[i] for i in train_idx]
    
    # Encoder
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Initialize extractors
    h1 = H1_HolonomySpectrum_Fixed(pca_dim=32, k_head=8)
    h2 = H2_AreaScaleLaw_Fixed()
    
    # Fit PCA for H1 (na holonomy vectors z train!)
    print("\n📐 Fitting H1 PCA on HOLONOMY VECTORS from train...")
    h1.fit_pca(encoder, train_images[:50])  # Użyj 50 obrazów dla szybkości
    
    # Extract features
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    # Baseline (Minimal)
    print("\n📊 Extracting: Baseline (Minimal = H_raw + shape)")
    baseline_features = []
    for img in tqdm(images, desc="Baseline"):
        feat = extract_minimal_features(encoder, img)
        baseline_features.append(feat['minimal'])
    baseline_features = np.array(baseline_features, dtype=np.float32)
    print(f"   Shape: {baseline_features.shape}")
    
    # H1
    print("\n📊 Extracting: H1 (Holonomy Spectrum - FIXED)")
    h1_features = []
    for img in tqdm(images, desc="H1"):
        try:
            feat = h1.extract_features(encoder, img)
            h1_features.append(feat)
        except Exception as e:
            print(f"H1 error: {e}")
            h1_features.append(np.zeros(15, dtype=np.float32))
    h1_features = np.array(h1_features, dtype=np.float32)
    print(f"   Shape: {h1_features.shape}")
    
    # H2
    print("\n📊 Extracting: H2 (Scale Law - FIXED)")
    h2_features = []
    for img in tqdm(images, desc="H2"):
        try:
            feat = h2.extract_features(encoder, img)
            h2_features.append(feat)
        except Exception as e:
            print(f"H2 error: {e}")
            h2_features.append(np.zeros(5, dtype=np.float32))
    h2_features = np.array(h2_features, dtype=np.float32)
    print(f"   Shape: {h2_features.shape}")
    
    # Test
    print("\n" + "="*70)
    print("TESTING WITH PIPELINE + GRIDSEARCH")
    print("="*70)
    
    results = {}
    
    # Baseline
    print("\n🔬 Testing: Baseline")
    res_baseline = test_with_pipeline(baseline_features, labels, "Baseline")
    results['Baseline'] = res_baseline
    baseline_auc = res_baseline['auc']
    print(f"  AUC: {res_baseline['auc']:.4f}")
    
    # H1
    print("\n🔬 Testing: H1_Spectrum")
    res_h1 = test_with_pipeline(h1_features, labels, "H1_Spectrum")
    results['H1_Spectrum'] = res_h1
    delta_h1 = res_h1['auc'] - baseline_auc
    symbol = "✅" if delta_h1 >= 0 else "❌"
    print(f"  AUC: {res_h1['auc']:.4f} ({symbol} {delta_h1:+.4f} vs baseline)")
    
    # H2
    print("\n🔬 Testing: H2_ScaleLaw")
    res_h2 = test_with_pipeline(h2_features, labels, "H2_ScaleLaw")
    results['H2_ScaleLaw'] = res_h2
    delta_h2 = res_h2['auc'] - baseline_auc
    symbol = "✅" if delta_h2 >= 0 else "❌"
    print(f"  AUC: {res_h2['auc']:.4f} ({symbol} {delta_h2:+.4f} vs baseline)")
    
    # Combinations
    print("\n" + "="*70)
    print("🧪 KOMBINACJE")
    print("="*70)
    
    # Baseline + H1
    combined_h1 = np.concatenate([baseline_features, h1_features], axis=1)
    res_bh1 = test_with_pipeline(combined_h1, labels, "Baseline+H1")
    delta = res_bh1['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"\n  Baseline + H1: AUC = {res_bh1['auc']:.4f} ({symbol} {delta:+.4f})")
    
    # Baseline + H2
    combined_h2 = np.concatenate([baseline_features, h2_features], axis=1)
    res_bh2 = test_with_pipeline(combined_h2, labels, "Baseline+H2")
    delta = res_bh2['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Baseline + H2: AUC = {res_bh2['auc']:.4f} ({symbol} {delta:+.4f})")
    
    # Baseline + H1 + H2
    combined_all = np.concatenate([baseline_features, h1_features, h2_features], axis=1)
    res_all = test_with_pipeline(combined_all, labels, "Baseline+H1+H2")
    delta = res_all['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Baseline + H1 + H2: AUC = {res_all['auc']:.4f} ({symbol} {delta:+.4f})")
    
    # H1 + H2 only
    combined_h1h2 = np.concatenate([h1_features, h2_features], axis=1)
    res_h1h2 = test_with_pipeline(combined_h1h2, labels, "H1+H2")
    print(f"  H1 + H2 tylko: AUC = {res_h1h2['auc']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Konfiguracja':<25} {'AUC':<8} {'vs Baseline':<12} {'Wymiary':<10}")
    print("-"*70)
    
    all_results = [
        res_baseline, res_h1, res_h2, 
        res_bh1, res_bh2, res_all, res_h1h2
    ]
    
    for res in all_results:
        delta = res['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "⚠️" if delta > -0.02 else "❌"
        print(f"{res['name']:<25} {res['auc']:.4f}  {symbol} {delta:+.4f}      {res['shape'][1]}D")
    
    # Best
    best = max(all_results, key=lambda x: x['auc'])
    print(f"\n🏆 NAJLEPSZY: {best['name']} (AUC={best['auc']:.4f})")
    
    # Save
    import json
    summary = {
        'baseline_auc': baseline_auc,
        'results': [{
            'name': r['name'],
            'auc': r['auc'],
            'acc': r['acc'],
            'delta': r['auc'] - baseline_auc,
            'dim': r['shape'][1]
        } for r in all_results]
    }
    
    with open(OUTPUT_DIR / "h1_h2_fixed_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_DIR}/")
    
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
