"""
test_holonomy_decomposition.py - Test Holonomy Decomposition

HIPOTEZA:
- Fake i real różnią się w rotacyjnej holonomii
- Artefakty AI są "orientacyjne" (rotacja cech), nie tylko translacyjne
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
from deepfake_guard.features.holonomy_decomposition import (
    HolonomyDecomposition,
    HolonomyDecompositionFast,
)
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/holonomy_decomposition")
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
    print("🔬 TEST: HOLONOMY DECOMPOSITION")
    print("   Translacyjna vs Rotacyjna holonomia")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data_cifake(SAMPLE_SIZE)
    
    # Encoder
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extractors
    decomp_full = HolonomyDecomposition(frame_dim=8)
    decomp_fast = HolonomyDecompositionFast(frame_dim=6)
    
    # Extract features
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    # Baseline
    print("\n📊 Extracting: Baseline (Minimal)")
    baseline_features = []
    for img in tqdm(images, desc="Baseline"):
        feat = extract_minimal_features(encoder, img)
        baseline_features.append(feat['minimal'])
    baseline_features = np.array(baseline_features, dtype=np.float32)
    print(f"   Shape: {baseline_features.shape}")
    
    # Fast version (dla szybkości)
    print("\n📊 Extracting: Holonomy Decomposition (Fast)")
    decomp_features = []
    for img in tqdm(images, desc="Decomposition"):
        try:
            feat = decomp_fast.extract_features(encoder, img)
            decomp_features.append(feat)
        except Exception as e:
            print(f"Error: {e}")
            decomp_features.append(np.zeros(12, dtype=np.float32))
    decomp_features = np.array(decomp_features, dtype=np.float32)
    print(f"   Shape: {decomp_features.shape}")
    
    # Test
    print("\n" + "="*70)
    print("TESTING WITH PIPELINE + GRIDSEARCH")
    print("="*70)
    
    # Baseline
    print("\n🔬 Testing: Baseline")
    res_baseline = test_with_pipeline(baseline_features, labels, "Baseline")
    baseline_auc = res_baseline['auc']
    print(f"  AUC: {res_baseline['auc']:.4f}")
    
    # Decomposition
    print("\n🔬 Testing: Holonomy Decomposition")
    res_decomp = test_with_pipeline(decomp_features, labels, "Decomposition")
    delta = res_decomp['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  AUC: {res_decomp['auc']:.4f} ({symbol} {delta:+.4f} vs baseline)")
    
    # Combinations
    print("\n" + "="*70)
    print("🧪 KOMBINACJE")
    print("="*70)
    
    # Baseline + Decomposition
    combined = np.concatenate([baseline_features, decomp_features], axis=1)
    res_combined = test_with_pipeline(combined, labels, "Baseline+Decomp")
    delta = res_combined['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"\n  Baseline + Decomposition: AUC = {res_combined['auc']:.4f} ({symbol} {delta:+.4f})")
    
    # Analyze feature importance (correlacja z labels)
    print("\n" + "="*70)
    print("📊 ANALIZA CECH DECOMPOSITION")
    print("="*70)
    
    feature_names = ['H_trans', 'frame_dev', 'trace', 'total_rot'] * 3
    feature_names = [f"loop{i//4}_{n}" for i, n in enumerate(feature_names)]
    
    print(f"\n{'Cecha':<25} {'Corr z label':<12} {'Mean Real':<12} {'Mean Fake':<12}")
    print("-"*60)
    
    for i, name in enumerate(feature_names[:12]):
        col = decomp_features[:, i]
        corr = np.corrcoef(col, labels)[0, 1]
        mean_real = col[labels == 1].mean()
        mean_fake = col[labels == 0].mean()
        print(f"{name:<25} {corr:>+.4f}       {mean_real:>8.4f}     {mean_fake:>8.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Konfiguracja':<25} {'AUC':<8} {'vs Baseline':<12} {'Wymiary':<10}")
    print("-"*60)
    
    for res in [res_baseline, res_decomp, res_combined]:
        delta = res['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "❌"
        print(f"{res['name']:<25} {res['auc']:.4f}  {symbol} {delta:+.4f}      {res['shape'][1]}D")
    
    # Save
    import json
    summary = {
        'baseline_auc': baseline_auc,
        'decomposition_auc': res_decomp['auc'],
        'combined_auc': res_combined['auc'],
        'delta': res_decomp['auc'] - baseline_auc,
    }
    
    with open(OUTPUT_DIR / "decomposition_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_DIR}/")
    
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
