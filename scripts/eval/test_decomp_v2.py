"""
test_decomp_v2.py - Test NAPRAWIONEJ Holonomy Decomposition V2
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.holonomy_decomposition_v2 import (
    HolonomyDecompositionV2,
    HolonomyDecompositionV2Fast,
)
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 150
RANDOM_STATE = 42


def load_data(n_per_class):
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
    
    return {'name': name, 'auc': auc, 'shape': features.shape}


def main():
    print("="*70)
    print("🔬 TEST: NAPRAWIONA Holonomy Decomposition V2")
    print("   Principal angles + Subspace distance")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data(SAMPLE_SIZE)
    
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    decomp = HolonomyDecompositionV2Fast(frame_dim=6)
    
    # Extract
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    # Baseline
    print("\n📊 Extracting: Baseline")
    baseline_features = []
    for img in tqdm(images, desc="Baseline"):
        feat = extract_minimal_features(encoder, img)
        baseline_features.append(feat['minimal'])
    baseline_features = np.array(baseline_features, dtype=np.float32)
    print(f"   Shape: {baseline_features.shape}")
    
    # Decomp V2
    print("\n📊 Extracting: Decomposition V2 (FIXED)")
    decomp_features = []
    for img in tqdm(images, desc="Decomp V2"):
        try:
            feat = decomp.extract_features(encoder, img)
            decomp_features.append(feat)
        except Exception as e:
            print(f"Error: {e}")
            decomp_features.append(np.zeros(15, dtype=np.float32))
    decomp_features = np.array(decomp_features, dtype=np.float32)
    print(f"   Shape: {decomp_features.shape}")
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    res_baseline = test_with_pipeline(baseline_features, labels, "Baseline")
    baseline_auc = res_baseline['auc']
    print(f"\n  Baseline: AUC = {baseline_auc:.4f}")
    
    res_decomp = test_with_pipeline(decomp_features, labels, "Decomp_V2")
    delta = res_decomp['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Decomp V2: AUC = {res_decomp['auc']:.4f} ({symbol} {delta:+.4f})")
    
    # Combined
    combined = np.concatenate([baseline_features, decomp_features], axis=1)
    res_combined = test_with_pipeline(combined, labels, "Baseline+Decomp_V2")
    delta = res_combined['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Baseline + Decomp V2: AUC = {res_combined['auc']:.4f} ({symbol} {delta:+.4f})")
    
    # Feature analysis
    print("\n📊 Analiza cech Decomp V2:")
    names = ['H_trans', 'subspace_dist', 'theta_sum', 'theta_mean', 'cos_sum'] * 3
    names = [f"loop{i//5}_{n}" for i, n in enumerate(names)]
    
    print(f"\n{'Cecha':<25} {'Corr':<8} {'Mean Real':<12} {'Mean Fake':<12}")
    print("-"*60)
    for i, name in enumerate(names):
        col = decomp_features[:, i]
        corr = np.corrcoef(col, labels)[0, 1]
        mr = col[labels == 1].mean()
        mf = col[labels == 0].mean()
        print(f"{name:<25} {corr:>+.4f}   {mr:>8.4f}     {mf:>8.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Config':<25} {'AUC':<8} {'vs Baseline':<12}")
    print("-"*50)
    for r in [res_baseline, res_decomp, res_combined]:
        delta = r['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "❌"
        print(f"{r['name']:<25} {r['auc']:.4f}  {symbol} {delta:+.4f}")
    
    del encoder
    torch.cuda.empty_cache()
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
