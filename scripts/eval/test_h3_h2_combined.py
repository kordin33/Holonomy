"""
test_h3_h2_combined.py - Test łączący najlepsze: H3 V2 + H2 Scale Law

H3 V2 solo: AUC 0.835 ✅
H2 solo: AUC 0.804 ✅

Czy razem będą jeszcze lepsze?
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
from deepfake_guard.features.h3_normalized_dispersion_v2 import H3_NormalizedDispersionV2
from deepfake_guard.features.holonomy_h1_h2_fixed import H2_AreaScaleLaw_Fixed
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 200  # nieco więcej dla stabilności
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
    print("🔬 TEST: H3 V2 + H2 Scale Law (NAJLEPSZE HIPOTEZY)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data(SAMPLE_SIZE)
    
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extractors
    h3_v2 = H3_NormalizedDispersionV2()
    h2 = H2_AreaScaleLaw_Fixed()
    
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
    
    # H3 V2
    print("\n📊 Extracting: H3 V2")
    h3_features = []
    for img in tqdm(images, desc="H3 V2"):
        try:
            h3_features.append(h3_v2.extract_features(encoder, img))
        except Exception as e:
            h3_features.append(np.zeros(9, dtype=np.float32))
    h3_features = np.array(h3_features, dtype=np.float32)
    print(f"   Shape: {h3_features.shape}")
    
    # H2
    print("\n📊 Extracting: H2 Scale Law")
    h2_features = []
    for img in tqdm(images, desc="H2"):
        try:
            h2_features.append(h2.extract_features(encoder, img))
        except Exception as e:
            h2_features.append(np.zeros(5, dtype=np.float32))
    h2_features = np.array(h2_features, dtype=np.float32)
    print(f"   Shape: {h2_features.shape}")
    
    # Test
    print("\n" + "="*70)
    print("TESTING WITH PIPELINE + GRIDSEARCH")
    print("="*70)
    
    results = []
    
    # Baseline
    res_baseline = test_with_pipeline(baseline_features, labels, "Baseline")
    baseline_auc = res_baseline['auc']
    print(f"\n  Baseline: AUC = {baseline_auc:.4f} ({baseline_features.shape[1]}D)")
    results.append(res_baseline)
    
    # H3 V2 solo
    res_h3 = test_with_pipeline(h3_features, labels, "H3_V2")
    delta = res_h3['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  H3 V2 solo: AUC = {res_h3['auc']:.4f} ({symbol} {delta:+.4f}) [{h3_features.shape[1]}D]")
    results.append(res_h3)
    
    # H2 solo
    res_h2 = test_with_pipeline(h2_features, labels, "H2")
    delta = res_h2['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  H2 solo: AUC = {res_h2['auc']:.4f} ({symbol} {delta:+.4f}) [{h2_features.shape[1]}D]")
    results.append(res_h2)
    
    # Combinations
    print("\n🧪 KOMBINACJE:")
    
    # H3 + H2
    h3_h2 = np.concatenate([h3_features, h2_features], axis=1)
    res_h3h2 = test_with_pipeline(h3_h2, labels, "H3+H2")
    delta = res_h3h2['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  H3 V2 + H2: AUC = {res_h3h2['auc']:.4f} ({symbol} {delta:+.4f}) [{h3_h2.shape[1]}D]")
    results.append(res_h3h2)
    
    # Baseline + H3
    base_h3 = np.concatenate([baseline_features, h3_features], axis=1)
    res_bh3 = test_with_pipeline(base_h3, labels, "Baseline+H3")
    delta = res_bh3['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Baseline + H3 V2: AUC = {res_bh3['auc']:.4f} ({symbol} {delta:+.4f}) [{base_h3.shape[1]}D]")
    results.append(res_bh3)
    
    # Baseline + H2
    base_h2 = np.concatenate([baseline_features, h2_features], axis=1)
    res_bh2 = test_with_pipeline(base_h2, labels, "Baseline+H2")
    delta = res_bh2['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Baseline + H2: AUC = {res_bh2['auc']:.4f} ({symbol} {delta:+.4f}) [{base_h2.shape[1]}D]")
    results.append(res_bh2)
    
    # Baseline + H3 + H2 (ALL)
    all_combined = np.concatenate([baseline_features, h3_features, h2_features], axis=1)
    res_all = test_with_pipeline(all_combined, labels, "Baseline+H3+H2")
    delta = res_all['auc'] - baseline_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  Baseline + H3 + H2: AUC = {res_all['auc']:.4f} ({symbol} {delta:+.4f}) [{all_combined.shape[1]}D]")
    results.append(res_all)
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Konfiguracja':<25} {'AUC':<8} {'vs Baseline':<12} {'Wymiary'}")
    print("-"*60)
    
    for res in sorted(results, key=lambda x: x['auc'], reverse=True):
        delta = res['auc'] - baseline_auc
        symbol = "🏆" if res['auc'] == max(r['auc'] for r in results) else ("✅" if delta >= 0 else "❌")
        print(f"{res['name']:<25} {res['auc']:.4f}  {symbol} {delta:+.4f}      {res['shape'][1]}D")
    
    best = max(results, key=lambda x: x['auc'])
    print(f"\n🏆 NAJLEPSZY: {best['name']} (AUC={best['auc']:.4f})")
    
    # Save
    import json
    summary = {
        'baseline_auc': baseline_auc,
        'results': [{
            'name': r['name'],
            'auc': r['auc'],
            'delta': r['auc'] - baseline_auc,
            'dim': r['shape'][1]
        } for r in results]
    }
    
    Path("./results/h3_h2_combined").mkdir(parents=True, exist_ok=True)
    with open("./results/h3_h2_combined/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
