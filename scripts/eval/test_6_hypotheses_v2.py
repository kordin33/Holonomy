"""
test_6_hypotheses_v2.py - Test POPRAWIONYCH hipotez na CIFAKE

NAPRAWY:
1. H4: Prawdziwy coupling wektorowy
2. H1/H5/H6: Właściwy fit PCA z train
3. H2: Stała struktura pętli
4. H3: Seed per-image
5. H6: u = (z0 - μ_train)
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
from deepfake_guard.features.holonomy_hypotheses_v2 import (
    H1_HolonomySpectrum,
    H2_AreaScaleLaw,
    H3_ConsistencyIndex,
    H4_PatchCoupled,
    H5_HolonomyTensor,
    H6_Orthogonality,
)
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

# CIFAKE - szybki test
DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/6_hypotheses_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 150
RANDOM_STATE = 42
BASELINE_AUC = 0.861


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
    """Test z Pipeline + GridSearch."""
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])
    
    param_grid = {
        'svm__C': [0.3, 1, 3, 10],
        'svm__gamma': ['scale', 0.01, 0.1]
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
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
    print("🔬 TEST POPRAWIONYCH HIPOTEZ (V2)")
    print("   Dataset: CIFAKE (szybki test)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data_cifake(SAMPLE_SIZE)
    
    # Encoder
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Collect embeddings for PCA (tylko train!)
    print("\nCollecting embeddings for PCA (train only)...")
    train_idx, _ = train_test_split(
        range(len(labels)), test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    train_images = [images[i] for i in train_idx]
    train_embeddings = encoder.encode_batch(train_images, batch_size=32, show_progress=True)
    
    # Initialize extractors
    h1 = H1_HolonomySpectrum()
    h2 = H2_AreaScaleLaw()
    h3 = H3_ConsistencyIndex(n_samples=15)
    h4 = H4_PatchCoupled(grid=3)
    h5 = H5_HolonomyTensor()
    h6 = H6_Orthogonality()
    
    # Fit PCA na TRAIN
    print("\nFitting PCA on TRAIN only...")
    h1.fit_pca(train_embeddings)
    h4.fit_pca(train_embeddings)
    h5.fit_pca(train_embeddings)
    h6.fit_pca(train_embeddings)
    
    extractors = {
        'H1_Spectrum': h1,
        'H2_ScaleLaw': h2,
        'H3_Consistency': h3,
        'H4_PatchCoupled': h4,
        'H5_Tensor': h5,
        'H6_Orthogonality': h6,
    }
    
    # Extract features
    print("\n" + "="*70)
    print("EXTRACTING FEATURES (V2 - FIXED)")
    print("="*70)
    
    all_features = {}
    
    for name, extractor in extractors.items():
        print(f"\n📊 Extracting: {name}")
        features_list = []
        
        for img in tqdm(images, desc=name):
            try:
                feat = extractor.extract_features(encoder, img)
                features_list.append(feat)
            except Exception as e:
                print(f"Error: {e}")
                features_list.append(np.zeros(10, dtype=np.float32))
        
        all_features[name] = np.array(features_list, dtype=np.float32)
        print(f"   Shape: {all_features[name].shape}")
    
    # Baseline
    print("\n📊 Extracting: Baseline (Minimal)")
    baseline_features = []
    for img in tqdm(images, desc="Baseline"):
        feat = extract_minimal_features(encoder, img)
        baseline_features.append(feat['minimal'])
    
    all_features['Baseline'] = np.array(baseline_features, dtype=np.float32)
    print(f"   Shape: {all_features['Baseline'].shape}")
    
    # Test each
    print("\n" + "="*70)
    print("TESTING WITH PIPELINE + GRIDSEARCH")
    print("="*70)
    
    results = []
    
    for name in ['Baseline', 'H1_Spectrum', 'H2_ScaleLaw', 'H3_Consistency', 
                 'H4_PatchCoupled', 'H5_Tensor', 'H6_Orthogonality']:
        print(f"\n🔬 Testing: {name}")
        res = test_with_pipeline(all_features[name], labels, name)
        results.append(res)
        
        delta = res['auc'] - BASELINE_AUC
        symbol = "✅" if delta >= 0 else "❌"
        print(f"  AUC: {res['auc']:.4f} ({symbol} {delta:+.4f} vs baseline)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE V2 (POPRAWIONE)")
    print("="*70)
    
    print(f"\n{'Hipoteza':<20} {'AUC':<8} {'vs Baseline':<12} {'Wymiary':<10}")
    print("-"*70)
    
    for res in results:
        delta = res['auc'] - BASELINE_AUC
        symbol = "✅" if delta >= 0 else "⚠️" if delta > -0.05 else "❌"
        print(f"{res['name']:<20} {res['auc']:.4f}  {symbol} {delta:+.4f}      {res['shape'][1]}D")
    
    # Best
    new_results = [r for r in results if r['name'] != 'Baseline']
    best_new = max(new_results, key=lambda x: x['auc'])
    print(f"\n🏆 NAJLEPSZA NOWA: {best_new['name']} (AUC={best_new['auc']:.4f})")
    
    # Combined test
    print("\n" + "="*70)
    print("🧪 KOMBINACJE")
    print("="*70)
    
    combined = np.concatenate([
        all_features['Baseline'],
        all_features[best_new['name']]
    ], axis=1)
    
    res_combined = test_with_pipeline(combined, labels, f"Baseline + {best_new['name']}")
    print(f"  Baseline + {best_new['name']}: AUC = {res_combined['auc']:.4f}")
    
    # All combined
    all_combined = np.concatenate([all_features[name] for name in all_features.keys()], axis=1)
    res_all = test_with_pipeline(all_combined, labels, "ALL Combined")
    print(f"  ALL Combined ({all_combined.shape[1]}D): AUC = {res_all['auc']:.4f}")
    
    # Save
    import json
    summary = {
        'results': [{
            'name': r['name'],
            'auc': r['auc'],
            'acc': r['acc'],
            'shape': list(r['shape'])
        } for r in results + [res_combined, res_all]]
    }
    
    with open(OUTPUT_DIR / "v2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_DIR}/")
    
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
