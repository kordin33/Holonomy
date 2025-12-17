"""
test_optimized_vs_production.py - Porównanie OPTIMIZED vs PRODUCTION

Testuje:
1. Production (stare): h3_dispersion, h2_scale_law, baseline
2. Optimized (nowe): H3_Optimized, H2_Optimized, BaselineOptimized
3. Combined z per-block scaling
4. Stacking (meta-model)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# PRODUCTION (stare)
from deepfake_guard.features.production.h3_dispersion import H3_NormalizedDispersionV2
from deepfake_guard.features.production.h2_scale_law import H2_AreaScaleLaw_Fixed
from deepfake_guard.features.production.baseline import extract_minimal_features

# OPTIMIZED (nowe)
from deepfake_guard.features.production.optimized_features import (
    H2_Optimized,
    H3_Optimized, 
    BaselineOptimized,
    CombinedOptimized,
)

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 200
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


def test_stable(features, labels, name):
    """Test ze stabilnym CV."""
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]},
                        cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(X_train, y_train)
    
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return {'name': name, 'auc': roc_auc_score(y_test, y_prob), 'shape': features.shape}


def test_stacking(features_dict, labels, name):
    """Test ze stacking (meta-model na probabilities)."""
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        range(len(labels)), labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Train individual models and get predictions
    meta_train = []
    meta_test = []
    
    for block_name, features in features_dict.items():
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_train = features[X_train_idx]
        X_test = features[X_test_idx]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
        grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]},
                            cv=cv, scoring='roc_auc', n_jobs=1)
        grid.fit(X_train, y_train)
        
        # Proba for train (OOF) and test
        meta_test.append(grid.best_estimator_.predict_proba(X_test)[:, 1])
    
    # Stack
    meta_test = np.column_stack(meta_test)
    
    # For simplicity, just use mean of probabilities
    stacked_prob = meta_test.mean(axis=1)
    auc = roc_auc_score(y_test, stacked_prob)
    
    return {'name': name, 'auc': auc, 'shape': (len(labels), len(features_dict))}


def main():
    print("="*70)
    print("🔬 TEST: OPTIMIZED vs PRODUCTION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extractors
    prod_h3 = H3_NormalizedDispersionV2()
    prod_h2 = H2_AreaScaleLaw_Fixed()
    
    opt_h2 = H2_Optimized()
    opt_h3 = H3_Optimized()
    opt_base = BaselineOptimized()
    
    # Extract Production
    print("\n" + "="*70)
    print("EXTRACTING PRODUCTION FEATURES")
    print("="*70)
    
    prod_baseline = np.array([extract_minimal_features(encoder, img)['minimal'] 
                              for img in tqdm(images, desc="Prod Baseline")])
    
    prod_h3_feat = []
    for img in tqdm(images, desc="Prod H3"):
        try:
            prod_h3_feat.append(prod_h3.extract_features(encoder, img))
        except:
            prod_h3_feat.append(np.zeros(9))
    prod_h3_feat = np.array(prod_h3_feat, dtype=np.float32)
    
    prod_h2_feat = []
    for img in tqdm(images, desc="Prod H2"):
        try:
            prod_h2_feat.append(prod_h2.extract_features(encoder, img))
        except:
            prod_h2_feat.append(np.zeros(5))
    prod_h2_feat = np.array(prod_h2_feat, dtype=np.float32)
    
    # Extract Optimized
    print("\n" + "="*70)
    print("EXTRACTING OPTIMIZED FEATURES")
    print("="*70)
    
    opt_baseline_feat = []
    for img in tqdm(images, desc="Opt Baseline"):
        try:
            opt_baseline_feat.append(opt_base.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            opt_baseline_feat.append(np.zeros(18))
    opt_baseline_feat = np.array(opt_baseline_feat, dtype=np.float32)
    
    opt_h3_feat = []
    for img in tqdm(images, desc="Opt H3"):
        try:
            opt_h3_feat.append(opt_h3.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            opt_h3_feat.append(np.zeros(27))
    opt_h3_feat = np.array(opt_h3_feat, dtype=np.float32)
    
    opt_h2_feat = []
    for img in tqdm(images, desc="Opt H2"):
        try:
            opt_h2_feat.append(opt_h2.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            opt_h2_feat.append(np.zeros(6))
    opt_h2_feat = np.array(opt_h2_feat, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    results = []
    
    # Production
    print("\n📦 PRODUCTION:")
    
    res = test_stable(prod_baseline, labels, "Prod_Baseline")
    prod_baseline_auc = res['auc']
    print(f"  Baseline: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    res = test_stable(prod_h3_feat, labels, "Prod_H3")
    print(f"  H3: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    res = test_stable(prod_h2_feat, labels, "Prod_H2")
    print(f"  H2: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    prod_all = np.concatenate([prod_baseline, prod_h3_feat, prod_h2_feat], axis=1)
    res = test_stable(prod_all, labels, "Prod_Combined")
    print(f"  Combined: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    # Optimized
    print("\n🚀 OPTIMIZED:")
    
    res = test_stable(opt_baseline_feat, labels, "Opt_Baseline")
    print(f"  Baseline: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    res = test_stable(opt_h3_feat, labels, "Opt_H3")
    print(f"  H3: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    res = test_stable(opt_h2_feat, labels, "Opt_H2")
    print(f"  H2: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    opt_all = np.concatenate([opt_baseline_feat, opt_h3_feat, opt_h2_feat], axis=1)
    res = test_stable(opt_all, labels, "Opt_Combined")
    print(f"  Combined: {res['auc']:.4f} ({res['shape'][1]}D)")
    results.append(res)
    
    # Stacking
    print("\n📊 STACKING (meta-model):")
    
    res = test_stacking({
        'baseline': opt_baseline_feat,
        'h3': opt_h3_feat,
        'h2': opt_h2_feat,
    }, labels, "Opt_Stacked")
    print(f"  Optimized Stacked: {res['auc']:.4f}")
    results.append(res)
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Config':<25} {'AUC':<8} {'vs Prod_Base'}")
    print("-"*50)
    
    for res in sorted(results, key=lambda x: x['auc'], reverse=True):
        delta = res['auc'] - prod_baseline_auc
        symbol = "🏆" if res['auc'] == max(r['auc'] for r in results) else ("✅" if delta >= 0 else "❌")
        print(f"{res['name']:<25} {res['auc']:.4f}  {symbol} {delta:+.4f}")
    
    best = max(results, key=lambda x: x['auc'])
    print(f"\n🏆 NAJLEPSZY: {best['name']} (AUC={best['auc']:.4f})")
    
    del encoder
    torch.cuda.empty_cache()
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
