"""
test_h2h3_v5.py - Test H2_V5 (Curvature Density) + H3_V5 (Coherence + Roughness)

Cel: Sprawdzić czy nowe "Frontier" H2/H3 dają:
1. Lepszy standalone niż poprzednie wersje.
2. Pozytywne ablacje z V12/V13 Baseline.
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
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.h2_v5 import H2_V5
from deepfake_guard.features.h3_v5 import H3_V5
from deepfake_guard.features.optimized_v3 import BaselineV3
from deepfake_guard.features.production.baseline import extract_minimal_features

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

def main():
    print("="*70)
    print("🔬 TEST: H2_V5 & H3_V5 FRONTIER")
    print("   H2: Curvature-Density (Commutator Surface)")
    print("   H3: Patch-Coherence + Roughness")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    h2 = H2_V5()
    h3 = H3_V5()
    baseline = BaselineV3()
    
    # Extract
    print("\n📊 Extracting features...")
    feats_base, feats_h2, feats_h3 = [], [], []
    
    for img in tqdm(images, desc="Extracting"):
        try:
            feats_base.append(baseline.extract_features(encoder, img))
        except:
            feats_base.append(np.zeros(63))
            
        try:
            feats_h2.append(h2.extract_features(encoder, img))
        except Exception as e:
            print(f"H2 Error: {e}")
            feats_h2.append(np.zeros(16))
            
        try:
            feats_h3.append(h3.extract_features(encoder, img))
        except Exception as e:
            print(f"H3 Error: {e}")
            feats_h3.append(np.zeros(18))
    
    feats_base = np.array(feats_base, dtype=np.float32)
    feats_h2 = np.array(feats_h2, dtype=np.float32)
    feats_h3 = np.array(feats_h3, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Ref
    res = test_stable(feats_base, labels, "Baseline_V3")
    base_auc = res['auc']
    print(f"\n  Baseline_V3: {base_auc:.4f} [{res['shape'][1]}D]")
    
    # Standalone H2/H3
    print("\n📊 STANDALONE:")
    res = test_stable(feats_h2, labels, "H2_V5")
    print(f"  H2_V5 (Curvature Density): {res['auc']:.4f} [{res['shape'][1]}D]")
    
    res = test_stable(feats_h3, labels, "H3_V5")
    print(f"  H3_V5 (Coherence + Roughness): {res['auc']:.4f} [{res['shape'][1]}D]")
    
    # Ablations
    print("\n📊 ABLACJE (Baseline + H2/H3):")
    
    combined_h2 = np.concatenate([feats_base, feats_h2], axis=1)
    res = test_stable(combined_h2, labels, "Base+H2")
    delta = res['auc'] - base_auc
    print(f"  Base + H2_V5: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    combined_h3 = np.concatenate([feats_base, feats_h3], axis=1)
    res = test_stable(combined_h3, labels, "Base+H3")
    delta = res['auc'] - base_auc
    print(f"  Base + H3_V5: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    combined_all = np.concatenate([feats_base, feats_h2, feats_h3], axis=1)
    res = test_stable(combined_all, labels, "Base+H2+H3")
    delta = res['auc'] - base_auc
    print(f"  Base + H2 + H3: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    # Correlations
    print("\n📊 H2_V5 CORRELATIONS:")
    h2_names = ['k_mean', 'k_std', 'k_p90', 'k_max', 'k_aniso', 'k_rank_eff', 'k_patch_range', 'k_patch_sync']
    for i, name in enumerate(h2_names[:feats_h2.shape[1]]):
        corr = np.corrcoef(feats_h2[:, i], labels)[0, 1]
        if abs(corr) > 0.1:
            print(f"  {name:<15}: {corr:+.4f}")
    
    print("\n📊 H3_V5 CORRELATIONS:")
    h3_names = ['rough_R', 'allan', 'jitter_std', 'sync_mean', 'sync_min', 'sync_spread', 'worst_patch', 'range_step', 'range_jitter', 'mad_step']
    for i, name in enumerate(h3_names[:feats_h3.shape[1]]):
        corr = np.corrcoef(feats_h3[:, i], labels)[0, 1]
        if abs(corr) > 0.1:
            print(f"  {name:<15}: {corr:+.4f}")

    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
