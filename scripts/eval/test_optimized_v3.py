"""
test_optimized_v3.py - Test Optimized V3 (pełna implementacja)
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
from deepfake_guard.features.optimized_v3 import BaselineV3, H2_V3, H3_V3, CombinedV3
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
    print("🔬 TEST: OPTIMIZED V3 (PEŁNA IMPLEMENTACJA)")
    print("   Baseline 63D + H2 15D + H3 20D = 98D")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    combined_v3 = CombinedV3()
    
    # Extract Prod Baseline (Ref)
    print("\n📊 Extracting Production Baseline (Ref)...")
    prod_base = np.array([extract_minimal_features(encoder, img)['minimal'] 
                          for img in tqdm(images, desc="Prod Base")])
    
    # Extract V3
    print("\n📊 Extracting Optimized V3...")
    v3_base, v3_h2, v3_h3 = [], [], []
    
    for img in tqdm(images, desc="V3"):
        try:
            feats = combined_v3.extract_features(encoder, img)
            v3_base.append(feats['baseline'])
            v3_h2.append(feats['h2'])
            v3_h3.append(feats['h3'])
        except Exception as e:
            print(f"Error: {e}")
            v3_base.append(np.zeros(63))
            v3_h2.append(np.zeros(15))
            v3_h3.append(np.zeros(20))
    
    v3_base = np.array(v3_base, dtype=np.float32)
    v3_h2 = np.array(v3_h2, dtype=np.float32)
    v3_h3 = np.array(v3_h3, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    # Reference
    res_prod = test_stable(prod_base, labels, "Prod_Base")
    base_auc = res_prod['auc']
    print(f"\n  Reference (Prod_Base): {base_auc:.4f} ({prod_base.shape[1]}D)")
    
    # V3 Components
    res = test_stable(v3_base, labels, "V3_Base")
    delta = res['auc'] - base_auc
    print(f"  V3_Baseline: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    res = test_stable(v3_h2, labels, "V3_H2")
    delta = res['auc'] - base_auc
    print(f"  V3_H2: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    res = test_stable(v3_h3, labels, "V3_H3")
    delta = res['auc'] - base_auc
    print(f"  V3_H3: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    # Ablacje
    print("\n📊 ABLACJE:")
    
    combined_bh2 = np.concatenate([v3_base, v3_h2], axis=1)
    res = test_stable(combined_bh2, labels, "V3_Base+H2")
    delta = res['auc'] - base_auc
    print(f"  Base+H2: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    combined_bh3 = np.concatenate([v3_base, v3_h3], axis=1)
    res = test_stable(combined_bh3, labels, "V3_Base+H3")
    delta = res['auc'] - base_auc
    print(f"  Base+H3: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    combined_all = np.concatenate([v3_base, v3_h2, v3_h3], axis=1)
    res = test_stable(combined_all, labels, "V3_All")
    delta = res['auc'] - base_auc
    print(f"  Base+H2+H3: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    # Top correlations
    print("\n📊 TOP CORRELATIONS (V3_H2):")
    h2_names = ['alpha_all', 'r2_all', 'residual_std', 'alpha_low', 'alpha_high', 'delta_alpha',
                'rho_low', 'rho_high', 'delta_rho', 'slack_low', 'slack_high', 'delta_slack',
                'spearman_H', 'spearman_rho', 'commutator']
    for i, name in enumerate(h2_names):
        if i < v3_h2.shape[1]:
            corr = np.corrcoef(v3_h2[:, i], labels)[0, 1]
            if abs(corr) > 0.1:
                print(f"  {name:<15}: {corr:+.4f}")
    
    print("\n📊 TOP CORRELATIONS (V3_H3):")
    for i in range(min(10, v3_h3.shape[1])):
        corr = np.corrcoef(v3_h3[:, i], labels)[0, 1]
        if abs(corr) > 0.15:
            print(f"  Feature {i}: {corr:+.4f}")
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
