"""
test_optimized_v4.py - Test H1/H2/H3 V4 (Full Optimization)
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
from deepfake_guard.features.optimized_v4 import H1_V4, H2_V4, H3_V4, CombinedV4
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
    print("🔬 TEST: OPTIMIZED V4 (H1/H2/H3 Full Optimization)")
    print("   Log-Map + Patch-H1 + 2D Fit + Commutator Curve")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    h1 = H1_V4()
    h2 = H2_V4()
    h3 = H3_V4()
    baseline = BaselineV3()
    
    # Extract
    print("\n📊 Extracting Production Baseline (Ref)...")
    prod_base = np.array([extract_minimal_features(encoder, img)['minimal'] 
                          for img in tqdm(images, desc="Prod Base")])
    
    print("\n📊 Extracting V4...")
    v4_h1, v4_h2, v4_h3, v4_base = [], [], [], []
    
    for img in tqdm(images, desc="V4"):
        try:
            v4_h1.append(h1.extract_features(encoder, img))
            v4_h2.append(h2.extract_features(encoder, img))
            v4_h3.append(h3.extract_features(encoder, img))
            v4_base.append(baseline.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            v4_h1.append(np.zeros(42))
            v4_h2.append(np.zeros(25))
            v4_h3.append(np.zeros(27))
            v4_base.append(np.zeros(63))
    
    v4_h1 = np.array(v4_h1, dtype=np.float32)
    v4_h2 = np.array(v4_h2, dtype=np.float32)
    v4_h3 = np.array(v4_h3, dtype=np.float32)
    v4_base = np.array(v4_base, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    # Reference
    res_prod = test_stable(prod_base, labels, "Prod_Base")
    base_auc = res_prod['auc']
    print(f"\n  Reference (Prod_Base): {base_auc:.4f}")
    
    # V4 Components
    res = test_stable(v4_base, labels, "V4_Baseline")
    print(f"  V4_Baseline: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    res = test_stable(v4_h1, labels, "V4_H1")
    print(f"  V4_H1: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    res = test_stable(v4_h2, labels, "V4_H2")
    print(f"  V4_H2: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    res = test_stable(v4_h3, labels, "V4_H3")
    print(f"  V4_H3: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    # Ablacje
    print("\n📊 ABLACJE:")
    
    combined_bh1 = np.concatenate([v4_base, v4_h1], axis=1)
    res = test_stable(combined_bh1, labels, "Base+H1")
    print(f"  Base+H1: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    combined_bh2 = np.concatenate([v4_base, v4_h2], axis=1)
    res = test_stable(combined_bh2, labels, "Base+H2")
    print(f"  Base+H2: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    combined_bh3 = np.concatenate([v4_base, v4_h3], axis=1)
    res = test_stable(combined_bh3, labels, "Base+H3")
    print(f"  Base+H3: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    combined_all = np.concatenate([v4_base, v4_h1, v4_h2, v4_h3], axis=1)
    res = test_stable(combined_all, labels, "All")
    print(f"  Base+H1+H2+H3: {res['auc']:.4f} ({'✅' if res['auc'] >= base_auc else '❌'} {res['auc']-base_auc:+.4f}) [{res['shape'][1]}D]")
    
    # Top correlations H2
    print("\n📊 TOP H2 CORRELATIONS:")
    h2_names = ['a', 'b', 'aniso', 'c', 'r2', 'res_std', 'rho_low', 'rho_high', 'd_rho',
                'slack_low', 'slack_high', 'd_slack', 'comm_mean', 'comm_max', 'comm_std', 'spear_comm']
    for i, name in enumerate(h2_names[:min(16, v4_h2.shape[1])]):
        corr = np.corrcoef(v4_h2[:, i], labels)[0, 1]
        if abs(corr) > 0.15:
            print(f"  {name:<12}: {corr:+.4f}")

    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
