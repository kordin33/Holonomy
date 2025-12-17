"""
test_v4base_with_prod_h2h3.py - V4 Baseline + Production H2/H3

Cel: Sprawdzić czy Production H2/H3 (te stare, stabilne) dodają wartość do nowego Baseline V4.
Jeśli TAK -> znaczy że nasze nowe V4 H2/H3 są gorsze od starych (popłynęliśmy).
Jeśli NIE -> znaczy że Baseline sam w sobie już zawiera całą informację (sufit CLIPa).
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
from deepfake_guard.features.optimized_v3 import BaselineV3  # Nowy silny Baseline (0.877)
from deepfake_guard.features.production.h2_scale_law import H2_AreaScaleLaw_Fixed  # Production H2
from deepfake_guard.features.production.h3_dispersion import H3_NormalizedDispersionV2  # Production H3
from deepfake_guard.features.production.baseline import extract_minimal_features  # Prod Baseline (ref)

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
    print("🔬 TEST: V4 BASELINE + PRODUCTION H2/H3")
    print("   Checking if old Production H2/H3 add value to new Baseline")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    baseline_v4 = BaselineV3()
    prod_h2 = H2_AreaScaleLaw_Fixed()
    prod_h3 = H3_NormalizedDispersionV2()
    
    # Extract Prod Baseline (Reference)
    print("\n📊 Extracting Prod Baseline (old ref)...")
    prod_base = np.array([extract_minimal_features(encoder, img)['minimal'] 
                          for img in tqdm(images, desc="Prod Base")])
    
    # Extract New V4 Baseline
    print("\n📊 Extracting V4 Baseline (new)...")
    v4_base = []
    for img in tqdm(images, desc="V4 Base"):
        try:
            v4_base.append(baseline_v4.extract_features(encoder, img))
        except:
            v4_base.append(np.zeros(63))
    v4_base = np.array(v4_base, dtype=np.float32)
    
    # Extract Production H2
    print("\n📊 Extracting Production H2...")
    prod_h2_feats = []
    for img in tqdm(images, desc="Prod H2"):
        try:
            prod_h2_feats.append(prod_h2.extract_features(encoder, img))
        except:
            prod_h2_feats.append(np.zeros(12))
    prod_h2_feats = np.array(prod_h2_feats, dtype=np.float32)
    
    # Extract Production H3
    print("\n📊 Extracting Production H3...")
    prod_h3_feats = []
    for img in tqdm(images, desc="Prod H3"):
        try:
            prod_h3_feats.append(prod_h3.extract_features(encoder, img))
        except:
            prod_h3_feats.append(np.zeros(24))
    prod_h3_feats = np.array(prod_h3_feats, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # References
    res = test_stable(prod_base, labels, "Prod_Base_Old")
    old_base_auc = res['auc']
    print(f"\n  Prod_Base (old ref): {old_base_auc:.4f} [{res['shape'][1]}D]")
    
    res = test_stable(v4_base, labels, "V4_Baseline")
    v4_auc = res['auc']
    print(f"  V4_Baseline (new):   {v4_auc:.4f} ({'+' if v4_auc >= old_base_auc else ''}{v4_auc-old_base_auc:+.4f}) [{res['shape'][1]}D]")
    
    # Standalone Production H2/H3
    print("\n📊 STANDALONE (Production):")
    res = test_stable(prod_h2_feats, labels, "Prod_H2")
    print(f"  Prod_H2: {res['auc']:.4f} [{res['shape'][1]}D]")
    
    res = test_stable(prod_h3_feats, labels, "Prod_H3")
    print(f"  Prod_H3: {res['auc']:.4f} [{res['shape'][1]}D]")
    
    # Ablations with V4 Baseline
    print("\n📊 ABLACJE (V4_Baseline + Prod H2/H3):")
    
    combined_v4_h2 = np.concatenate([v4_base, prod_h2_feats], axis=1)
    res = test_stable(combined_v4_h2, labels, "V4+ProdH2")
    delta = res['auc'] - v4_auc
    print(f"  V4_Base + Prod_H2: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    combined_v4_h3 = np.concatenate([v4_base, prod_h3_feats], axis=1)
    res = test_stable(combined_v4_h3, labels, "V4+ProdH3")
    delta = res['auc'] - v4_auc
    print(f"  V4_Base + Prod_H3: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    combined_all = np.concatenate([v4_base, prod_h2_feats, prod_h3_feats], axis=1)
    res = test_stable(combined_all, labels, "V4+ProdH2+H3")
    delta = res['auc'] - v4_auc
    print(f"  V4_Base + Prod_H2 + Prod_H3: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    # Also test:Old Prod Baseline + Prod H2/H3 (sanity check from before)
    print("\n📊 SANITY CHECK (Old Prod_Base + Prod H2/H3):")
    combined_old_all = np.concatenate([prod_base, prod_h2_feats, prod_h3_feats], axis=1)
    res = test_stable(combined_old_all, labels, "ProdBase+H2+H3")
    delta = res['auc'] - old_base_auc
    print(f"  Prod_Base + Prod_H2 + Prod_H3: {res['auc']:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    print("\n" + "="*70)
    print("WNIOSKI:")
    print("  Jeśli V4+ProdH2/H3 > V4 -> Production H2/H3 działają, nasze V4 H2/H3 są gorsze.")
    print("  Jeśli V4+ProdH2/H3 = V4 -> H2/H3 są redundantne (sufit CLIPa).")
    print("="*70)
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
