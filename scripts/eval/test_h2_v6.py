"""
test_h2_v6.py - Test H2_V6 (Log-Map Area + WLS) vs H2_V6_EXP (Richardson)
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
from deepfake_guard.features.h2_v6 import H2_V6
from deepfake_guard.features.h2_v6_exp import H2_V6_EXP
from deepfake_guard.features.optimized_v3 import BaselineV3

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
    features = np.nan_to_num(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]},
                        cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(X_train, y_train)
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

def main():
    print("="*70)
    print("🔬 TEST: H2_V6 (Log-Map WLS) vs H2_V6_EXP (Richardson)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    h2_v6 = H2_V6()
    h2_exp = H2_V6_EXP()
    baseline = BaselineV3()
    
    print("\n📊 Extracting features...")
    feats_base, feats_v6, feats_exp = [], [], []
    
    for img in tqdm(images, desc="Extracting"):
        try:
            feats_base.append(baseline.extract_features(encoder, img))
        except:
            feats_base.append(np.zeros(63))
        try:
            feats_v6.append(h2_v6.extract_features(encoder, img))
        except Exception as e:
            print(f"V6 Error: {e}")
            feats_v6.append(np.zeros(8))
        try:
            feats_exp.append(h2_exp.extract_features(encoder, img))
        except Exception as e:
            print(f"EXP Error: {e}")
            feats_exp.append(np.zeros(12))
    
    feats_base = np.array(feats_base, dtype=np.float32)
    feats_v6 = np.array(feats_v6, dtype=np.float32)
    feats_exp = np.array(feats_exp, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    auc_base = test_stable(feats_base, labels, "Baseline")
    print(f"\n  Baseline_V3: {auc_base:.4f} [63D]")
    
    auc_v6 = test_stable(feats_v6, labels, "H2_V6")
    print(f"  H2_V6 (Log-Map WLS): {auc_v6:.4f} [8D]")
    
    auc_exp = test_stable(feats_exp, labels, "H2_V6_EXP")
    print(f"  H2_V6_EXP (Richardson): {auc_exp:.4f} [12D]")
    
    # Ablations
    print("\n📊 ABLACJE:")
    combined_v6 = np.concatenate([feats_base, feats_v6], axis=1)
    auc_bv6 = test_stable(combined_v6, labels, "Base+V6")
    delta = auc_bv6 - auc_base
    print(f"  Base + H2_V6: {auc_bv6:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    combined_exp = np.concatenate([feats_base, feats_exp], axis=1)
    auc_bexp = test_stable(combined_exp, labels, "Base+EXP")
    delta = auc_bexp - auc_base
    print(f"  Base + H2_V6_EXP: {auc_bexp:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    # Correlations
    print("\n📊 H2_V6 CORRELATIONS:")
    names_v6 = ['kappa', 'lambda', 'R2', 'MAD', 'w_perp', 'area', 'kappa_raw', 'stability']
    for i, name in enumerate(names_v6[:feats_v6.shape[1]]):
        corr = np.corrcoef(feats_v6[:, i], labels)[0, 1]
        if abs(corr) > 0.05:
            print(f"  {name:<12}: {corr:+.4f}")
    
    print("\n📊 H2_V6_EXP CORRELATIONS:")
    names_exp = ['k_mid', 'k_high', 'k_rich', 'scale_ratio', 'scale_diff', 'sym_var', 'k_mid_std', 'k_high_std', 'w_mean', 'w_max', 'w_std', 'w_med']
    for i, name in enumerate(names_exp[:feats_exp.shape[1]]):
        corr = np.corrcoef(feats_exp[:, i], labels)[0, 1]
        if abs(corr) > 0.05:
            print(f"  {name:<12}: {corr:+.4f}")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
