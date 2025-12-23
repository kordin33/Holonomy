"""
test_sota_combo.py - Combine Best Components: Baseline + H2_V6_EXP + Commutator_Slim

Components:
1. Baseline (63D) - Global Holonomy (SOTA base)
2. H2_V6_EXP (12D) - Richardson Curvature (proven +0.010)
3. Commutator Slim (6D) - Best comm pairs (proven +0.003)

Goal: Check if they stack to > 0.89.
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
from deepfake_guard.features.production.final_v12 import FinalV12Ensemble # use internal baseline
from deepfake_guard.features.h2_v6_exp import H2_V6_EXP
from deepfake_guard.features.holonomy_decomposition_v16 import compute_commutator_features_slim

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

def test_auc(features, labels, name):
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
    print("🔬 TEST: SOTA COMBO (Base + H2_V6_EXP + Comm_Slim)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extractors
    v12 = FinalV12Ensemble() # using generic baseline method inside
    h2_exp = H2_V6_EXP()
    
    print("\n📊 Extracting features...")
    feats_base = []
    feats_h2 = []
    feats_comm = []
    
    for img in tqdm(images, desc="Extracting"):
        # 1. Baseline
        try:
            f = v12._compute_baseline_features(encoder, img)
            feats_base.append(f)
        except:
            feats_base.append(np.zeros(63))
            
        # 2. H2 V6 EXP
        try:
            f = h2_exp.extract_features(encoder, img)
            feats_h2.append(f)
        except:
            feats_h2.append(np.zeros(12))
            
        # 3. Commutator Slim
        try:
            f = compute_commutator_features_slim(encoder, img)
            feats_comm.append(f)
        except:
            feats_comm.append(np.zeros(6))
            
    feats_base = np.array(feats_base, dtype=np.float32)
    feats_h2 = np.array(feats_h2, dtype=np.float32)
    feats_comm = np.array(feats_comm, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Check individuals
    auc_base = test_auc(feats_base, labels, "Baseline")
    print(f"  Baseline (63D): {auc_base:.4f}")
    
    auc_h2 = test_auc(feats_h2, labels, "H2_V6_EXP")
    print(f"  H2_V6_EXP (12D): {auc_h2:.4f}")

    auc_comm = test_auc(feats_comm, labels, "Comm_Slim")
    print(f"  Comm_Slim (6D): {auc_comm:.4f}")
    
    # Pairs
    comb_bh2 = np.concatenate([feats_base, feats_h2], axis=1)
    auc_bh2 = test_auc(comb_bh2, labels, "Base+H2")
    print(f"  Base + H2: {auc_bh2:.4f} (Delta: {auc_bh2-auc_base:+.4f})")
    
    comb_bc = np.concatenate([feats_base, feats_comm], axis=1)
    auc_bc = test_auc(comb_bc, labels, "Base+Comm")
    print(f"  Base + Comm: {auc_bc:.4f} (Delta: {auc_bc-auc_base:+.4f})")
    
    # Full Combo
    comb_all = np.concatenate([feats_base, feats_h2, feats_comm], axis=1)
    auc_all = test_auc(comb_all, labels, "ALL (Base+H2+Comm)")
    delta = auc_all - auc_base
    print(f"  ALL COMBO: {auc_all:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
