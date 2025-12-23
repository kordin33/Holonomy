"""
test_residual_fusion.py - Orthogonal Fusion of SOTA Components

Strategy:
Instead of stacking raw features (which correlates with Global),
we stack "Innovation": Residuals of features that Global cannot explain.

1. Train Ridge: Feature ~ Global
2. Residual = Feature - Ridge.predict(Global)
3. Stack Global + Residuals
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
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.production.final_v12 import FinalV12Ensemble 
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

def compute_residuals(X_target, X_base):
    """
    Regress X_target on X_base and return residuals.
    residual = target - predict(base)
    """
    # Fit on all data (unsupervised clean-up style) or train-only?
    # To be safe and prevent leak, fit on X_base.
    # But for feature engineering simpler to fit on whole set if unsupervised.
    # Ridge is supervised by X_target values but X_target is not label.
    # It's geometric orthogonalization.
    
    res_list = []
    # Normalize base for numerical stability
    scaler = StandardScaler()
    X_base_sc = scaler.fit_transform(X_base)
    
    for i in range(X_target.shape[1]):
        y = X_target[:, i]
        model = Ridge(alpha=1.0)
        model.fit(X_base_sc, y)
        y_pred = model.predict(X_base_sc)
        res = y - y_pred
        res_list.append(res)
        
    return np.column_stack(res_list)

def main():
    print("="*70)
    print("🔬 TEST: RESIDUAL FUSION (Orthogonal Stacking)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v12 = FinalV12Ensemble()
    h2_exp = H2_V6_EXP()
    
    print("\n📊 Extracting features...")
    feats_base = []
    feats_h2 = []
    feats_comm = []
    
    for img in tqdm(images, desc="Extracting"):
        try:
            f = v12._compute_baseline_features(encoder, img)
            feats_base.append(f)
        except:
            feats_base.append(np.zeros(63))
        try:
            f = h2_exp.extract_features(encoder, img)
            feats_h2.append(f)
        except:
            feats_h2.append(np.zeros(12))
        try:
            f = compute_commutator_features_slim(encoder, img)
            feats_comm.append(f)
        except:
            feats_comm.append(np.zeros(6))
            
    feats_base = np.array(feats_base, dtype=np.float32)
    feats_h2 = np.array(feats_h2, dtype=np.float32)
    feats_comm = np.array(feats_comm, dtype=np.float32)
    
    # Orthogonalize H2 vs Base
    print("\n🧹 Orthogonalizing H2 vs Base...")
    feats_h2_res = compute_residuals(feats_h2, feats_base)
    
    # Orthogonalize Comm vs Base
    print("🧹 Orthogonalizing Comm vs Base...")
    feats_comm_res = compute_residuals(feats_comm, feats_base)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    auc_base = test_auc(feats_base, labels, "Baseline")
    print(f"  Baseline (63D): {auc_base:.4f}")
    
    # Naive Combo (Reference)
    comb_naive = np.concatenate([feats_base, feats_h2, feats_comm], axis=1)
    auc_naive = test_auc(comb_naive, labels, "Naive_Combo")
    print(f"  Naive Combo: {auc_naive:.4f}")
    
    # Residual Combo
    comb_res = np.concatenate([feats_base, feats_h2_res, feats_comm_res], axis=1)
    auc_res = test_auc(comb_res, labels, "Residual_Combo")
    delta = auc_res - auc_base
    print(f"  Residual Combo: {auc_res:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
