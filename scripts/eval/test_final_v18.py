"""
test_final_v18.py - The "Simple & Strong" Ensemble
Based on discovery that Global+PatchMean (0.8961) outperforms full V12 (0.8850).

Features:
1. Global Baseline (63D)
2. Patch Mean (63D)
3. H2_V6_EXP (12D) - Richardson Curvature

Total: 138D.
Target: > 0.90 AUC.
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
from deepfake_guard.features.holonomy_decomposition_v17 import compute_baseline_features, get_v12_patches
from deepfake_guard.features.h2_v6_exp import H2_V6_EXP

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
    print("🔬 TEST: V18 (Global + PatchMean + H2_V6_EXP)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    h2_exp = H2_V6_EXP()
    
    print("\n📊 Extracting features...")
    feats_global = []
    feats_patch_mean = []
    feats_h2 = []
    
    for img in tqdm(images, desc="Extracting"):
        # 1. Base (Global + PatchMean)
        try:
            g = compute_baseline_features(encoder, img)
            patches = get_v12_patches(img)
            p_feats = np.array([compute_baseline_features(encoder, p) for p in patches])
            pm = np.mean(p_feats, axis=0)
            
            feats_global.append(g)
            feats_patch_mean.append(pm)
        except:
            feats_global.append(np.zeros(63))
            feats_patch_mean.append(np.zeros(63))
            
        # 2. H2 V6 EXP
        try:
            f = h2_exp.extract_features(encoder, img)
            feats_h2.append(f)
        except:
            feats_h2.append(np.zeros(12))
            
    feats_global = np.array(feats_global, dtype=np.float32)
    feats_patch_mean = np.array(feats_patch_mean, dtype=np.float32)
    feats_h2 = np.array(feats_h2, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # 1. Global + PatchMean (The Discovery)
    comb_gp = np.concatenate([feats_global, feats_patch_mean], axis=1)
    auc_gp = test_auc(comb_gp, labels, "Global+PatchMean")
    print(f"  Global + PatchMean (126D): {auc_gp:.4f}")
    
    # 2. Full V18
    comb_v18 = np.concatenate([feats_global, feats_patch_mean, feats_h2], axis=1)
    auc_v18 = test_auc(comb_v18, labels, "V18_Full")
    delta = auc_v18 - auc_gp
    print(f"  V18 (Global+PM+H2): {auc_v18:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    if auc_v18 > 0.90:
        print("\n🚀 MISSION ACCOMPLISHED: > 0.90 AUC!")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
