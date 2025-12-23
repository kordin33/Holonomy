"""
test_v18_operational.py - Comprehensive Operational Test of V18

Components to test:
1. Global (63D) - Baseline
2. PatchMean (63D) - Aggregated local holonomy
3. Disagreement (63D) - Patch Std (Previously removed, verifying removal)
4. H2_CLEAN (16D) - Pure Shape/Curvature

Hypotheses:
H1: Base + H2_CLEAN > Base (H2 adds orthogonal value)
H2: Base + PatchMean > Base (PatchMean adds robustness)
H3: Base + PatchMean + H2_CLEAN is the optimal set (SOTA)
H4: Disagreement adds noise and should remain removed.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
# Importing specific functions to build custom assemblies
from deepfake_guard.features.production.holonomy_v18 import (
    compute_baseline_features, 
    compute_h2_clean
)

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 200 # 200 per class = 400 total (Standard Benchmark)
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

def get_patches(image):
    w, h = image.size
    ps = min(w, h) // 2
    patches = [
        image.crop((0, 0, ps, ps)),
        image.crop((w-ps, 0, w, ps)),
        image.crop((0, h-ps, ps, h)),
        image.crop((w-ps, h-ps, w, h)),
        image.crop(((w-ps)//2, (h-ps)//2, (w+ps)//2, (h+ps)//2))
    ]
    return [p.resize((224, 224), Image.LANCZOS) for p in patches]

def test_auc(features, labels, name):
    features = np.nan_to_num(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
    
    # Simple grid search to be fair to larger feature sets
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 0.01]},
                        cv=cv, scoring='roc_auc', n_jobs=1)
    
    grid.fit(X_train, y_train)
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    return score

def main():
    print("="*70)
    print("🔬 OPERATIONAL TEST: V18 COMPONENTS & ABLATIONS")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    print("\n📊 Extracting Raw Components...")
    
    feats_global = []
    feats_patch_mean = []
    feats_patch_std = [] # Disagreement
    feats_h2 = []
    
    for img in tqdm(images, desc="Extracting"):
        # 1. Global
        try:
            g = compute_baseline_features(encoder, img)
        except:
            g = np.zeros(63)
        feats_global.append(g)
        
        # 2. Patches (Mean & Std)
        try:
            patches = get_patches(img)
            p_feats = [compute_baseline_features(encoder, p) for p in patches]
            p_feats = np.array(p_feats)
            
            p_mean = np.mean(p_feats, axis=0)
            p_std = np.std(p_feats, axis=0)
        except:
            p_mean = np.zeros(63)
            p_std = np.zeros(63)
        
        feats_patch_mean.append(p_mean)
        feats_patch_std.append(p_std)
        
        # 3. H2 CLEAN
        try:
            h = compute_h2_clean(encoder, img)
        except:
            h = np.zeros(16)
        feats_h2.append(h)
            
    # Convert to arrays
    F_global = np.array(feats_global, dtype=np.float32)
    F_p_mean = np.array(feats_patch_mean, dtype=np.float32)
    F_p_std = np.array(feats_patch_std, dtype=np.float32)
    F_h2 = np.array(feats_h2, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS (AUC)")
    print("="*70)
    
    # 1. BASELINE
    auc_base = test_auc(F_global, labels, "Base")
    print(f"1. Base (Global 63D):              {auc_base:.4f}")
    
    # 2. H2 IMPACT
    F_base_h2 = np.concatenate([F_global, F_h2], axis=1)
    auc_bh2 = test_auc(F_base_h2, labels, "Base+H2")
    print(f"2. Base + H2_CLEAN (79D):        {auc_bh2:.4f} (Delta: {auc_bh2-auc_base:+.4f})")
    
    # 3. PATCH MEAN IMPACT (V18 CORE)
    F_v18_core = np.concatenate([F_global, F_p_mean], axis=1)
    auc_core = test_auc(F_v18_core, labels, "V18_Core")
    print(f"3. V18_Core (Global+PM 126D):    {auc_core:.4f} (Delta: {auc_core-auc_base:+.4f})")
    
    # 4. FULL V18 (Proposed SOTA)
    F_v18_full = np.concatenate([F_global, F_p_mean, F_h2], axis=1)
    auc_full = test_auc(F_v18_full, labels, "V18_Full")
    print(f"4. V18_Full (Core+H2 142D):      {auc_full:.4f} (Delta vs Core: {auc_full-auc_core:+.4f})")
    
    # 5. DISAGREEMENT CHECK (Is it toxic?)
    F_kitchen_sink = np.concatenate([F_global, F_p_mean, F_p_std, F_h2], axis=1)
    auc_all = test_auc(F_kitchen_sink, labels, "Kitchen_Sink")
    print(f"5. V18 + Disagreement (205D):    {auc_all:.4f} (Delta vs Full: {auc_all-auc_full:+.4f})")
    
    print("\n" + "="*70)
    print("H2_CLEAN STANDALONE")
    print("="*70)
    auc_h2_only = test_auc(F_h2, labels, "H2_Only")
    print(f"  H2_CLEAN Only (16D):           {auc_h2_only:.4f}")

    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
