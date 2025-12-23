"""
test_consistency_check.py - Verify Determinism of V18 SOTA

Goal: Check if 0.9058 was a fluke due to unordered globbing.
We use sorted() file lists to ensure identical dataset across runs.
Comparing:
1. V18 Production (Global + PatchMean)
2. V18 Optimized (MegaBatch)

If they match, we have a solid baseline to improve upon.
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
from deepfake_guard.features.production.holonomy_v18 import HolonomyV18
from deepfake_guard.features.holonomy_v18_opt import HolonomyV18Opt

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 200
RANDOM_STATE = 42

def load_data(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        # CRITICAL: Sorted for determinism
        files = sorted(list((DATA_DIR / "test" / cls).glob("*.jpg")))[:n_per_class]
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
    print("🔬 CONSISTENCY CHECK: V18 PROD vs OPT")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v18_prod = HolonomyV18()
    v18_opt = HolonomyV18Opt()
    
    print("\n📊 Extracting PROD...")
    feats_prod = []
    for img in tqdm(images):
        feats_prod.append(v18_prod.extract_features(encoder, img))
    feats_prod = np.array(feats_prod, dtype=np.float32)
    
    print("\n📊 Extracting OPT...")
    feats_opt = []
    for img in tqdm(images):
        feats_opt.append(v18_opt.extract_features(encoder, img))
    feats_opt = np.array(feats_opt, dtype=np.float32)
    
    # Check numerical equality
    diff = np.mean(np.abs(feats_prod - feats_opt))
    print(f"\nMean Absolute Difference (Prod - Opt): {diff:.6f}")
    if diff > 1e-4:
        print("⚠️ WARNING: Significant numerical difference!")
    else:
        print("✅ Numerical Match.")
        
    print("\nRESULTS (AUC):")
    auc_prod = test_auc(feats_prod, labels, "PROD")
    auc_opt = test_auc(feats_opt, labels, "OPT")
    
    print(f"  V18 PROD: {auc_prod:.4f}")
    print(f"  V18 OPT:  {auc_opt:.4f}")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
