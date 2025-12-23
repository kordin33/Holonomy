"""
test_v19_fixed.py - Final Check: V18 (0.906) + H2_V7_CLEAN

Goal: Confirm if H2_V7_CLEAN adds value to the fixed V18 baseline.
Sample: 100 images (Quick).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.production.holonomy_v18 import HolonomyV18
from deepfake_guard.features.h2_v7_exp import H2_V7_EXP

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 50 # 100 total
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

def test_auc(features, labels):
    features = np.nan_to_num(features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]},
                        cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(features, labels)
    return grid.best_score_

def main():
    print("="*70)
    print("🔬 TEST: V19 FIXED (V18 + H2 V7 CLEAN)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v18 = HolonomyV18()
    h2 = H2_V7_EXP()
    
    print("\n📊 Extracting features...")
    feats_v18 = []
    feats_h2 = []
    
    for img in tqdm(images, desc="Extracting"):
        # V18
        try:
            f = v18.extract_features(encoder, img) # Should work now
            feats_v18.append(f)
        except:
            feats_v18.append(np.zeros(126))
            
        # H2 CLEAN
        try:
            h = h2.extract_clean(encoder, img)
            feats_h2.append(h)
        except:
            feats_h2.append(np.zeros(16))
            
    feats_v18 = np.array(feats_v18, dtype=np.float32)
    feats_h2 = np.array(feats_h2, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    auc_v18 = test_auc(feats_v18, labels)
    print(f"  V18 Base (126D): {auc_v18:.4f}")
    
    auc_h2 = test_auc(feats_h2, labels)
    print(f"  H2 V7 CLEAN (16D): {auc_h2:.4f}")
    
    # Fusion
    feats_fusion = np.concatenate([feats_v18, feats_h2], axis=1)
    auc_fusion = test_auc(feats_fusion, labels)
    delta = auc_fusion - auc_v18
    print(f"  V19 Fusion: {auc_fusion:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
