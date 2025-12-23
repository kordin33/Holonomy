"""
test_v19_frontier.py - V18 Base (126D) + H2 V7 EXP CLEAN (16D)

Goal: Check if extracting ONLY "Shape" features from H2 (orthogonality by design) 
improves the SOTA 0.9058 score.
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
from deepfake_guard.features.production.holonomy_v18 import HolonomyV18 # The SOTA class
from deepfake_guard.features.h2_v7_exp import H2_V7_EXP

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
    print("🔬 TEST: V19 FRONTIER (V18 Base + H2 V7 CLEAN)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v18_base = HolonomyV18()
    h2_v7 = H2_V7_EXP()
    
    print("\n📊 Extracting features...")
    feats_v18 = []
    feats_h2_clean = []
    
    # We will run slow extraction here because optimization logic is complex to re-implement for hybrid
    # But 200 images is bearable (~15-20 min).
    
    for img in tqdm(images, desc="Extracting"):
        try:
            f18 = v18.extract_features(encoder, img)
            feats_v18.append(f18)
        except Exception as e:
            feats_v18.append(np.zeros(126))
            
        try:
            h2c = h2_v7.extract_clean(encoder, img)
            feats_h2_clean.append(h2c)
        except:
            feats_h2_clean.append(np.zeros(16))
            
    feats_v18 = np.array(feats_v18, dtype=np.float32)
    feats_h2_clean = np.array(feats_h2_clean, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    auc_v18 = test_auc(feats_v18, labels, "V18_Base")
    print(f"  V18_Base (126D): {auc_v18:.4f}")
    
    auc_h2c = test_auc(feats_h2_clean, labels, "H2_V7_CLEAN")
    print(f"  H2_V7_CLEAN (16D): {auc_h2c:.4f}")
    
    # Fusion
    feats_fusion = np.concatenate([feats_v18, feats_h2_clean], axis=1)
    auc_fusion = test_auc(feats_fusion, labels, "V19_Frontier")
    delta = auc_fusion - auc_v18
    print(f"  V19 Frontier (142D): {auc_fusion:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
