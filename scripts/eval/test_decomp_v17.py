"""
test_decomp_v17.py - Test V17.0 (Global-Anchored Disagreement)
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
from deepfake_guard.features.holonomy_decomposition_v17 import HolonomyDecompositionV17

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 200 # Keeping sample size consistent
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
    print("🔬 TEST: V17.0 - GLOBAL-ANCHORED DISAGREEMENT")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v17 = HolonomyDecompositionV17()
    
    print("\n📊 Extracting features...")
    features = []
    for img in tqdm(images, desc="V17"):
        try:
            features.append(v17.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            features.append(np.zeros(153))
    
    features = np.array(features, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # 1. Global Only (Baseline check)
    auc_global = test_auc(features[:, :63], labels, "Global_Only")
    print(f"  Global_Only (63D): {auc_global:.4f}")

    # 2. Disagreement Only (27D)
    auc_dis = test_auc(features[:, 63:90], labels, "Dis_Only")
    print(f"  Disagreement_V17 (27D): {auc_dis:.4f}")

    # 3. Global + Disagreement (90D)
    features_gd = np.concatenate([features[:, :63], features[:, 63:90]], axis=1)
    auc_gd = test_auc(features_gd, labels, "Global+Dis")
    delta = auc_gd - auc_global
    print(f"  Global + Disagreement: {auc_gd:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    # 4. Global + Patch Mean (Traditional Ensemble, subset of V12)
    features_gp = np.concatenate([features[:, :63], features[:, 90:153]], axis=1)
    auc_gp = test_auc(features_gp, labels, "Global+PatchMean")
    print(f"  Global + PatchMean (126D): {auc_gp:.4f}")

    # 5. Full V17 set (Global + Dis + PatchMean)
    auc_full = test_auc(features, labels, "V17_Full")
    print(f"  V17_Full (153D): {auc_full:.4f}")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
