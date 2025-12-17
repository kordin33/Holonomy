"""
test_decomp_v14.py - Test V14 Fixed Grid + Spectral + Weighted
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
from deepfake_guard.features.holonomy_decomposition_v14 import HolonomyDecompositionV14
from deepfake_guard.features.production.baseline import extract_minimal_features

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
    print("🔬 TEST: DECOMPOSITION V14 - FIXED GRID + SPECTRAL + WEIGHTED")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v14 = HolonomyDecompositionV14()
    
    print("\n📊 Extracting features...")
    features = []
    for img in tqdm(images, desc="V14"):
        try:
            features.append(v14.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            features.append(np.zeros(285))
    
    features = np.array(features, dtype=np.float32)
    
    # Test full
    auc_full = test_stable(features, labels, "V14_Full")
    print(f"\n  V14_Full: {auc_full:.4f} [{features.shape[1]}D]")
    
    # Component tests
    auc_global = test_stable(features[:, :81], labels, "Global_Loop")
    print(f"  Global_Loop (81D): {auc_global:.4f}")
    
    auc_spectral = test_stable(features[:, 81:117], labels, "Global_Spectral")
    print(f"  Global_Spectral (36D): {auc_spectral:.4f}")
    
    auc_disagree = test_stable(features[:, -6:], labels, "Disagreement")
    print(f"  Disagreement (6D): {auc_disagree:.4f}")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
