"""
test_v21_quick.py - Check V21 Performance (Grid 3x3 + Centroids).
Sample: 100 images.
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
from deepfake_guard.features.holonomy_v21 import HolonomyV21

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 50 
RANDOM_STATE = 42

def load_data(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = sorted(list((DATA_DIR / "test" / cls).glob("*.jpg")))[:n_per_class]
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
    print("🔬 TEST: V21 (Grid 3x3 + Centroids)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v21 = HolonomyV21()
    feats = []
    
    print("Extracting (Mega-Inference)...")
    for img in tqdm(images):
        feats.append(v21.extract_features(encoder, img))
    
    feats = np.array(feats, dtype=np.float32)
    auc = test_auc(feats, labels)
    
    print(f"\n  V21 (180D): {auc:.4f}")
    if auc > 0.90:
        print("🚀 SUCCESS: BROKEN 0.90 BARRIER!")
    else:
        print("😐 Still below 0.90.")

if __name__ == "__main__":
    main()
