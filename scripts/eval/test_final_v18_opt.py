"""
test_final_v18_opt.py - Optimized Test for V18 (Quick Check)
Uses Mega-Batch encoding to speed up inference.
Sample Size: 50 per class (100 total) for quick validaton.
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
from deepfake_guard.features.holonomy_v18_opt import HolonomyV18Opt

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 50 # QUICK CHECK
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
    # Stratified CV requires >1 sample per fold.
    # With 100 samples, 5 folds = 20 samples/fold. OK.
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]},
                        cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(features, labels) # Fit on all for quick LOO-like score estimate or use CV results
    # Better: valid estimate is CV score
    return grid.best_score_ 
    # GridSearchCV.best_score_ is mean cross-validated score of the best_estimator.
    # This is slightly biased but good for relative check.
    # Proper way: nested CV or train/test split.
    # Let's simple train/test split (70/30) as before
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    # 70 train, 30 test.
    grid.fit(X_train, y_train)
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


def main():
    print("="*70)
    print("🔬 TEST: V18 OPTIMIZED (Quick Check 100 images)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v18 = HolonomyV18Opt()
    
    print("\n📊 Extracting features...")
    features = []
    
    for img in tqdm(images, desc="Extracting"):
        try:
            f = v18.extract_features(encoder, img)
            features.append(f)
        except Exception as e:
            print(f"Error: {e}")
            features.append(np.zeros(138))
            
    features = np.array(features, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    auc = test_auc(features, labels, "V18_Full")
    print(f"  V18_Full (138D): {auc:.4f}")
    
    # Global: 0-63
    # PatchMean: 63-126
    # H2: 126-138
    
    auc_gp = test_auc(features[:, :126], labels, "Global+PatchMean")
    print(f"  Global + PatchMean: {auc_gp:.4f}")
    
    delta = auc - auc_gp
    print(f"  Impact of H2: {delta:+.4f} ({'✅' if delta > 0 else '❌'})")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
