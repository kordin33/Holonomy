"""
test_decomp_v12.py - Test V12 Global + Patch Baseline Ensemble
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
from deepfake_guard.features.holonomy_decomposition_v12 import HolonomyDecompositionV12
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
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]},
                        cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(X_train, y_train)
    
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return {'name': name, 'auc': roc_auc_score(y_test, y_prob), 'shape': features.shape}


def main():
    print("="*70)
    print("🔬 TEST: DECOMPOSITION V12 - GLOBAL + PATCH BASELINE ENSEMBLE")
    print("   Target: > 0.90 (The Final Push)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v12 = HolonomyDecompositionV12()
    
    # Extract
    print("\n📊 Extracting Prod Baseline (Ref)...")
    prod_base = np.array([extract_minimal_features(encoder, img)['minimal'] 
                          for img in tqdm(images, desc="Prod Base")])
    
    print("\n📊 Extracting V12...")
    v12_features = []
    
    for img in tqdm(images, desc="V12"):
        try:
            v12_features.append(v12.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            v12_features.append(np.zeros(252))
            
    v12_features = np.array(v12_features, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Ref
    res = test_stable(prod_base, labels, "Prod_Base")
    base_auc = res['auc']
    print(f"\n  Reference (Prod_Base): {base_auc:.4f}")
    
    # V12
    res = test_stable(v12_features, labels, "V12_Ensemble")
    delta = res['auc'] - base_auc
    print(f"  V12_Ensemble: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    # Test sub-components
    # Global only (first 63)
    global_only = v12_features[:, :63]
    res = test_stable(global_only, labels, "V12_Global")
    delta = res['auc'] - base_auc
    print(f"  V12_Global_Only: {res['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{res['shape'][1]}D]")
    
    # Disagreement (63:126)
    disagree = v12_features[:, 63:126]
    res = test_stable(disagree, labels, "V12_Disagreement")
    print(f"  V12_Disagreement_Only: {res['auc']:.4f} [{res['shape'][1]}D]")
    
    # Gap (126:189)
    gap = v12_features[:, 126:189]
    res = test_stable(gap, labels, "V12_Gap")
    print(f"  V12_Gap_Only: {res['auc']:.4f} [{res['shape'][1]}D]")
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
