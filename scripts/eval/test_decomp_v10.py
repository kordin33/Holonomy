"""
test_decomp_v10.py - Test V10 Dense Commutator Field
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
from deepfake_guard.features.holonomy_decomposition_v10 import HolonomyDecompositionV10
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
    print("🔬 TEST: DECOMPOSITION V10 - DENSE COMMUTATOR FIELD")
    print("   Baseline Enhanced + 4x4 Commutator Grid")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v10 = HolonomyDecompositionV10()
    
    # Extract
    print("\n📊 Extracting Prod Baseline (Ref)...")
    prod_base = np.array([extract_minimal_features(encoder, img)['minimal'] 
                          for img in tqdm(images, desc="Prod Base")])
    
    print("\n📊 Extracting V10...")
    v10_features = []
    
    for img in tqdm(images, desc="V10"):
        try:
            v10_features.append(v10.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            pass
            
    v10_features = np.array(v10_features, dtype=np.float32)
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    # Ref
    res_b = test_stable(prod_base, labels, "Prod_Base")
    base_auc = res_b['auc']
    print(f"  Reference (Prod_Base): {base_auc:.4f}")
    
    # V10
    res_v10 = test_stable(v10_features, labels, "V10_Decomp")
    delta = res_v10['auc'] - base_auc
    symbol = "✅" if delta >= 0 else "❌"
    print(f"  V10_Decomp: {res_v10['auc']:.4f} ({symbol} {delta:+.4f}) [{v10_features.shape[1]}D]")
    
    # Check correlations of Dense Commutator (last 5 features)
    print("\n📊 DENSE COMMUTATOR CORRELATIONS (Last 5):")
    names = ['mean_comm', 'std_comm', 'max_comm', 'range_comm', 'entropy']
    for i, name in enumerate(names):
        col = v10_features[:, -(5-i)]
        corr = np.corrcoef(col, labels)[0, 1]
        print(f"  {name:<15}: {corr:+.4f}")

    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
