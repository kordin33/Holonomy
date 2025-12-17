"""
test_decomp_v6.py - Test V6 Grand Unified
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
from deepfake_guard.features.holonomy_decomposition_v6 import HolonomyDecompositionV6
from deepfake_guard.features.production.baseline import extract_minimal_features

# PRODUCTION (dla porównania)
from deepfake_guard.features.production.h3_dispersion import H3_NormalizedDispersionV2
from deepfake_guard.features.production.h2_scale_law import H2_AreaScaleLaw_Fixed

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
    print("🔬 TEST: DECOMPOSITION V6 - GRAND UNIFIED")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v6 = HolonomyDecompositionV6()
    
    # Extract Production Baseline (Reference)
    print("\n📊 Extracting Prod Baseline (Ref)...")
    prod_base = np.array([extract_minimal_features(encoder, img)['minimal'] 
                          for img in tqdm(images, desc="Prod Base")])
    
    # Extract V6
    print("\n📊 Extracting V6 (Grand Unified)...")
    v6_features = {
        'baseline': [], 'commutator': [], 'h3': [], 'h2': [], 'all': []
    }
    
    for img in tqdm(images, desc="V6"):
        try:
            feats = v6.extract_features(encoder, img)
            for k in v6_features:
                v6_features[k].append(feats[k])
        except Exception as e:
            print(f"Error: {e}")
            # Add zeros if fail
            # (Just assume dimensions based on one run if needed, but here we skip to keep simple)
            pass
            
    for k in v6_features:
        v6_features[k] = np.array(v6_features[k], dtype=np.float32)
        
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    # Prod Reference
    res_b = test_stable(prod_base, labels, "Prod_Base")
    base_auc = res_b['auc']
    print(f"  Reference (Prod_Base): {base_auc:.4f}")
    
    # V6 Components
    for key in ['baseline', 'commutator', 'h3', 'h2', 'all']:
        name = f"V6_{key}"
        res = test_stable(v6_features[key], labels, name)
        delta = res['auc'] - base_auc
        symbol = "✅" if delta >= 0 else "❌"
        print(f"  {name:<15}: {res['auc']:.4f} ({symbol} {delta:+.4f}) [{res['shape'][1]}D]")
        
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
