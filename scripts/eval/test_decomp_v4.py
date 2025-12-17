"""
test_decomp_v4.py - Test Path-Ordered Frame Transport
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
from deepfake_guard.features.holonomy_decomposition_v4 import (
    HolonomyDecompositionV4,
    HolonomyDecompositionV4Multi,
)
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

DATA_DIR = Path("./data/cifake")
SAMPLE_SIZE = 150
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


def test_with_stable_pipeline(features, labels, name):
    """STABILNY test z StratifiedKFold."""
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]}, 
                        cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(X_train, y_train)
    
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return {'name': name, 'auc': roc_auc_score(y_test, y_prob), 'shape': features.shape}


def main():
    print("="*70)
    print("🔬 TEST: Decomposition V4 - Path-Ordered Frame Transport")
    print("   PRAWDZIWA holonomia: Q_loop = Q_0 @ Q_1 @ ... @ Q_{m-1}")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v4 = HolonomyDecompositionV4(frame_dim=6)
    v4_multi = HolonomyDecompositionV4Multi(frame_dim=6)
    
    # Extract
    print("\n📊 Extracting Baseline...")
    baseline_features = np.array([extract_minimal_features(encoder, img)['minimal'] 
                                  for img in tqdm(images, desc="Baseline")])
    
    print("\n📊 Extracting V4 (single loop)...")
    v4_features = []
    for img in tqdm(images, desc="V4"):
        try:
            v4_features.append(v4.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            v4_features.append(np.zeros(6, dtype=np.float32))
    v4_features = np.array(v4_features)
    
    print("\n📊 Extracting V4 Multi (2 loops)...")
    v4m_features = []
    for img in tqdm(images, desc="V4 Multi"):
        try:
            v4m_features.append(v4_multi.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            v4m_features.append(np.zeros(8, dtype=np.float32))
    v4m_features = np.array(v4m_features)
    
    # Test
    print("\n" + "="*70)
    print("TESTING (STABLE CV)")
    print("="*70)
    
    res_b = test_with_stable_pipeline(baseline_features, labels, "Baseline")
    baseline_auc = res_b['auc']
    print(f"\n  Baseline: AUC = {baseline_auc:.4f} ({baseline_features.shape[1]}D)")
    
    res_v4 = test_with_stable_pipeline(v4_features, labels, "V4")
    delta = res_v4['auc'] - baseline_auc
    print(f"  V4: AUC = {res_v4['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{v4_features.shape[1]}D]")
    
    res_v4m = test_with_stable_pipeline(v4m_features, labels, "V4_Multi")
    delta = res_v4m['auc'] - baseline_auc
    print(f"  V4 Multi: AUC = {res_v4m['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f}) [{v4m_features.shape[1]}D]")
    
    # Combined
    combined = np.concatenate([baseline_features, v4_features], axis=1)
    res_c = test_with_stable_pipeline(combined, labels, "Baseline+V4")
    delta = res_c['auc'] - baseline_auc
    print(f"  Baseline+V4: AUC = {res_c['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f})")
    
    # Feature analysis
    print("\n📊 Analiza cech V4:")
    names = v4.get_feature_names()
    for i, name in enumerate(names):
        col = v4_features[:, i]
        corr = np.corrcoef(col, labels)[0, 1]
        mr = col[labels == 1].mean()
        mf = col[labels == 0].mean()
        star = " ⭐" if abs(corr) > 0.15 else ""
        print(f"  {name:<20} corr={corr:>+.4f}  real={mr:.4f}  fake={mf:.4f}{star}")
    
    del encoder
    torch.cuda.empty_cache()
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
