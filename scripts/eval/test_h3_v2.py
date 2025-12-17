"""
test_h3_v2.py - Test H3 V2 (FIXED)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.h3_normalized_dispersion_v2 import (
    H3_NormalizedDispersionV2,
    H3_NormalizedDispersionV2Fast,
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


def test_with_pipeline(features, labels, name):
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])
    grid = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]}, 
                        cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return {'name': name, 'auc': roc_auc_score(y_test, y_prob), 'shape': features.shape}


def main():
    print("="*70)
    print("🔬 TEST: H3 V2 (NAPRAWIONY)")
    print("   L2 norm + cosine path + grid patches")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    h3_v2 = H3_NormalizedDispersionV2()
    
    # Extract
    print("\n📊 Extracting Baseline...")
    baseline_features = np.array([extract_minimal_features(encoder, img)['minimal'] 
                                  for img in tqdm(images, desc="Baseline")])
    
    print("\n📊 Extracting H3 V2...")
    h3_features = []
    for img in tqdm(images, desc="H3 V2"):
        try:
            h3_features.append(h3_v2.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            h3_features.append(np.zeros(9, dtype=np.float32))
    h3_features = np.array(h3_features)
    
    # Test
    print("\n" + "="*70)
    res_b = test_with_pipeline(baseline_features, labels, "Baseline")
    baseline_auc = res_b['auc']
    print(f"  Baseline: AUC = {baseline_auc:.4f}")
    
    res_h3 = test_with_pipeline(h3_features, labels, "H3_V2")
    delta = res_h3['auc'] - baseline_auc
    print(f"  H3 V2: AUC = {res_h3['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f})")
    
    combined = np.concatenate([baseline_features, h3_features], axis=1)
    res_c = test_with_pipeline(combined, labels, "Baseline+H3_V2")
    delta = res_c['auc'] - baseline_auc
    print(f"  Baseline+H3 V2: AUC = {res_c['auc']:.4f} ({'✅' if delta >= 0 else '❌'} {delta:+.4f})")
    
    # Feature analysis
    print("\n📊 Cechy H3 V2:")
    names = h3_v2.get_feature_names()
    for i, name in enumerate(names):
        col = h3_features[:, i]
        corr = np.corrcoef(col, labels)[0, 1]
        print(f"  {name:<25} corr={corr:>+.4f}")
    
    del encoder
    torch.cuda.empty_cache()
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
