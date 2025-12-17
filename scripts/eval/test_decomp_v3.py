"""
test_decomp_v3.py - Test Decomposition V3 + Ablacje
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
from deepfake_guard.features.holonomy_decomposition_v3 import (
    HolonomyDecompositionV3,
    DecompV3_Minimal,
    DecompV3_Loop0Only,
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
    print("🔬 TEST: Decomposition V3 + ABLACJE")
    print("   subspace_dist z S², geometria chmury, loop0 only")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extractors
    decomp_full = HolonomyDecompositionV3(use_all_loops=True)
    decomp_minimal = DecompV3_Minimal(use_all_loops=True)
    decomp_loop0 = DecompV3_Loop0Only()
    
    # Extract
    print("\n📊 Extracting features...")
    
    # Baseline
    baseline_features = np.array([extract_minimal_features(encoder, img)['minimal'] 
                                  for img in tqdm(images, desc="Baseline")])
    
    # V3 Full
    v3_full = []
    for img in tqdm(images, desc="V3 Full"):
        try:
            v3_full.append(decomp_full.extract_features(encoder, img))
        except Exception as e:
            v3_full.append(np.zeros(18, dtype=np.float32))
    v3_full = np.array(v3_full)
    
    # V3 Minimal (H_trans + theta_max only)
    v3_minimal = []
    for img in tqdm(images, desc="V3 Minimal"):
        try:
            v3_minimal.append(decomp_minimal.extract_features(encoder, img))
        except Exception as e:
            v3_minimal.append(np.zeros(6, dtype=np.float32))
    v3_minimal = np.array(v3_minimal)
    
    # V3 Loop0 Only
    v3_loop0 = []
    for img in tqdm(images, desc="V3 Loop0"):
        try:
            v3_loop0.append(decomp_loop0.extract_features(encoder, img))
        except Exception as e:
            v3_loop0.append(np.zeros(6, dtype=np.float32))
    v3_loop0 = np.array(v3_loop0)
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    res_b = test_with_pipeline(baseline_features, labels, "Baseline")
    baseline_auc = res_b['auc']
    print(f"\n  Baseline: AUC = {baseline_auc:.4f} ({baseline_features.shape[1]}D)")
    
    results = []
    for name, feat in [
        ("V3_Full", v3_full),
        ("V3_Minimal", v3_minimal),
        ("V3_Loop0", v3_loop0),
    ]:
        res = test_with_pipeline(feat, labels, name)
        delta = res['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "❌"
        print(f"  {name}: AUC = {res['auc']:.4f} ({symbol} {delta:+.4f}) [{feat.shape[1]}D]")
        results.append(res)
    
    # Combined
    print("\n🧪 Kombinacje:")
    
    for name, feat in [
        ("Baseline+V3_Full", np.concatenate([baseline_features, v3_full], axis=1)),
        ("Baseline+V3_Minimal", np.concatenate([baseline_features, v3_minimal], axis=1)),
        ("Baseline+V3_Loop0", np.concatenate([baseline_features, v3_loop0], axis=1)),
    ]:
        res = test_with_pipeline(feat, labels, name)
        delta = res['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "❌"
        print(f"  {name}: AUC = {res['auc']:.4f} ({symbol} {delta:+.4f}) [{feat.shape[1]}D]")
    
    # Feature analysis V3_Full
    print("\n📊 Analiza cech V3_Full:")
    names = decomp_full.get_feature_names()
    for i, name in enumerate(names):
        col = v3_full[:, i]
        corr = np.corrcoef(col, labels)[0, 1]
        if abs(corr) > 0.15:
            print(f"  {name:<25} corr={corr:>+.4f} ⭐")
        else:
            print(f"  {name:<25} corr={corr:>+.4f}")
    
    del encoder
    torch.cuda.empty_cache()
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
