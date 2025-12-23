"""
test_decomp_v15.py - Test V15 (V12 Core + Commutator Curvature)
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
from deepfake_guard.features.holonomy_decomposition_v15 import HolonomyDecompositionV15
from deepfake_guard.features.holonomy_decomposition_v15 import compute_commutator_features
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
    print("🔬 TEST: DECOMPOSITION V15 - V12 CORE + COMMUTATOR CURVATURE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v15 = HolonomyDecompositionV15()
    
    print("\n📊 Extracting V15 features...")
    feats_v15 = []
    feats_comm = []
    
    for img in tqdm(images, desc="V15"):
        try:
            feats_v15.append(v15.extract_features(encoder, img))
            feats_comm.append(compute_commutator_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            feats_v15.append(np.zeros(276))
            feats_comm.append(np.zeros(24))
    
    feats_v15 = np.array(feats_v15, dtype=np.float32)
    feats_comm = np.array(feats_comm, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Full V15
    auc_full = test_stable(feats_v15, labels, "V15_Full")
    print(f"\n  V15_Full: {auc_full:.4f} [{feats_v15.shape[1]}D]")
    
    # Components
    auc_global = test_stable(feats_v15[:, :63], labels, "Global_Base")
    print(f"  Global_Base (63D): {auc_global:.4f}")
    
    auc_comm = test_stable(feats_comm, labels, "Commutator_Only")
    print(f"  Commutator_Only (24D): {auc_comm:.4f}")
    
    auc_disagree = test_stable(feats_v15[:, 150:213], labels, "Disagreement")
    print(f"  Disagreement_Robust (63D): {auc_disagree:.4f}")
    
    # Ablations
    print("\n📊 ABLACJE:")
    auc_gbase_comm = test_stable(np.concatenate([feats_v15[:, :63], feats_comm], axis=1), labels, "Global+Comm")
    delta = auc_gbase_comm - auc_global
    print(f"  Global + Commutator: {auc_gbase_comm:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    # Correlations
    print("\n📊 COMMUTATOR CORRELATIONS:")
    comm_names = ['d_comm', 'dens', 'ratio', 'angle'] * 6
    pair_names = ['blur03_jpg90', 'blur05_scl09', 'jpg80_scl09', 'shrp_jpg90', 'blur03_scl75', 'jpg70_blur05']
    for i in range(min(24, feats_comm.shape[1])):
        corr = np.corrcoef(feats_comm[:, i], labels)[0, 1]
        pair_idx = i // 4
        feat_idx = i % 4
        name = f"{pair_names[pair_idx]}_{comm_names[feat_idx]}"
        if abs(corr) > 0.1:
            print(f"  {name:<25}: {corr:+.4f}")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
