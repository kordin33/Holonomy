"""
test_decomp_v16.py - Test V16 (Orthogonalized + Per-Loop)
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
from deepfake_guard.features.holonomy_decomposition_v16 import HolonomyDecompositionV16

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

def test_auc(features, labels, name):
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
    print("🔬 TEST: V16 - ORTHOGONALIZED + PER-LOOP DISAGREEMENT")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v16 = HolonomyDecompositionV16()
    
    print("\n📊 Extracting features...")
    features = []
    for img in tqdm(images, desc="V16"):
        try:
            features.append(v16.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            features.append(np.zeros(96))
    
    features = np.array(features, dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Full
    auc_full = test_auc(features, labels, "V16_Full")
    print(f"\n  V16_Full: {auc_full:.4f} [{features.shape[1]}D]")
    
    # Components
    auc_global = test_auc(features[:, :63], labels, "Global")
    print(f"  Global (63D): {auc_global:.4f}")
    
    auc_disagree = test_auc(features[:, 63:81], labels, "PerLoopDisagree")
    print(f"  Per-Loop Disagreement (18D): {auc_disagree:.4f}")
    
    auc_comm = test_auc(features[:, 81:87], labels, "Commutator")
    print(f"  Commutator Slim (6D): {auc_comm:.4f}")
    
    auc_interact = test_auc(features[:, 87:], labels, "Interaction")
    print(f"  Interaction (9D): {auc_interact:.4f}")
    
    # Ablations
    print("\n📊 ABLACJE:")
    auc_g_d = test_auc(features[:, :81], labels, "Global+Disagree")
    delta = auc_g_d - auc_global
    print(f"  Global + PerLoop: {auc_g_d:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    auc_g_c = test_auc(np.concatenate([features[:, :63], features[:, 81:87]], axis=1), labels, "Global+Comm")
    delta = auc_g_c - auc_global
    print(f"  Global + Comm: {auc_g_c:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    # Compare to V12/V15
    print("\n📊 REFERENCE (from previous tests):")
    print("  V12 Ensemble: 0.8850")
    print("  V15 Full: 0.8764")
    print("  Base + H2_V6_EXP: 0.8885")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
