"""
test_h2h3_v7.py - Test V7: 2-Head (CLEAN vs MAX) + EXP (Richardson)

Testujemy:
1. H2_V7 CLEAN/MAX standalone
2. H2_V7_EXP CLEAN/MAX standalone
3. Ablacje: Base + H2_CLEAN, Base + H2_EXP_CLEAN
4. H3 analogicznie
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
from deepfake_guard.features.h2_v7 import H2_V7
from deepfake_guard.features.h2_v7_exp import H2_V7_EXP
from deepfake_guard.features.h3_v7 import H3_V7
from deepfake_guard.features.h3_v7_exp import H3_V7_EXP
from deepfake_guard.features.optimized_v3 import BaselineV3

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
    print("🔬 TEST: V7 2-Head (CLEAN/MAX) + EXP (Richardson)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    h2 = H2_V7()
    h2_exp = H2_V7_EXP()
    h3 = H3_V7()
    h3_exp = H3_V7_EXP()
    baseline = BaselineV3()
    
    print("\n📊 Extracting features...")
    feats = {
        'base': [], 
        'h2_clean': [], 'h2_max': [],
        'h2e_clean': [], 'h2e_max': [],
        'h3_clean': [], 'h3_max': [],
        'h3e_clean': [], 'h3e_max': []
    }
    
    for img in tqdm(images, desc="Extracting"):
        try:
            feats['base'].append(baseline.extract_features(encoder, img))
            feats['h2_clean'].append(h2.extract_clean(encoder, img))
            feats['h2_max'].append(h2.extract_max(encoder, img))
            feats['h2e_clean'].append(h2_exp.extract_clean(encoder, img))
            feats['h2e_max'].append(h2_exp.extract_max(encoder, img))
            feats['h3_clean'].append(h3.extract_clean(encoder, img))
            feats['h3_max'].append(h3.extract_max(encoder, img))
            feats['h3e_clean'].append(h3_exp.extract_clean(encoder, img))
            feats['h3e_max'].append(h3_exp.extract_max(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            feats['base'].append(np.zeros(63))
            feats['h2_clean'].append(np.zeros(12))
            feats['h2_max'].append(np.zeros(20))
            feats['h2e_clean'].append(np.zeros(16))
            feats['h2e_max'].append(np.zeros(24))
            feats['h3_clean'].append(np.zeros(14))
            feats['h3_max'].append(np.zeros(20))
            feats['h3e_clean'].append(np.zeros(16))
            feats['h3e_max'].append(np.zeros(24))
    
    for k in feats:
        feats[k] = np.array(feats[k], dtype=np.float32)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Reference
    auc_base = test_auc(feats['base'], labels, "Base")
    print(f"\n  Baseline_V3: {auc_base:.4f} [63D]")
    
    # H2 Standalone MAX
    print("\n📊 H2 STANDALONE (MAX):")
    auc = test_auc(feats['h2_max'], labels, "H2_MAX")
    print(f"  H2_V7 MAX: {auc:.4f} [20D]")
    auc = test_auc(feats['h2e_max'], labels, "H2E_MAX")
    print(f"  H2_V7_EXP MAX: {auc:.4f} [24D]")
    
    # H2 Ablacje CLEAN
    print("\n📊 H2 ABLACJE (Base + CLEAN):")
    combined = np.concatenate([feats['base'], feats['h2_clean']], axis=1)
    auc = test_auc(combined, labels, "Base+H2C")
    delta = auc - auc_base
    print(f"  Base + H2_V7_CLEAN: {auc:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    combined = np.concatenate([feats['base'], feats['h2e_clean']], axis=1)
    auc = test_auc(combined, labels, "Base+H2EC")
    delta = auc - auc_base
    print(f"  Base + H2_V7_EXP_CLEAN: {auc:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    # H3 Standalone MAX
    print("\n📊 H3 STANDALONE (MAX):")
    auc = test_auc(feats['h3_max'], labels, "H3_MAX")
    print(f"  H3_V7 MAX: {auc:.4f} [20D]")
    auc = test_auc(feats['h3e_max'], labels, "H3E_MAX")
    print(f"  H3_V7_EXP MAX: {auc:.4f} [24D]")
    
    # H3 Ablacje CLEAN
    print("\n📊 H3 ABLACJE (Base + CLEAN):")
    combined = np.concatenate([feats['base'], feats['h3_clean']], axis=1)
    auc = test_auc(combined, labels, "Base+H3C")
    delta = auc - auc_base
    print(f"  Base + H3_V7_CLEAN: {auc:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    combined = np.concatenate([feats['base'], feats['h3e_clean']], axis=1)
    auc = test_auc(combined, labels, "Base+H3EC")
    delta = auc - auc_base
    print(f"  Base + H3_V7_EXP_CLEAN: {auc:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    # Best combination
    print("\n📊 BEST COMBO:")
    combined = np.concatenate([feats['base'], feats['h2e_clean'], feats['h3e_clean']], axis=1)
    auc = test_auc(combined, labels, "Base+H2EC+H3EC")
    delta = auc - auc_base
    print(f"  Base + H2_EXP_CLEAN + H3_EXP_CLEAN: {auc:.4f} ({'✅' if delta > 0 else '❌'} {delta:+.4f})")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
