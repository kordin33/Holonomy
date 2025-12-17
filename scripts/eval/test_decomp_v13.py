"""
test_decomp_v13.py - Test V13 Smart Anomaly Engine
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.holonomy_decomposition_v13 import HolonomyDecompositionV13
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

def test_models(X, y, name):
    print(f"\n--- Testing {name} [{X.shape[1]}D] ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 1. SVM
    print("  SVM...", end="", flush=True)
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
    svm.fit(X_train_s, y_train)
    y_prob_svm = svm.predict_proba(X_test_s)[:, 1]
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    print(f" AUC: {auc_svm:.4f}")
    
    # 2. GBM
    print("  GBM...", end="", flush=True)
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, 
        subsample=0.8, random_state=RANDOM_STATE
    )
    gbm.fit(X_train, y_train)
    y_prob_gbm = gbm.predict_proba(X_test)[:, 1]
    auc_gbm = roc_auc_score(y_test, y_prob_gbm)
    print(f" AUC: {auc_gbm:.4f}")
    
    # 3. Ensemble
    y_prob_avg = (y_prob_svm + y_prob_gbm) / 2
    auc_avg = roc_auc_score(y_test, y_prob_avg)
    print(f"  Ensemble (SVM+GBM): {auc_avg:.4f}")
    
    return auc_avg

def main():
    print("="*70)
    print("🔬 TEST: DECOMPOSITION V13 - SMART ANOMALY ENGINE")
    print("   Smart Patches + Outlier Scores + Enhanced Baseline (81D)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v13 = HolonomyDecompositionV13()
    
    print("\n📊 Extracting V13 features...")
    features = []
    
    for img in tqdm(images, desc="V13"):
        try:
            features.append(v13.extract_features(encoder, img))
        except Exception as e:
            print(f"Error: {e}")
            features.append(np.zeros(327))
            
    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features)
    
    # Test Full
    test_models(features, labels, "V13_Full")
    
    # Test Components
    # Global (first 81)
    test_models(features[:, :81], labels, "V13_Global_Only")
    
    # Outliers (last 3)
    test_models(features[:, -3:], labels, "V13_Outlier_Scalars")
    
    # Abs Gap (243:324)
    test_models(features[:, 243:324], labels, "V13_AbsGap")

    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
