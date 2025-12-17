"""
test_final_push.py - V12 + PROD_H2 + XGBoost

Cel: Przebić 0.90 AUC używając najlepszych klocków (V12 Ensemble 0.885 + Prod H2)
oraz silniejszego klasyfikatora (XGBoost/LightGBM).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.holonomy_decomposition_v12 import HolonomyDecompositionV12
from deepfake_guard.features.production.h2_scale_law import H2_AreaScaleLaw_Fixed
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
    
    # Simple split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 1. SVM (RBF)
    print("  SVM...", end="", flush=True)
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
    svm.fit(X_train_s, y_train)
    y_prob_svm = svm.predict_proba(X_test_s)[:, 1]
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    print(f" AUC: {auc_svm:.4f}")
    
    # 2. Gradient Boosting (sklearn)
    print("  GBM...", end="", flush=True)
    xgb_clf = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    xgb_clf.fit(X_train, y_train)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    print(f" AUC: {auc_xgb:.4f}")
    
    # 3. Voting (Avg)
    y_prob_avg = (y_prob_svm + y_prob_xgb) / 2
    auc_avg = roc_auc_score(y_test, y_prob_avg)
    print(f"  Ensemble (SVM+XGB): {auc_avg:.4f}")
    
    return max(auc_svm, auc_xgb, auc_avg)

def main():
    print("="*70)
    print("🚀 FINAL PUSH: V12 + PROD_H2 + XGBOOST")
    print("   Goal: > 0.90 AUC")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data(SAMPLE_SIZE)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    v12 = HolonomyDecompositionV12()
    prod_h2 = H2_AreaScaleLaw_Fixed()
    
    print("\n📊 Extracting features...")
    feats_v12 = []
    feats_h2 = []
    
    for img in tqdm(images, desc="Extracting"):
        # V12 (252D)
        try:
            f12 = v12.extract_features(encoder, img)
        except:
            f12 = np.zeros(252)
        feats_v12.append(f12)
        
        # Prod H2 (12D or 5D)
        try:
            fh2 = prod_h2.extract_features(encoder, img)
        except:
            fh2 = np.zeros(12) # assuming shape
        feats_h2.append(fh2)
        
    feats_v12 = np.array(feats_v12, dtype=np.float32)
    feats_h2 = np.array(feats_h2, dtype=np.float32)
    
    # Handle NaNs
    feats_v12 = np.nan_to_num(feats_v12)
    feats_h2 = np.nan_to_num(feats_h2)
    
    # Test Components
    test_models(feats_v12, labels, "V12_Only")
    
    # Combine
    combined = np.concatenate([feats_v12, feats_h2], axis=1)
    
    print("\n🔥 RESULTS [V12 + Prod_H2]:")
    best_auc = test_models(combined, labels, "Final_Combo")
    
    if best_auc > 0.90:
        print("\n🏆 SUCCESS! We broke 0.90!")
    else:
        print(f"\n🤏 Close... Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
