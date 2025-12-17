"""test_trajectory.py - Test TRAJECTORY FEATURES"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))
from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.degradation_commutator_optimized import extract_all_optimized_features_v2

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/optimization_tests")
SAMPLE_SIZE = 200
RANDOM_STATE = 42

def load_data(n):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:n]
        for p in tqdm(files, desc=f"Loading {cls}", leave=False):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    return images, np.array(labels)

print("🔬 TEST: Trajectory Features")
device = "cuda" if torch.cuda.is_available() else "cpu"
images, labels = load_data(SAMPLE_SIZE)
encoder = get_encoder("clip", "ViT-L/14", device)

features_list = []
for img in tqdm(images, desc="Extracting"):
    feats = extract_all_optimized_features_v2(encoder, img)
    features_list.append(feats['trajectory_features'])

features = np.array(features_list)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels)
svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE, probability=True)
svm.fit(X_train, y_train)
auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

print(f"✅ Trajectory AUC: {auc:.4f}")
np.savez(OUTPUT_DIR / "trajectory_result.npz", auc=auc, features=features, labels=labels)
