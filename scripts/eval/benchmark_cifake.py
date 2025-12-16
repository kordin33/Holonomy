"""
benchmark_cifake.py - Benchmark na CIFAKE (Real vs Stable Diffusion)

Dataset: CIFAKE - 120,000 images
- 60k Real (CIFAR-10)
- 60k Fake (Stable Diffusion v1.4)

Trudny test: małe obrazy 32x32, podobna zawartość w obu klasach!
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/cifake_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use subset for faster testing (full dataset = 50000)
MAX_TRAIN = 5000  # 5k per class
MAX_TEST = 2000   # 2k per class
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_split(split_dir, max_per_class):
    """Load images from a split (train/test)."""
    images, labels = [], []
    
    for class_name, label in [("REAL", 1), ("FAKE", 0)]:
        class_dir = split_dir / class_name
        files = list(class_dir.glob("*.jpg"))
        np.random.shuffle(files)
        files = files[:max_per_class]
        
        print(f"  Loading {class_name}: {len(files)}")
        for p in tqdm(files, desc=class_name):
            try:
                # Upscale from 32x32 to 224x224 for CLIP
                img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
                images.append(img)
                labels.append(label)
            except:
                continue
    
    return images, np.array(labels)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 CIFAKE BENCHMARK: Real (CIFAR-10) vs Fake (Stable Diffusion)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load train data
    print("\n" + "="*50)
    print("LOADING TRAIN DATA")
    print("="*50)
    
    train_images, y_train = load_split(DATA_DIR / "train", MAX_TRAIN)
    print(f"Train total: {len(train_images)}")
    
    # Load test data
    print("\n" + "="*50)
    print("LOADING TEST DATA")
    print("="*50)
    
    test_images, y_test = load_split(DATA_DIR / "test", MAX_TEST)
    print(f"Test total: {len(test_images)}")
    
    # Extract embeddings
    print("\n" + "="*50)
    print("EXTRACTING EMBEDDINGS (ViT-L/14)")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    print("Encoding train...")
    X_train = encoder.encode_batch(train_images, batch_size=64, show_progress=True)
    
    print("Encoding test...")
    X_test = encoder.encode_batch(test_images, batch_size=64, show_progress=True)
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Train classifiers
    print("\n" + "="*50)
    print("TRAINING & EVALUATING CLASSIFIERS")
    print("="*50)
    
    classifiers = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=RANDOM_STATE),
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name}:")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        # Per-class
        real_mask = y_test == 1
        fake_mask = y_test == 0
        real_acc = accuracy_score(y_test[real_mask], y_pred[real_mask])
        fake_acc = accuracy_score(y_test[fake_mask], y_pred[fake_mask])
        
        print(f"  Overall: {acc:.2%}")
        print(f"  Real (CIFAR-10): {real_acc:.2%}")
        print(f"  Fake (Stable Diffusion): {fake_acc:.2%}")
        
        results[clf_name] = {
            "accuracy": acc,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "y_pred": y_pred,
            "confusion": confusion_matrix(y_test, y_pred),
        }
    
    # Best model
    best_clf = max(results.keys(), key=lambda x: results[x]["accuracy"])
    best_acc = results[best_clf]["accuracy"]
    
    # Confusion matrix
    cm = results[best_clf]["confusion"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (SD)', 'Real (CIFAR)'],
                yticklabels=['Fake (SD)', 'Real (CIFAR)'])
    plt.title(f'CIFAKE: {best_clf}\nAccuracy: {best_acc:.2%}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    print(f"\n✓ Saved: confusion_matrix.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("📊 CIFAKE RESULTS SUMMARY")
    print("="*70)
    
    for name, res in results.items():
        print(f"{name}: {res['accuracy']:.2%} (Real: {res['real_acc']:.2%}, Fake: {res['fake_acc']:.2%})")
    
    print("\n" + "="*70)
    print(f"🏆 BEST: {best_clf} = {best_acc:.2%}")
    print("="*70)
    
    # Interpretation
    if best_acc > 0.95:
        print("✅ Model dobrze generalizuje na Stable Diffusion!")
    elif best_acc > 0.80:
        print("⚠️ Przyzwoite wyniki, ale jest miejsce na poprawę.")
    else:
        print("❌ Słabe wyniki - model nie radzi sobie z tym typem fake.")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
