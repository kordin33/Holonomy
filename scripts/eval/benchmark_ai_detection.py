"""
benchmark_ai_detection.py - Benchmark na nowym datasecie AI Detection 2025

Dataset: AI vs Deepfake vs Real
- Real: prawdziwe zdjęcia
- Deepfake: manipulowane twarze  
- Artificial: AI-generated (DALL-E, Midjourney, etc.)

Testujemy czy nasz model generalizuje na NOWY, TRUDNIEJSZY dataset!
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
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

DATA_DIR = Path("./data/ai_detection_2025")
OUTPUT_DIR = Path("./results/ai_detection_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 1000  # Use all available
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset():
    """Load the new AI detection dataset."""
    images, labels, class_names = [], [], []
    
    # Map folder names to labels
    class_map = {
        "Real": 0,
        "Deepfake": 1,
        "None": 2,  # Artificial / AI-generated
        "Artificial": 2,
    }
    
    label_names = {0: "Real", 1: "Deepfake", 2: "AI-Generated"}
    
    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            label = class_map.get(folder.name, -1)
            if label == -1:
                continue
            
            files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            np.random.shuffle(files)
            files = files[:MAX_PER_CLASS]
            
            print(f"Loading {folder.name} ({label_names[label]}): {len(files)}")
            for p in tqdm(files, desc=folder.name):
                try:
                    img = Image.open(p).convert("RGB").resize((224, 224))
                    images.append(img)
                    labels.append(label)
                    class_names.append(label_names[label])
                except:
                    continue
    
    return images, np.array(labels), np.array(class_names), label_names


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print("="*70)
    print("🔬 BENCHMARK: AI Detection 2025 Dataset")
    print("   Real vs Deepfake vs AI-Generated")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    images, labels, class_names, label_names = load_dataset()
    print(f"\nTotal: {len(images)} images")
    print(f"Classes: {label_names}")
    
    # Check distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {label_names[u]}: {c}")
    
    # Extract embeddings
    print("\n" + "="*50)
    print("EXTRACTING EMBEDDINGS (ViT-L/14)")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    embeddings = encoder.encode_batch(images, batch_size=32, show_progress=True)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Split
    train_idx, test_idx = train_test_split(
        np.arange(len(images)), test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    class_names_test = class_names[test_idx]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train classifiers
    print("\n" + "="*50)
    print("TRAINING CLASSIFIERS")
    print("="*50)
    
    classifiers = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=RANDOM_STATE),
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"  Overall Accuracy: {acc:.2%}")
        
        # Per-class accuracy
        for label_id, label_name in label_names.items():
            mask = y_test == label_id
            if mask.sum() > 0:
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {label_name}: {class_acc:.2%}")
        
        results[clf_name] = {
            "accuracy": acc,
            "y_pred": y_pred,
            "report": classification_report(y_test, y_pred, target_names=list(label_names.values())),
            "confusion": confusion_matrix(y_test, y_pred),
        }
        
        print(f"\n{results[clf_name]['report']}")
    
    # Visualize confusion matrix for best model
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    best_clf = max(results.keys(), key=lambda x: results[x]["accuracy"])
    cm = results[best_clf]["confusion"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_names.values()),
                yticklabels=list(label_names.values()))
    plt.title(f'Confusion Matrix: {best_clf}\nAccuracy: {results[best_clf]["accuracy"]:.2%}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    print(f"✓ Saved: confusion_matrix.png")
    plt.close()
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print(f"🏆 BEST: {best_clf} = {results[best_clf]['accuracy']:.2%}")
    print("="*70)


if __name__ == "__main__":
    main()
