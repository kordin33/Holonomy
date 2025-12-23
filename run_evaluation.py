"""
run_evaluation.py - Main Evaluation Script for Loop Holonomy Features
======================================================================

This is the primary script to evaluate the V18 model on CIFAKE dataset.

Usage:
    python run_evaluation.py --sample-size 200 --model v18
    
Options:
    --sample-size : Number of images per class (default: 200)
    --model       : Model version ('v18', 'baseline', 'h3', 'h2')
    --device      : 'cuda' or 'cpu' (default: auto-detect)
    --output      : Output directory for results
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.production.holonomy_v18 import HolonomyV18
from deepfake_guard.features.production.h3_dispersion import H3_NormalizedDispersionV2
from deepfake_guard.features.production.h2_scale_law import H2_AreaScaleLaw_Fixed
from deepfake_guard.features.production.baseline import extract_minimal_features


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Loop Holonomy Features")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Number of images per class")
    parser.add_argument("--model", type=str, default="v18",
                        choices=["v18", "baseline", "h3", "h2"],
                        help="Model version to evaluate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--data-dir", type=str, default="./data/cifake",
                        help="Path to CIFAKE dataset")
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--encoder", type=str, default="ViT-L/14",
                        help="CLIP model variant")
    return parser.parse_args()


def load_data(data_dir: Path, n_per_class: int):
    """Load CIFAKE dataset."""
    images, labels = [], []
    
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        cls_dir = data_dir / "test" / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Directory not found: {cls_dir}")
        
        files = list(cls_dir.glob("*.jpg"))[:n_per_class]
        print(f"  Loading {len(files)} {cls} images...")
        
        for p in tqdm(files, desc=cls, leave=False):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    
    return images, np.array(labels)


def extract_features(encoder, images, model_name: str):
    """Extract features based on model type."""
    features = []
    
    if model_name == "v18":
        extractor = HolonomyV18()
        for img in tqdm(images, desc="Extracting V18"):
            try:
                f = extractor.extract_features(encoder, img)
            except Exception:
                f = np.zeros(126, dtype=np.float32)
            features.append(f)
            
    elif model_name == "h3":
        extractor = H3_NormalizedDispersionV2()
        for img in tqdm(images, desc="Extracting H3"):
            try:
                f = extractor.extract_features(encoder, img)
            except Exception:
                f = np.zeros(9, dtype=np.float32)
            features.append(f)
            
    elif model_name == "h2":
        extractor = H2_AreaScaleLaw_Fixed()
        for img in tqdm(images, desc="Extracting H2"):
            try:
                f = extractor.extract_features(encoder, img)
            except Exception:
                f = np.zeros(5, dtype=np.float32)
            features.append(f)
            
    elif model_name == "baseline":
        for img in tqdm(images, desc="Extracting Baseline"):
            try:
                f = extract_minimal_features(encoder, img)['minimal']
            except Exception:
                f = np.zeros(36, dtype=np.float32)
            features.append(f)
    
    return np.array(features, dtype=np.float32)


def train_and_evaluate(features, labels, random_state=42):
    """Train SVM and evaluate with cross-validation."""
    features = np.nan_to_num(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=0.3, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Pipeline with grid search
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=random_state))
    ])
    
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.01, 0.001]
    }
    
    grid = GridSearchCV(
        pipe, param_grid, 
        cv=cv, 
        scoring='roc_auc', 
        n_jobs=-1,
        verbose=0
    )
    
    print("  Training SVM with GridSearchCV...")
    grid.fit(X_train, y_train)
    
    # Evaluate
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = grid.best_estimator_.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    
    results = {
        'auc_roc': float(auc),
        'best_params': grid.best_params_,
        'cv_score': float(grid.best_score_),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return results


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔬 LOOP HOLONOMY FEATURE EVALUATION")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Sample Size: {args.sample_size} per class")
    print(f"  Encoder:     CLIP {args.encoder}")
    
    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:      {device}")
    
    # Load data
    print("\n📂 Loading data...")
    data_dir = Path(args.data_dir)
    images, labels = load_data(data_dir, args.sample_size)
    print(f"  Total images: {len(images)}")
    
    # Initialize encoder
    print("\n🧠 Loading encoder...")
    encoder = get_encoder("clip", args.encoder, device)
    
    # Extract features
    print("\n📊 Extracting features...")
    features = extract_features(encoder, images, args.model)
    print(f"  Feature shape: {features.shape}")
    
    # Train and evaluate
    print("\n🎯 Training and evaluating...")
    results = train_and_evaluate(features, labels)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  AUC-ROC:         {results['auc_roc']:.4f}")
    print(f"  CV Score:        {results['cv_score']:.4f}")
    print(f"  Best Params:     {results['best_params']}")
    
    cr = results['classification_report']
    print(f"\n  Precision (FAKE): {cr['0']['precision']:.4f}")
    print(f"  Recall (FAKE):    {cr['0']['recall']:.4f}")
    print(f"  F1 (FAKE):        {cr['0']['f1-score']:.4f}")
    
    print(f"\n  Precision (REAL): {cr['1']['precision']:.4f}")
    print(f"  Recall (REAL):    {cr['1']['recall']:.4f}")
    print(f"  F1 (REAL):        {cr['1']['f1-score']:.4f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"eval_{args.model}_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'model': args.model,
        'sample_size': args.sample_size,
        'encoder': args.encoder,
        'feature_dim': int(features.shape[1]),
        'results': results
    }
    
    with open(result_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
