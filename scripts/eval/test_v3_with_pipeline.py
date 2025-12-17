"""
test_v3_with_pipeline.py - Test poprawionych cech z Pipeline

NAPRAWY:
1. Shape features (curvature, tortuosity, std_step) zamiast scale
2. Pipeline(StandardScaler -> SVM)
3. Grid search dla C/gamma
4. H_res (residual) zamiast H_norm
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.degradation_commutator_v3_fixed import (
    extract_batch_minimal_features,
    LOOPS
)

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/v3_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:n_per_class]
        for p in tqdm(files, desc=f"Loading {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    return images, np.array(labels)


def compute_residual_features(H_raw_all, path_lengths, train_mask):
    """
    Oblicza H_res = H_raw - (β*L + c) per pętla.
    
    Fit na train, apply na wszystkich.
    """
    n_samples, n_loops = H_raw_all.shape
    H_res = np.zeros_like(H_raw_all)
    
    for loop_idx in range(n_loops):
        H = H_raw_all[:, loop_idx]
        L = path_lengths[:, loop_idx]
        
        # Fit na train
        reg = LinearRegression()
        reg.fit(L[train_mask].reshape(-1, 1), H[train_mask])
        
        # Predict i residual
        H_pred = reg.predict(L.reshape(-1, 1))
        H_res[:, loop_idx] = H - H_pred
    
    return H_res


def test_with_pipeline(features, labels, name):
    """Test z Pipeline(StandardScaler -> SVM) + GridSearch."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Pipeline ze StandardScaler!
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])
    
    # Quick grid search
    param_grid = {
        'svm__C': [0.3, 1, 3, 10],
        'svm__gamma': ['scale', 0.01, 0.1]
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Best model
    best = grid.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, best.predict(X_test))
    
    return {
        'name': name,
        'auc': auc,
        'acc': acc,
        'best_params': grid.best_params_,
        'shape': features.shape
    }


def main():
    print("="*70)
    print("🔬 TEST V3 - POPRAWIONE CECHY")
    print("   Shape features + Pipeline + StandardScaler")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data(SAMPLE_SIZE)
    
    # Encoder
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    features = extract_batch_minimal_features(encoder, images, show_progress=True)
    
    for key, arr in features.items():
        print(f"  {key}: {arr.shape}")
    
    # Prepare train mask for H_res
    _, _, train_idx, _ = train_test_split(
        range(len(labels)), labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    train_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train_idx] = True
    
    # Extract path_lengths for H_res
    print("\nExtracting path_lengths for H_res...")
    path_lengths = []
    for img in tqdm(images, desc="Path lengths"):
        from deepfake_guard.features.degradation_commutator_v3_fixed import (
            compute_holonomy_with_shape, LOOPS
        )
        pls = []
        for loop in LOOPS:
            stats = compute_holonomy_with_shape(encoder, img, loop)
            pls.append(stats['path_length'])
        path_lengths.append(pls)
    
    path_lengths = np.array(path_lengths, dtype=np.float32)
    
    # Compute H_res
    print("\nComputing H_res (residual)...")
    H_res = compute_residual_features(features['H_raw'], path_lengths, train_mask)
    features['H_res'] = H_res
    
    # Create combined features
    # Minimal + H_res
    features['minimal_with_Hres'] = np.concatenate([
        features['minimal'],
        features['H_res']
    ], axis=1)
    
    # Test each
    print("\n" + "="*70)
    print("TESTING WITH PIPELINE + GRIDSEARCH")
    print("="*70)
    
    results = []
    
    for feat_name, display_name in [
        ('H_raw', 'H_raw only (baseline)'),
        ('shape', 'Shape only (tort+curv+std)'),
        ('minimal', 'Minimal (H_raw + shape)'),
        ('H_res', 'H_res (residual)'),
        ('minimal_with_Hres', 'Minimal + H_res'),
    ]:
        print(f"\n🔬 Testing: {display_name}")
        res = test_with_pipeline(features[feat_name], labels, display_name)
        results.append(res)
        
        symbol = "✅" if res['auc'] > 0.74 else "⚠️" if res['auc'] > 0.65 else "❌"
        print(f"  {symbol} AUC: {res['auc']:.4f}, Acc: {res['acc']:.4f}")
        print(f"     Best params: {res['best_params']}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<30} {'AUC':<8} {'Acc':<8} {'Shape':<12}")
    print("-"*70)
    
    baseline_auc = results[0]['auc']
    
    for res in results:
        delta = res['auc'] - baseline_auc
        symbol = "✅" if delta >= 0 else "❌"
        
        print(f"{res['name']:<30} {res['auc']:.4f}  {res['acc']:.4f}  "
              f"{str(res['shape']):<12} {symbol} {delta:+.4f}")
    
    # Best
    best = max(results, key=lambda x: x['auc'])
    print(f"\n🏆 BEST: {best['name']} (AUC={best['auc']:.4f})")
    
    # Save
    np.savez_compressed(
        OUTPUT_DIR / "v3_results.npz",
        labels=labels,
        **features
    )
    
    import json
    with open(OUTPUT_DIR / "v3_summary.json", 'w') as f:
        results_json = []
        for r in results:
            r_copy = r.copy()
            r_copy['shape'] = list(r['shape'])
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_DIR}/")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
