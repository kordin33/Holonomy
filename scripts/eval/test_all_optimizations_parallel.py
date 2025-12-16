"""
test_all_optimizations_parallel.py - RÓWNOLEGŁE testy wszystkich optymalizacji

Testuje:
1. Baseline (oryginalna holonomia)
2. Normalized holonomy
3. Patchwise holonomy  
4. Relative commutator
5. Combined (wszystko razem)

RÓWNOLEGLE na GPU + wizualizacje każdej!
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pacmap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.degradation_commutator_optimized import (
    extract_all_optimized_features_v2
)


# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/optimization_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200  # Szybki test
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:n_per_class]
        for p in tqdm(files, desc=f"Loading {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    return images, np.array(labels)


# ============================================================================
# EXTRACTION
# ============================================================================

def extract_all_features(encoder, images):
    """Ekstraktuje WSZYSTKIE wersje cech dla wszystkich obrazów."""
    print("\n" + "="*70)
    print("EXTRACTING ALL OPTIMIZED FEATURES")
    print("="*70)
    
    all_features = {
        'baseline_holonomy': [],
        'trajectory_features': [],
        'patchwise_holonomy': [],
        'commutator': [],
        'combined': []
    }
    
    for img in tqdm(images, desc="Extracting"):
        feats = extract_all_optimized_features_v2(encoder, img)
        
        for key in all_features.keys():
            all_features[key].append(feats[key])
    
    # Convert to arrays
    for key in all_features.keys():
        all_features[key] = np.array(all_features[key], dtype=np.float32)
        print(f"  {key}: {all_features[key].shape}")
    
    return all_features


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_features(features, labels, name):
    """Ocenia pojedynczą wersję cech."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {name}")
    print(f"{'='*70}")
    print(f"Feature shape: {features.shape}")
    
    # Statistical test
    real_mask = labels == 1
    fake_mask = labels == 0
    
    real_feat = features[real_mask]
    fake_feat = features[fake_mask]
    
    mean_diff = real_feat.mean() - fake_feat.mean()
    pooled_std = np.sqrt((real_feat.std()**2 + fake_feat.std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    print(f"\nReal mean: {real_feat.mean():.6f}")
    print(f"Fake mean: {fake_feat.mean():.6f}")
    print(f"Difference: {mean_diff:+.6f} ({(mean_diff/fake_feat.mean())*100:+.2f}%)")
    print(f"Cohen's d: {cohens_d:.4f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Logistic Regression
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    
    # SVM
    svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE, probability=True)
    svm.fit(X_train, y_train)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    
    # Silhouette
    sil = silhouette_score(features, labels)
    
    print(f"\nLogistic Regression:")
    print(f"  AUC: {auc_lr:.4f}")
    print(f"  Acc: {acc_lr:.4f}")
    
    print(f"\nSVM (RBF):")
    print(f"  AUC: {auc_svm:.4f} {'✅' if auc_svm > 0.78 else '⚠️' if auc_svm > 0.70 else '❌'}")
    print(f"  Acc: {acc_svm:.4f}")
    
    print(f"\nSilhouette: {sil:.4f}")
    
    return {
        'name': name,
        'shape': features.shape,
        'mean_diff': mean_diff,
        'cohens_d': cohens_d,
        'lr_auc': auc_lr,
        'svm_auc': auc_svm,
        'svm_acc': acc_svm,
        'silhouette': sil
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_pacmap(features, labels, name):
    """PaCMAP visualization."""
    print(f"\nGenerating PaCMAP for {name}...")
    
    # Sample if too many
    if len(features) > 400:
        indices = np.random.choice(len(features), 400, replace=False)
        features_sub = features[indices]
        labels_sub = labels[indices]
    else:
        features_sub = features
        labels_sub = labels
    
    # PaCMAP
    pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj = pacmap_model.fit_transform(features_sub)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {0: "#e74c3c", 1: "#2ecc71"}
    labels_map = {0: "Fake", 1: "Real"}
    
    for label in [0, 1]:
        mask = labels_sub == label
        ax.scatter(proj[mask, 0], proj[mask, 1],
                  c=colors[label], label=labels_map[label],
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    # Silhouette
    sil = silhouette_score(features_sub, labels_sub)
    
    ax.set_title(f"PaCMAP: {name}\n{features.shape[1]}D, Silhouette={sil:.4f}", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PaCMAP Dim 1")
    ax.set_ylabel("PaCMAP Dim 2")
    
    plt.tight_layout()
    
    safe_name = name.lower().replace(' ', '_')
    plt.savefig(OUTPUT_DIR / f"pacmap_{safe_name}.png", dpi=200)
    print(f"✓ Saved: pacmap_{safe_name}.png")
    plt.close()


# ============================================================================
# COMPARISON
# ============================================================================

def compare_all_results(results_list):
    """Porównuje wszystkie wyniki."""
    print("\n" + "="*70)
    print("📊 COMPARISON - ALL OPTIMIZATIONS")
    print("="*70)
    
    # Table
    print(f"\n{'Method':<25} {'AUC':<8} {'Acc':<8} {'Sil':<8} {'Cohen':<8}")
    print("-"*70)
    
    baseline_auc = None
    for res in results_list:
        if res['name'] == 'Baseline':
            baseline_auc = res['svm_auc']
        
        print(f"{res['name']:<25} {res['svm_auc']:.4f}  {res['svm_acc']:.4f}  "
              f"{res['silhouette']:.4f}  {res['cohens_d']:.4f}")
    
    # Improvements
    if baseline_auc:
        print("\n" + "="*70)
        print("📈 IMPROVEMENTS OVER BASELINE")
        print("="*70)
        
        for res in results_list:
            if res['name'] != 'Baseline':
                delta = res['svm_auc'] - baseline_auc
                symbol = "✅" if delta > 0.02 else "⚠️" if delta > 0 else "❌"
                print(f"{symbol} {res['name']:<25} {delta:+.4f} ({(delta/baseline_auc)*100:+.2f}%)")
    
    # Bar plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    names = [r['name'] for r in results_list]
    aucs = [r['svm_auc'] for r in results_list]
    cohens = [r['cohens_d'] for r in results_list]
    
    # AUC
    axes[0].barh(names, aucs, color=['#3498db' if n == 'Baseline' else '#2ecc71' for n in names])
    axes[0].set_xlabel('SVM ROC-AUC', fontsize=12)
    axes[0].set_title('Predictive Power (AUC)', fontsize=14, fontweight='bold')
    axes[0].axvline(0.78, color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Cohen's d
    axes[1].barh(names, cohens, color=['#3498db' if n == 'Baseline' else '#e67e22' for n in names])
    axes[1].set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    axes[1].set_title('Statistical Separability', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_all.png", dpi=200)
    print(f"\n✓ Saved: comparison_all.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚀 PARALLEL OPTIMIZATION TESTING")
    print("   Testing ALL optimizations simultaneously!")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data(SAMPLE_SIZE)
    
    # Encoder
    print("\nInitializing CLIP ViT-L/14...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract ALL features
    all_features = extract_all_features(encoder, images)
    
    # Evaluate each
    results = []
    
    for feat_name in ['baseline_holonomy', 'trajectory_features', 'patchwise_holonomy', 
                      'commutator', 'combined']:
        
        display_name = {
            'baseline_holonomy': 'Baseline (H_raw)',
            'trajectory_features': 'Trajectory Features',
            'patchwise_holonomy': 'Patchwise (H_raw)',
            'commutator': 'Commutator (deterministic)',
            'combined': 'Combined (All)'
        }[feat_name]
        
        res = evaluate_features(all_features[feat_name], labels, display_name)
        results.append(res)
        
        # Visualize
        visualize_pacmap(all_features[feat_name], labels, display_name)
    
    # Compare
    compare_all_results(results)
    
    # Save results
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    np.savez_compressed(
        OUTPUT_DIR / "all_features.npz",
        labels=labels,
        **all_features
    )
    
    import json
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        # Convert numpy types to Python types
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            for k, v in r_copy.items():
                if isinstance(v, (np.integer, np.floating)):
                    r_copy[k] = float(v)
                elif isinstance(v, tuple):
                    r_copy[k] = list(v)
            results_serializable.append(r_copy)
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"✓ Saved to {OUTPUT_DIR}/")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    
    # Summary
    best = max(results, key=lambda x: x['svm_auc'])
    print(f"\n🏆 BEST OPTIMIZATION: {best['name']}")
    print(f"   AUC: {best['svm_auc']:.4f}")
    print(f"   Cohen's d: {best['cohens_d']:.4f}")


if __name__ == "__main__":
    main()
