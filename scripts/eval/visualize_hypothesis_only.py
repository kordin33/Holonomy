"""
visualize_hypothesis_only.py - Wizualizacja SAMYCH cech z hipotezy

Sprawdza czy hipoteza matematyczna (BEZ RGB embeddings) 
sama w sobie separuje Real vs Fake.

Quick test przed pełnym benchmarkiem - używamy małej próbki!
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pacmap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.quantization_scaling import (
    extract_batch_quantization_scaling,
    extract_scaling_exponent,
    plot_scaling_law
)


# ============================================================================
# CONFIG - MAŁA PRÓBKA DO SZYBKIEGO TESTU
# ============================================================================

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/hypothesis_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 100  # po 100 Real + 100 Fake = 200 total (szybki test!)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# QUICK DATA LOADING
# ============================================================================

def load_sample_data(sample_per_class: int = 100):
    """
    Ładuje małą próbkę danych do szybkiego testu.
    """
    print("=" * 50)
    print(f"LOADING SAMPLE DATA ({sample_per_class} per class)")
    print("=" * 50)
    
    images, labels = [], []
    
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:sample_per_class]
        
        for p in tqdm(files, desc=f"Loading {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    
    print(f"✓ Loaded {len(images)} images ({sum(labels)} Real, {len(labels) - sum(labels)} Fake)")
    
    return images, np.array(labels)


# ============================================================================
# VISUALIZE SINGLE HYPOTHESIS
# ============================================================================

def visualize_single_hypothesis(
    hypothesis_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list = None
):
    """
    Wizualizuje PaCMAP dla pojedynczej hipotezy.
    
    Args:
        hypothesis_name: Nazwa hipotezy
        features: Array (N, D) - cechy z hipotezy
        labels: Array (N,) - 0=Fake, 1=Real
        feature_names: Opcjonalne nazwy wymiarów
    """
    print("\n" + "=" * 50)
    print(f"VISUALIZING: {hypothesis_name}")
    print(f"Feature shape: {features.shape}")
    print("=" * 50)
    
    # Statystyki cech
    print("\n📊 Feature Statistics:")
    print(f"  Mean: {features.mean(axis=0)}")
    print(f"  Std:  {features.std(axis=0)}")
    
    # Różnice Real vs Fake
    real_mask = labels == 1
    fake_mask = labels == 0
    
    print("\n🔍 Real vs Fake (mean difference):")
    real_mean = features[real_mask].mean(axis=0)
    fake_mean = features[fake_mask].mean(axis=0)
    diff = real_mean - fake_mean
    
    if feature_names:
        for i, (name, d) in enumerate(zip(feature_names, diff)):
            print(f"  {name}: {d:+.4f}")
    else:
        for i, d in enumerate(diff):
            print(f"  Feature {i}: {d:+.4f}")
    
    # PaCMAP
    print("\n🗺️  Computing PaCMAP...")
    pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj = pacmap_model.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {0: "#e74c3c", 1: "#2ecc71"}
    labels_map = {0: "Fake (SD)", 1: "Real (CIFAR)"}
    
    for label in [0, 1]:
        mask = labels == label
        ax.scatter(proj[mask, 0], proj[mask, 1],
                  c=colors[label], label=labels_map[label],
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax.set_title(f"PaCMAP: {hypothesis_name} (Standalone)\n{features.shape[1]} dimensions", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PaCMAP Dim 1", fontsize=11)
    ax.set_ylabel("PaCMAP Dim 2", fontsize=11)
    
    plt.tight_layout()
    
    # Save
    safe_name = hypothesis_name.lower().replace(' ', '_')
    save_path = OUTPUT_DIR / f"pacmap_{safe_name}_only.png"
    plt.savefig(save_path, dpi=200)
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Calculate separation metric (silhouette-like)
    from sklearn.metrics import silhouette_score
    try:
        sil_score = silhouette_score(proj, labels)
        print(f"\n📈 Silhouette Score (separation): {sil_score:.4f}")
        print("   (Closer to 1.0 = better separation)")
    except:
        print("\n⚠️  Could not compute silhouette score")


# ============================================================================
# VISUALIZE EXAMPLE SCALING LAWS
# ============================================================================

def visualize_example_scaling_laws(
    encoder,
    images: list,
    labels: np.ndarray,
    n_examples: int = 4
):
    """
    Wizualizuje przykładowe prawa skali dla Real i Fake.
    """
    print("\n" + "=" * 50)
    print("EXAMPLE SCALING LAWS (Real vs Fake)")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, n_examples, figsize=(16, 8))
    
    for cls_idx, (cls_label, cls_name) in enumerate([(1, "Real"), (0, "Fake")]):
        cls_mask = labels == cls_label
        cls_images = [img for img, m in zip(images, cls_mask) if m]
        
        # Wybierz losowe przykłady
        indices = np.random.choice(len(cls_images), n_examples, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[cls_idx, i]
            
            img = cls_images[idx]
            deltas = [4, 8, 16, 32, 64]
            
            # Extract scaling
            stats = extract_scaling_exponent(encoder, img, deltas)
            
            # Plot log-log
            from scipy.stats import linregress
            
            S_values = []
            for delta in deltas:
                from deepfake_guard.features.quantization_scaling import measure_embedding_distance
                S = measure_embedding_distance(encoder, img, delta)
                S_values.append(S)
            
            S_values = np.array(S_values)
            log_deltas = np.log(deltas)
            log_S = np.log(S_values + 1e-10)
            
            # Plot
            ax.plot(deltas, S_values, 'o', markersize=6, label='Measured')
            
            # Fitted line
            deltas_fit = np.linspace(min(deltas), max(deltas), 100)
            S_fit = np.exp(stats['alpha'] * np.log(deltas_fit) + stats['intercept'])
            ax.plot(deltas_fit, S_fit, 'r--', 
                   label=f'α={stats["alpha"]:.3f}\nR²={stats["r_squared"]:.3f}')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"{cls_name} #{i+1}", fontsize=10)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_ylabel('log S(Δ)', fontsize=9)
            if cls_idx == 1:
                ax.set_xlabel('log Δ', fontsize=9)
    
    plt.suptitle("Quantization Scaling Laws: Real vs Fake Examples", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "scaling_laws_examples.png"
    plt.savefig(save_path, dpi=200)
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# FEATURE DISTRIBUTION COMPARISON
# ============================================================================

def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list
):
    """
    Porównuje rozkłady cech dla Real vs Fake.
    """
    print("\n" + "=" * 50)
    print("FEATURE DISTRIBUTIONS (Real vs Fake)")
    print("=" * 50)
    
    n_features = features.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()
    
    real_mask = labels == 1
    fake_mask = labels == 0
    
    for i in range(n_features):
        ax = axes[i]
        
        real_feat = features[real_mask, i]
        fake_feat = features[fake_mask, i]
        
        # Histogramy
        bins = 30
        ax.hist(real_feat, bins=bins, alpha=0.5, color='#2ecc71', label='Real', density=True)
        ax.hist(fake_feat, bins=bins, alpha=0.5, color='#e74c3c', label='Fake', density=True)
        
        # Statystyki
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(real_feat, fake_feat)
        
        title = feature_names[i] if i < len(feature_names) else f"Feature {i}"
        ax.set_title(f"{title}\np={p_val:.2e}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Quantization Scaling Features: Distribution Comparison", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(save_path, dpi=200)
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🔬 HYPOTHESIS VISUALIZATION (STANDALONE)")
    print("   Testing Quantization Scaling hypothesis BEFORE full benchmark")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load sample data
    images, labels = load_sample_data(SAMPLE_SIZE)
    
    # Initialize encoder
    print("\n" + "=" * 50)
    print("INITIALIZING ENCODER")
    print("=" * 50)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract Quantization Scaling features
    print("\n" + "=" * 50)
    print("EXTRACTING QUANTIZATION SCALING FEATURES")
    print("=" * 50)
    features = extract_batch_quantization_scaling(encoder, images, show_progress=True)
    
    feature_names = [
        'alpha',
        'r_squared',
        'residual_std',
        'intercept',
        'mean_S',
        'std_S',
        'p_value',
        'std_err'
    ]
    
    print(f"\n✓ Extracted features: {features.shape}")
    
    # 1. PaCMAP visualization
    visualize_single_hypothesis(
        "Quantization Scaling",
        features,
        labels,
        feature_names
    )
    
    # 2. Example scaling laws
    visualize_example_scaling_laws(encoder, images, labels, n_examples=4)
    
    # 3. Feature distributions
    plot_feature_distributions(features, labels, feature_names)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ HYPOTHESIS VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review PaCMAP to see if hypothesis separates Real/Fake")
    print("  2. Check scaling law examples for Real vs Fake differences")
    print("  3. Analyze feature distributions")
    print("  4. If promising → add to full benchmark!")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
