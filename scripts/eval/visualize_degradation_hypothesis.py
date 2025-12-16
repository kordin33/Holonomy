"""
visualize_degradation_hypothesis.py - Wizualizacja Hipotezy #2 (STANDALONE)

Testuje Commutator Energy + Loop Holonomy PRZED pełnym benchmarkiem.

Sprawdza czy hipoteza separuje Real vs Fake na PaCMAP.
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
from deepfake_guard.features.degradation_commutator import (
    extract_batch_degradation_invariance,
    extract_commutator_features,
    extract_holonomy_features,
    visualize_commutator_example,
    visualize_loop_trajectory,
    COMMUTATOR_PAIRS,
    HOLONOMY_LOOPS
)


# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/hypothesis_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 100  # po 100 Real + 100 Fake = 200 total
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# QUICK DATA LOADING
# ============================================================================

def load_sample_data(sample_per_class: int = 100):
    """Ładuje małą próbkę danych."""
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
# VISUALIZATIONS
# ============================================================================

def visualize_single_hypothesis(
    hypothesis_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list = None
):
    """Wizualizuje PaCMAP dla pojedynczej hipotezy."""
    print("\n" + "=" * 50)
    print(f"VISUALIZING: {hypothesis_name}")
    print(f"Feature shape: {features.shape}")
    print("=" * 50)
    
    # Statystyki
    print("\n📊 Feature Statistics:")
    real_mask = labels == 1
    fake_mask = labels == 0
    
    real_mean = features[real_mask].mean(axis=0)
    fake_mean = features[fake_mask].mean(axis=0)
    diff = real_mean - fake_mean
    
    print("\n🔍 Real vs Fake (mean difference):")
    if feature_names:
        for i, (name, d) in enumerate(zip(feature_names, diff)):
            significance = "***" if abs(d) > 0.1 else "**" if abs(d) > 0.05 else "*" if abs(d) > 0.01 else ""
            print(f"  {name}: {d:+.4f} {significance}")
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
    
    # Separation metric
    from sklearn.metrics import silhouette_score
    try:
        sil_score = silhouette_score(proj, labels)
        print(f"\n📈 Silhouette Score (separation): {sil_score:.4f}")
        print("   (Closer to 1.0 = better separation)")
    except:
        print("\n⚠️  Could not compute silhouette score")


def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list
):
    """Porównuje rozkłady cech dla Real vs Fake."""
    print("\n" + "=" * 50)
    print("FEATURE DISTRIBUTIONS (Real vs Fake)")
    print("=" * 50)
    
    n_features = features.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
    axes = axes.flatten()
    
    real_mask = labels == 1
    fake_mask = labels == 0
    
    for i in range(n_features):
        ax = axes[i]
        
        real_feat = features[real_mask, i]
        fake_feat = features[fake_mask, i]
        
        # Histogramy
        bins = 25
        ax.hist(real_feat, bins=bins, alpha=0.5, color='#2ecc71', label='Real', density=True)
        ax.hist(fake_feat, bins=bins, alpha=0.5, color='#e74c3c', label='Fake', density=True)
        
        # T-test
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(real_feat, fake_feat)
        
        # Znacznik istotności
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        title = feature_names[i] if i < len(feature_names) else f"Feature {i}"
        ax.set_title(f"{title}\np={p_val:.2e} {sig}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide unused
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Degradation Invariance Features: Distribution Comparison", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "degrad_feature_distributions.png"
    plt.savefig(save_path, dpi=200)
    print(f"✓ Saved: {save_path}")
    plt.close()


def visualize_examples(encoder, images, labels):
    """Wizualizuje przykładowe komutatory i pętle."""
    print("\n" + "=" * 50)
    print("EXAMPLE COMMUTATORS & LOOPS")
    print("=" * 50)
    
    # Przykładowy obraz Real i Fake
    real_idx = np.where(labels == 1)[0][0]
    fake_idx = np.where(labels == 0)[0][0]
    
    real_img = images[real_idx]
    fake_img = images[fake_idx]
    
    # Commutator przykład
    print("\nGenerating commutator examples...")
    pair = COMMUTATOR_PAIRS[0]  # ('jpeg_60', 'blur_0.7')
    
    fig_real_comm = visualize_commutator_example(
        encoder, real_img, pair[0], pair[1], "Commutator: Real Image"
    )
    fig_real_comm.savefig(OUTPUT_DIR / "commutator_real_example.png", dpi=200)
    plt.close(fig_real_comm)
    
    fig_fake_comm = visualize_commutator_example(
        encoder, fake_img, pair[0], pair[1], "Commutator: Fake Image"
    )
    fig_fake_comm.savefig(OUTPUT_DIR / "commutator_fake_example.png", dpi=200)
    plt.close(fig_fake_comm)
    
    print("✓ Saved commutator examples")
    
    # Loop trajectory przykład
    print("\nGenerating loop trajectory examples...")
    loop = HOLONOMY_LOOPS[0]
    
    fig_real_loop = visualize_loop_trajectory(
        encoder, real_img, loop, "Loop Trajectory: Real Image"
    )
    fig_real_loop.savefig(OUTPUT_DIR / "loop_real_example.png", dpi=200)
    plt.close(fig_real_loop)
    
    fig_fake_loop = visualize_loop_trajectory(
        encoder, fake_img, loop, "Loop Trajectory: Fake Image"
    )
    fig_fake_loop.savefig(OUTPUT_DIR / "loop_fake_example.png", dpi=200)
    plt.close(fig_fake_loop)
    
    print("✓ Saved loop trajectory examples")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🔬 DEGRADATION INVARIANCE HYPOTHESIS (STANDALONE)")
    print("   Testing Commutator Energy + Loop Holonomy")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    images, labels = load_sample_data(SAMPLE_SIZE)
    
    # Initialize encoder
    print("\n" + "=" * 50)
    print("INITIALIZING ENCODER")
    print("=" * 50)
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract features
    print("\n" + "=" * 50)
    print("EXTRACTING DEGRADATION INVARIANCE FEATURES")
    print("   (10 Commutator + 8 Holonomy = 18 dimensions)")
    print("=" * 50)
    
    features = extract_batch_degradation_invariance(encoder, images, show_progress=True)
    
    # Feature names
    commutator_names = [f"Comm_{a}_{b}" for a, b in COMMUTATOR_PAIRS]
    holonomy_names = [f"Loop_{i+1}" for i in range(len(HOLONOMY_LOOPS))]
    feature_names = commutator_names + holonomy_names
    
    print(f"\n✓ Extracted features: {features.shape}")
    
    # Split features
    comm_features = features[:, :len(COMMUTATOR_PAIRS)]
    hol_features = features[:, len(COMMUTATOR_PAIRS):]
    
    # 1. FULL hypothesis PaCMAP
    visualize_single_hypothesis(
        "Degradation Invariance (Full)",
        features,
        labels,
        feature_names
    )
    
    # 2. Commutator ONLY
    visualize_single_hypothesis(
        "Commutator Energy Only",
        comm_features,
        labels,
        commutator_names
    )
    
    # 3. Holonomy ONLY
    visualize_single_hypothesis(
        "Loop Holonomy Only",
        hol_features,
        labels,
        holonomy_names
    )
    
    # 4. Feature distributions
    plot_feature_distributions(features, labels, feature_names)
    
    # 5. Example visualizations
    visualize_examples(encoder, images, labels)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DEGRADATION INVARIANCE VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated visualizations:")
    print("  - PaCMAP: Full 18D features")
    print("  - PaCMAP: Commutator Energy only (10D)")
    print("  - PaCMAP: Loop Holonomy only (8D)")
    print("  - Feature distributions (18 subplots)")
    print("  - Commutator examples (Real vs Fake)")
    print("  - Loop trajectory examples (Real vs Fake)")
    print("\nNext steps:")
    print("  1. Review PaCMAP separation quality")
    print("  2. Check which features discriminate best")
    print("  3. If promising → include in full benchmark!")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
