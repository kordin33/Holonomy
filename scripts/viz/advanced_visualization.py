"""
advanced_visualization.py - Zaawansowana wizualizacja embeddingów

Metody projekcji:
1. t-SNE 2D (baseline)
2. t-SNE 3D (interaktywna rotacja)
3. UMAP 2D (lepsza globalna struktura)
4. UMAP 3D

UMAP jest znacznie szybszy i lepiej zachowuje globalną topologię niż t-SNE!
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# VISUALIZATION BACKENDS
# ============================================================================

def compute_tsne(embeddings, n_components=2, perplexity=30):
    """Compute t-SNE projection."""
    from sklearn.manifold import TSNE
    
    print(f"Computing t-SNE ({n_components}D)...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto',
    )
    return tsne.fit_transform(embeddings)


def compute_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Compute UMAP projection.
    
    UMAP (Uniform Manifold Approximation and Projection) advantages over t-SNE:
    - Faster (especially for large datasets)
    - Better preserves global structure
    - More meaningful distances between clusters
    - Deterministic with random_state
    """
    try:
        import umap
    except ImportError:
        print("Installing umap-learn...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
        import umap
    
    print(f"Computing UMAP ({n_components}D)...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric='cosine',
    )
    return reducer.fit_transform(embeddings)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sample_data(max_per_class=300):
    """Load sample for visualization."""
    DATA_ROOT = Path("./data")
    
    images = []
    labels = []
    methods = []
    
    # Real
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    if real_path.exists():
        files = list(real_path.glob("*.jpg"))[:max_per_class]
        for p in tqdm(files, desc="Loading Real"):
            try:
                images.append(Image.open(p).convert("RGB").resize((224, 224)))
                labels.append("real")
                methods.append("Real")
            except:
                continue
    
    # Fakes
    fake_sources = {
        "Inpainting": DATA_ROOT / "DeepFakeFace/_temp_inpainting",
        "Insight": DATA_ROOT / "DeepFakeFace/_temp_insight",
        "Text2Img": DATA_ROOT / "DeepFakeFace/_temp_text2img",
        "Wiki": DATA_ROOT / "DeepFakeFace/_temp_wiki",
    }
    
    for method_name, path in fake_sources.items():
        if path.exists():
            files = list(path.rglob("*.jpg"))[:max_per_class]
            for p in tqdm(files, desc=f"Loading {method_name}"):
                try:
                    images.append(Image.open(p).convert("RGB").resize((224, 224)))
                    labels.append("fake")
                    methods.append(method_name)
                except:
                    continue
    
    return images, np.array(labels), np.array(methods)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_2d(projection, methods, title, save_path):
    """Plot 2D projection."""
    plt.figure(figsize=(12, 10))
    
    unique_methods = np.unique(methods)
    colors = {
        "Real": "#2ecc71",
        "Inpainting": "#e74c3c",
        "Insight": "#9b59b6",
        "Text2Img": "#e67e22",
        "Wiki": "#3498db",
    }
    
    for method in unique_methods:
        mask = methods == method
        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            c=colors.get(method, "#95a5a6"),
            label=method,
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.3,
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title="Method", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_3d(projection, methods, title, save_path):
    """Plot 3D projection with multiple angles."""
    fig = plt.figure(figsize=(16, 6))
    
    unique_methods = np.unique(methods)
    colors = {
        "Real": "#2ecc71",
        "Inpainting": "#e74c3c",
        "Insight": "#9b59b6",
        "Text2Img": "#e67e22",
        "Wiki": "#3498db",
    }
    
    # Three different viewing angles
    angles = [(30, 45), (30, 135), (60, 225)]
    
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        for method in unique_methods:
            mask = methods == method
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                projection[mask, 2],
                c=colors.get(method, "#95a5a6"),
                label=method if idx == 0 else "",
                alpha=0.6,
                s=20,
            )
        
        ax.set_xlabel("D1")
        ax.set_ylabel("D2")
        ax.set_zlabel("D3")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {idx+1}", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comparison(embeddings, methods, encoder_name, output_dir):
    """Generate all visualization variants."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. t-SNE 2D
    tsne_2d = compute_tsne(embeddings, n_components=2)
    plot_2d(tsne_2d, methods, f"t-SNE 2D: {encoder_name}", output_dir / f"{encoder_name}_tsne_2d.png")
    
    # 2. t-SNE 3D
    tsne_3d = compute_tsne(embeddings, n_components=3)
    plot_3d(tsne_3d, methods, f"t-SNE 3D: {encoder_name}", output_dir / f"{encoder_name}_tsne_3d.png")
    
    # 3. UMAP 2D
    umap_2d = compute_umap(embeddings, n_components=2)
    plot_2d(umap_2d, methods, f"UMAP 2D: {encoder_name}", output_dir / f"{encoder_name}_umap_2d.png")
    
    # 4. UMAP 3D
    umap_3d = compute_umap(embeddings, n_components=3)
    plot_3d(umap_3d, methods, f"UMAP 3D: {encoder_name}", output_dir / f"{encoder_name}_umap_3d.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🎨 ADVANCED EMBEDDING VISUALIZATION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("./results/visualizations")
    
    # Load data
    print("\nLoading data...")
    images, labels, methods = load_sample_data(max_per_class=400)
    print(f"Total: {len(images)} images")
    
    # Test multiple encoders
    encoders_to_test = [
        ("CLIP_ViT-B-32", "clip", "ViT-B/32"),
        ("CLIP_ViT-L-14", "clip", "ViT-L/14"),
        # ("DINOv2", "dinov2", None),  # Uncomment if DINOv2 is installed
    ]
    
    for encoder_name, encoder_type, variant in encoders_to_test:
        print(f"\n{'='*50}")
        print(f"Encoder: {encoder_name}")
        print("="*50)
        
        try:
            encoder = get_encoder(encoder_type, variant or "vitb14", device)
            
            # Extract embeddings
            embeddings = encoder.encode_batch(images, show_progress=True)
            print(f"Embedding shape: {embeddings.shape}")
            
            # Generate all visualizations
            plot_comparison(embeddings, methods, encoder_name, output_dir)
            
            # Cleanup
            del encoder
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {encoder_name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("✅ All visualizations complete!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
