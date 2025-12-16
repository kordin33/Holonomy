"""
complete_projection_viz.py - Kompletna wizualizacja t-SNE, UMAP, PaCMAP

Generuje:
- t-SNE 2D z zoomem
- UMAP 2D z zoomem
- PaCMAP 2D z zoomem
- Porównanie wszystkich trzech w jednym obrazie
- Wersje 3D
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = Path("./results/projections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 300
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Colors for methods
COLORS = {
    "Real": "#2ecc71",
    "Inpainting": "#e74c3c",
    "Insight": "#9b59b6",
    "Text2Img": "#f39c12",
    "Wiki": "#3498db",
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    DATA_ROOT = Path("./data")
    images, methods = [], []
    
    # Real
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    if real_path.exists():
        files = list(real_path.glob("*.jpg"))[:MAX_PER_CLASS]
        print(f"Loading Real: {len(files)}")
        for p in tqdm(files, desc="Real"):
            try:
                images.append(Image.open(p).convert("RGB").resize((224, 224)))
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
            files = list(path.rglob("*.jpg"))[:MAX_PER_CLASS]
            print(f"Loading {method_name}: {len(files)}")
            for p in tqdm(files, desc=method_name):
                try:
                    images.append(Image.open(p).convert("RGB").resize((224, 224)))
                    methods.append(method_name)
                except:
                    continue
    
    return images, np.array(methods)

# ============================================================================
# PROJECTIONS
# ============================================================================

def compute_tsne(embeddings, n_components=2):
    print("Computing t-SNE...")
    tsne = TSNE(n_components=n_components, perplexity=30, random_state=RANDOM_STATE, init='pca', learning_rate='auto')
    return tsne.fit_transform(embeddings)

def compute_umap(embeddings, n_components=2):
    import umap
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE, metric='cosine')
    return reducer.fit_transform(embeddings)

def compute_pacmap(embeddings, n_components=2):
    import pacmap
    print("Computing PaCMAP...")
    reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=15, random_state=RANDOM_STATE)
    return reducer.fit_transform(embeddings)

# ============================================================================
# PLOTTING
# ============================================================================

def plot_single(projection, methods, title, save_path, figsize=(12, 10), point_size=40, alpha=0.7):
    """Plot single 2D projection with good visibility."""
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_methods = list(COLORS.keys())
    
    for method in unique_methods:
        mask = methods == method
        if mask.sum() > 0:
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=COLORS[method],
                label=f"{method} ({mask.sum()})",
                alpha=alpha,
                s=point_size,
                edgecolors='white',
                linewidth=0.5,
            )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.legend(title="Method", fontsize=10, title_fontsize=11, loc='best', markerscale=1.2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add padding to axes for better visibility
    x_margin = (projection[:, 0].max() - projection[:, 0].min()) * 0.05
    y_margin = (projection[:, 1].max() - projection[:, 1].min()) * 0.05
    ax.set_xlim(projection[:, 0].min() - x_margin, projection[:, 0].max() + x_margin)
    ax.set_ylim(projection[:, 1].min() - y_margin, projection[:, 1].max() + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comparison(projections, methods, encoder_name):
    """Plot all three projections side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    titles = ["t-SNE", "UMAP", "PaCMAP"]
    
    for ax, (name, proj), title in zip(axes, projections.items(), titles):
        unique_methods = list(COLORS.keys())
        
        for method in unique_methods:
            mask = methods == method
            if mask.sum() > 0:
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    c=COLORS[method],
                    label=method,
                    alpha=0.6,
                    s=25,
                    edgecolors='white',
                    linewidth=0.3,
                )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add margin
        x_margin = (proj[:, 0].max() - proj[:, 0].min()) * 0.05
        y_margin = (proj[:, 1].max() - proj[:, 1].min()) * 0.05
        ax.set_xlim(proj[:, 0].min() - x_margin, proj[:, 0].max() + x_margin)
        ax.set_ylim(proj[:, 1].min() - y_margin, proj[:, 1].max() + y_margin)
    
    # Single legend for all
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle(f"Projection Comparison: {encoder_name}", fontsize=16, fontweight='bold', y=1.08)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / f"comparison_{encoder_name.replace('/', '_')}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_3d(projection, methods, title, save_path):
    """Plot 3D projection from multiple angles."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(18, 5))
    angles = [(30, 45), (30, 135), (60, 225)]
    
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        for method in COLORS.keys():
            mask = methods == method
            if mask.sum() > 0:
                ax.scatter(
                    projection[mask, 0],
                    projection[mask, 1],
                    projection[mask, 2],
                    c=COLORS[method],
                    label=method if idx == 0 else "",
                    alpha=0.6,
                    s=15,
                )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("D1", fontsize=8)
        ax.set_ylabel("D2", fontsize=8)
        ax.set_zlabel("D3", fontsize=8)
        ax.set_title(f"View {idx+1}", fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🎨 COMPLETE PROJECTION VISUALIZATION")
    print("    t-SNE | UMAP | PaCMAP")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    images, methods = load_data()
    print(f"\nTotal: {len(images)} images")
    
    # Use best encoder from benchmark
    encoder_name = "ViT-L/14"
    print(f"\nUsing encoder: {encoder_name}")
    
    encoder = get_encoder("clip", encoder_name, device)
    embeddings = encoder.encode_batch(images, batch_size=32, show_progress=True)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute all projections
    projections_2d = {
        "tsne": compute_tsne(embeddings, n_components=2),
        "umap": compute_umap(embeddings, n_components=2),
        "pacmap": compute_pacmap(embeddings, n_components=2),
    }
    
    # Individual plots with good visibility
    for name, proj in projections_2d.items():
        plot_single(
            proj, methods,
            title=f"{name.upper()}: {encoder_name} Embeddings",
            save_path=OUTPUT_DIR / f"{name}_2d_{encoder_name.replace('/', '_')}.png",
            figsize=(12, 10),
            point_size=50,
            alpha=0.7,
        )
    
    # Comparison plot
    plot_comparison(projections_2d, methods, encoder_name)
    
    # 3D versions
    print("\nGenerating 3D projections...")
    
    tsne_3d = compute_tsne(embeddings, n_components=3)
    plot_3d(tsne_3d, methods, f"t-SNE 3D: {encoder_name}", OUTPUT_DIR / f"tsne_3d_{encoder_name.replace('/', '_')}.png")
    
    umap_3d = compute_umap(embeddings, n_components=3)
    plot_3d(umap_3d, methods, f"UMAP 3D: {encoder_name}", OUTPUT_DIR / f"umap_3d_{encoder_name.replace('/', '_')}.png")
    
    pacmap_3d = compute_pacmap(embeddings, n_components=3)
    plot_3d(pacmap_3d, methods, f"PaCMAP 3D: {encoder_name}", OUTPUT_DIR / f"pacmap_3d_{encoder_name.replace('/', '_')}.png")
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
