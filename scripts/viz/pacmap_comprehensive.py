"""
pacmap_comprehensive.py - Kompleksowa wizualizacja PaCMAP

Porównanie:
1. RGB vs RGB+FFT embeddings
2. k-NN vs SVM predictions
3. Błędy klasyfikacji

Wszystko na ViT-L/14 + CUDA
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pacmap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = Path("./results/pacmap_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 300
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

COLORS = {
    "Real": "#2ecc71",
    "Inpainting": "#e74c3c",
    "Insight": "#9b59b6",
    "Text2Img": "#f39c12",
    "Wiki": "#3498db",
}

# ============================================================================
# FFT FEATURES
# ============================================================================

def extract_fft_features(image: Image.Image, feature_dim: int = 64) -> np.ndarray:
    img_gray = np.array(image.convert('L'), dtype=np.float32)
    fft = np.fft.fft2(img_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)
    
    h, w = log_magnitude.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    max_r = min(center_x, center_y)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.sum() > 0:
            radial_profile[i] = log_magnitude[mask].mean()
    
    if len(radial_profile) > feature_dim:
        indices = np.linspace(0, len(radial_profile)-1, feature_dim).astype(int)
        features = radial_profile[indices]
    else:
        features = np.zeros(feature_dim)
        features[:len(radial_profile)] = radial_profile
    
    features = (features - features.mean()) / (features.std() + 1e-8)
    return features.astype(np.float32)

# ============================================================================
# DATA
# ============================================================================

def load_data():
    DATA_ROOT = Path("./data")
    images, labels, methods = [], [], []
    
    # Real
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    if real_path.exists():
        files = list(real_path.glob("*.jpg"))[:MAX_PER_CLASS]
        print(f"Loading Real: {len(files)}")
        for p in tqdm(files, desc="Real"):
            try:
                images.append(Image.open(p).convert("RGB").resize((224, 224)))
                labels.append(1)  # 1 = real
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
                    labels.append(0)  # 0 = fake
                    methods.append(method_name)
                except:
                    continue
    
    return images, np.array(labels), np.array(methods)

# ============================================================================
# PLOTTING
# ============================================================================

def plot_pacmap(projection, colors_arr, title, save_path, legend_labels=None, point_size=40, alpha=0.7):
    """Plot PaCMAP with custom colors."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if legend_labels:
        for label in legend_labels:
            mask = colors_arr == label
            if mask.sum() > 0:
                ax.scatter(
                    projection[mask, 0], projection[mask, 1],
                    c=COLORS.get(label, '#95a5a6'),
                    label=f"{label} ({mask.sum()})",
                    alpha=alpha, s=point_size,
                    edgecolors='white', linewidth=0.5,
                )
    else:
        scatter = ax.scatter(
            projection[:, 0], projection[:, 1],
            c=colors_arr, cmap='RdYlGn',
            alpha=alpha, s=point_size,
            edgecolors='white', linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax, label='Confidence')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("PaCMAP Dimension 1", fontsize=12)
    ax.set_ylabel("PaCMAP Dimension 2", fontsize=12)
    if legend_labels:
        ax.legend(fontsize=10, loc='best', markerscale=1.2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comparison_grid(projections, methods, predictions, title, save_path):
    """Plot 2x2 grid comparing features and classifiers."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    titles = [
        ("RGB - Ground Truth", "rgb", None),
        ("RGB+FFT - Ground Truth", "rgb_fft", None),
        ("RGB - k-NN Predictions", "rgb", "knn"),
        ("RGB - SVM Predictions", "rgb", "svm"),
    ]
    
    for ax, (subtitle, feature_key, clf_key) in zip(axes.flatten(), titles):
        proj = projections[feature_key]
        
        if clf_key is None:
            # Ground truth - color by method
            for method in COLORS.keys():
                mask = methods == method
                if mask.sum() > 0:
                    ax.scatter(
                        proj[mask, 0], proj[mask, 1],
                        c=COLORS[method], label=method,
                        alpha=0.6, s=25, edgecolors='white', linewidth=0.3,
                    )
        else:
            # Predictions - color by correct/incorrect
            preds = predictions[f"{feature_key}_{clf_key}"]
            labels_binary = (methods == "Real").astype(int)
            correct = preds == labels_binary
            
            # Correct predictions
            ax.scatter(
                proj[correct, 0], proj[correct, 1],
                c='#2ecc71', label=f'Correct ({correct.sum()})',
                alpha=0.6, s=25, edgecolors='white', linewidth=0.3,
            )
            # Incorrect predictions
            ax.scatter(
                proj[~correct, 0], proj[~correct, 1],
                c='#e74c3c', label=f'Error ({(~correct).sum()})',
                alpha=0.8, s=40, edgecolors='black', linewidth=0.5,
                marker='x',
            )
        
        ax.set_title(subtitle, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Dim 1", fontsize=9)
        ax.set_ylabel("Dim 2", fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_rgb_vs_fft(proj_rgb, proj_fft, methods, save_path):
    """Side by side RGB vs RGB+FFT."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax, proj, title in zip(axes, [proj_rgb, proj_fft], ["RGB Only (768 dim)", "RGB + FFT (832 dim)"]):
        for method in COLORS.keys():
            mask = methods == method
            if mask.sum() > 0:
                ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    c=COLORS[method], label=method,
                    alpha=0.7, s=40, edgecolors='white', linewidth=0.4,
                )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PaCMAP Dim 1", fontsize=11)
        ax.set_ylabel("PaCMAP Dim 2", fontsize=11)
    
    plt.suptitle("PaCMAP Comparison: RGB vs RGB+FFT Features (ViT-L/14)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_classifier_comparison(proj, methods, preds_knn, preds_svm, save_path):
    """Compare kNN vs SVM errors."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    labels_binary = (methods == "Real").astype(int)
    
    for ax, preds, clf_name in zip(axes, [preds_knn, preds_svm], ["k-NN", "SVM"]):
        correct = preds == labels_binary
        acc = correct.mean()
        
        # Correct
        ax.scatter(
            proj[correct, 0], proj[correct, 1],
            c='#2ecc71', label=f'Correct ({correct.sum()})',
            alpha=0.5, s=30, edgecolors='white', linewidth=0.3,
        )
        # Errors
        ax.scatter(
            proj[~correct, 0], proj[~correct, 1],
            c='#e74c3c', label=f'Errors ({(~correct).sum()})',
            alpha=0.9, s=60, edgecolors='black', linewidth=1,
            marker='X',
        )
        
        ax.set_title(f"{clf_name} (Accuracy: {acc:.1%})", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PaCMAP Dim 1", fontsize=11)
        ax.set_ylabel("PaCMAP Dim 2", fontsize=11)
    
    plt.suptitle("PaCMAP: Classifier Comparison - k-NN vs SVM (ViT-L/14 RGB)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🎨 PACMAP COMPREHENSIVE ANALYSIS")
    print("   ViT-L/14 | RGB vs RGB+FFT | k-NN vs SVM")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    images, labels, methods = load_data()
    print(f"\nTotal: {len(images)} images")
    
    # Extract RGB embeddings
    print("\n" + "="*50)
    print("EXTRACTING EMBEDDINGS (ViT-L/14)")
    print("="*50)
    
    encoder = get_encoder("clip", "ViT-L/14", device)
    rgb_embeddings = encoder.encode_batch(images, batch_size=32, show_progress=True)
    print(f"RGB shape: {rgb_embeddings.shape}")
    
    # Extract FFT features
    print("\nExtracting FFT features...")
    fft_features = np.array([extract_fft_features(img) for img in tqdm(images, desc="FFT")])
    print(f"FFT shape: {fft_features.shape}")
    
    # Combine
    rgb_fft_embeddings = np.hstack([rgb_embeddings, fft_features])
    print(f"RGB+FFT shape: {rgb_fft_embeddings.shape}")
    
    # Split for classifier training
    train_idx, test_idx = train_test_split(
        np.arange(len(images)), test_size=0.3, random_state=RANDOM_STATE, stratify=methods
    )
    
    # Train classifiers
    print("\n" + "="*50)
    print("TRAINING CLASSIFIERS")
    print("="*50)
    
    classifiers = {}
    predictions = {}
    
    for feature_name, embeddings in [("rgb", rgb_embeddings), ("rgb_fft", rgb_fft_embeddings)]:
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        for clf_name, clf in [("knn", KNeighborsClassifier(n_neighbors=10, metric='cosine')),
                              ("svm", SVC(kernel='rbf', random_state=RANDOM_STATE))]:
            print(f"Training {clf_name} on {feature_name}...")
            clf.fit(X_train, y_train)
            
            # Predict on ALL data for visualization
            preds = clf.predict(embeddings)
            predictions[f"{feature_name}_{clf_name}"] = preds
            
            acc = (preds[test_idx] == y_test).mean()
            print(f"  Test accuracy: {acc:.2%}")
    
    # Compute PaCMAP projections
    print("\n" + "="*50)
    print("COMPUTING PACMAP PROJECTIONS")
    print("="*50)
    
    print("PaCMAP on RGB...")
    pacmap_rgb = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj_rgb = pacmap_rgb.fit_transform(rgb_embeddings)
    
    print("PaCMAP on RGB+FFT...")
    pacmap_fft = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE)
    proj_fft = pacmap_fft.fit_transform(rgb_fft_embeddings)
    
    projections = {"rgb": proj_rgb, "rgb_fft": proj_fft}
    
    # Generate visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # 1. RGB only - by method
    plot_pacmap(proj_rgb, methods, 
                "PaCMAP: ViT-L/14 RGB Embeddings (768 dim)",
                OUTPUT_DIR / "pacmap_rgb_methods.png",
                legend_labels=list(COLORS.keys()))
    
    # 2. RGB+FFT - by method
    plot_pacmap(proj_fft, methods,
                "PaCMAP: ViT-L/14 RGB+FFT Embeddings (832 dim)",
                OUTPUT_DIR / "pacmap_rgb_fft_methods.png",
                legend_labels=list(COLORS.keys()))
    
    # 3. RGB vs FFT comparison
    plot_rgb_vs_fft(proj_rgb, proj_fft, methods,
                    OUTPUT_DIR / "pacmap_rgb_vs_fft.png")
    
    # 4. k-NN vs SVM comparison (on RGB)
    plot_classifier_comparison(proj_rgb, methods,
                               predictions["rgb_knn"], predictions["rgb_svm"],
                               OUTPUT_DIR / "pacmap_knn_vs_svm.png")
    
    # 5. Full comparison grid
    plot_comparison_grid(projections, methods, predictions,
                         "PaCMAP Analysis: Features & Classifiers (ViT-L/14)",
                         OUTPUT_DIR / "pacmap_full_comparison.png")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("✅ PACMAP ANALYSIS COMPLETE!")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
