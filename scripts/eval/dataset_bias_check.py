"""
dataset_bias_check.py - Sprawdzenie biasu w datasecie

Czy Real = portrety, a Fake = różne pozy?
To by oznaczało że model uczy się wykrywać portrety, nie deepfake!
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("./results/dataset_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("./data")

def load_random_samples(folder, n=20):
    """Load random samples from folder."""
    files = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    random.shuffle(files)
    
    images = []
    for p in files[:n]:
        try:
            img = Image.open(p).convert("RGB")
            images.append((img, p.name))
        except:
            continue
    return images

def plot_grid(images, title, save_path, cols=5):
    """Plot grid of images."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    
    for ax, (img, name) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(name[:15] + "..." if len(name) > 15 else name, fontsize=8)
        ax.axis('off')
    
    # Hide empty axes
    for ax in axes[len(images):]:
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def main():
    print("="*70)
    print("🔍 DATASET BIAS CHECK")
    print("    Czy Real = portrety, Fake = różne pozy?")
    print("="*70)
    
    # Real samples
    print("\nLoading Real samples...")
    real_path = DATA_ROOT / "deepfake_vs_real/Real"
    real_samples = load_random_samples(real_path, n=25)
    print(f"  Loaded: {len(real_samples)} Real images")
    
    # Fake samples from each method
    fake_sources = {
        "Inpainting": DATA_ROOT / "DeepFakeFace/_temp_inpainting",
        "Insight": DATA_ROOT / "DeepFakeFace/_temp_insight",
        "Text2Img": DATA_ROOT / "DeepFakeFace/_temp_text2img",
        "Wiki": DATA_ROOT / "DeepFakeFace/_temp_wiki",
    }
    
    all_fake_samples = []
    for method, path in fake_sources.items():
        if path.exists():
            samples = load_random_samples(path, n=6)
            print(f"  Loaded: {len(samples)} {method} images")
            all_fake_samples.extend([(img, f"{method}: {name}") for img, name in samples])
    
    # Plot Real
    plot_grid(real_samples, 
              "REAL Images - Check: Are they all frontal portraits?",
              OUTPUT_DIR / "real_samples.png", cols=5)
    
    # Plot Fake by method
    for method, path in fake_sources.items():
        if path.exists():
            samples = load_random_samples(path, n=25)
            plot_grid(samples,
                      f"FAKE ({method}) - Check: Poses, backgrounds, compositions",
                      OUTPUT_DIR / f"fake_{method.lower()}_samples.png", cols=5)
    
    # Side-by-side comparison
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Top row: Real
    for ax, (img, name) in zip(axes[0], real_samples[:5]):
        ax.imshow(img)
        ax.set_title("REAL", fontsize=10, color='green', fontweight='bold')
        ax.axis('off')
    
    # Bottom row: Fake (mixed)
    random.shuffle(all_fake_samples)
    for ax, (img, name) in zip(axes[1], all_fake_samples[:5]):
        ax.imshow(img)
        ax.set_title("FAKE", fontsize=10, color='red', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle("BIAS CHECK: Real vs Fake - Look for composition differences!", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_real_vs_fake.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: comparison_real_vs_fake.png")
    plt.close()
    
    print("\n" + "="*70)
    print("📋 CHECKLIST - Sprawdź wizualnie:")
    print("="*70)
    print("""
    1. Czy Real to TYLKO portrety frontalne?
    2. Czy Fake ma różne pozy, kadry, tła?
    3. Czy różnica w kompozycji jest systematyczna?
    
    Jeśli TAK - model uczy się kompozycji, nie deepfake!
    
    ROZWIĄZANIE:
    - Użyć datasetu gdzie Real i Fake mają podobne kompozycje
    - Lub: Użyć face detection + cropping do normalizacji
    """)
    print("="*70)
    print(f"\nOtwórz folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
