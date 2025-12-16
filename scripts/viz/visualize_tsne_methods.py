"""
visualize_tsne_methods.py - Wizualizacja t-SNE dla różnych metod generowania

Pokazuje przestrzeń embeddingów z podziałem na metody:
- Real
- Fake (Inpainting)
- Fake (Insight)
- Fake (Text2Img)
- Fake (Wiki)

To pomaga zrozumieć dlaczego generalizacja działa (lub nie).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.stage1_baseline import Stage1Config
from deepfake_guard.embeddings.encoders import get_encoder

def load_images_from_folder(folder, label, max_images=300):
    images = []
    labels = []
    
    paths = list(Path(folder).rglob("*.jpg")) + list(Path(folder).rglob("*.png"))
    np.random.shuffle(paths)
    paths = paths[:max_images]
    
    print(f"Loading {label}: {len(paths)} images...")
    for p in tqdm(paths):
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            labels.append(label)
        except:
            continue
            
    return images, labels

def main():
    print("="*60)
    print("🎨 MULTI-METHOD T-SNE VISUALIZATION")
    print("="*60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder("clip", "ViT-B/32", device)
    
    data_root = Path("./data")
    
    # 1. Load Data
    all_images = []
    all_labels = []
    
    # Real (używamy fixed dataset)
    imgs, lbls = load_images_from_folder(
        data_root / "A_standardized_224_fixed/train/real", 
        "Real", 
        max_images=500
    )
    all_images.extend(imgs)
    all_labels.extend(lbls)
    
    # Fake methods
    methods = {
        "DeepFake (Inpainting)": data_root / "DeepFakeFace/_temp_inpainting",
        "DeepFake (Insight)": data_root / "DeepFakeFace/_temp_insight",
        "DeepFake (Text2Img)": data_root / "DeepFakeFace/_temp_text2img",
        "DeepFake (Wiki)": data_root / "DeepFakeFace/_temp_wiki",
    }
    
    for name, path in methods.items():
        if path.exists():
            imgs, lbls = load_images_from_folder(path, name, max_images=300)
            all_images.extend(imgs)
            all_labels.extend(lbls)
    
    print(f"\nTotal images: {len(all_images)}")
    
    # 2. Extract Embeddings
    print("\nExtracting embeddings...")
    embeddings = encoder.encode_batch(all_images, show_progress=True)
    
    # 3. Compute t-SNE
    print("\nComputing t-SNE (this may take a moment)...")
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 4. Plot
    print("Plotting...")
    plt.figure(figsize=(12, 10))
    
    # Custom colors
    unique_labels = sorted(list(set(all_labels)))
    palette = sns.color_palette("bright", len(unique_labels))
    
    # Map colors specifically if possible
    color_map = {
        "Real": "#2ecc71",              # Green
        "DeepFake (Inpainting)": "#e74c3c", # Red
        "DeepFake (Insight)": "#9b59b6",    # Purple
        "DeepFake (Text2Img)": "#e67e22",   # Orange
        "DeepFake (Wiki)": "#3498db"        # Blue
    }
    
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=all_labels,
        palette=color_map,
        alpha=0.6,
        s=40,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title("t-SNE: Real vs Distinct Deepfake Methods (CLIP ViT-B/32)", fontsize=16, fontweight='bold')
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Class / Method", fontsize=10, title_fontsize=12, loc='best')
    
    plt.tight_layout()
    output_path = "dataset_methods_tsne.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")

if __name__ == "__main__":
    main()
