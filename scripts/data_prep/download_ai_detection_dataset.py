"""
download_ai_detection_dataset.py - Pobieranie datasetu AI vs Deepfake vs Real

Dataset: prithivMLmods/AI-vs-Deepfake-vs-Real
- 9,999 obrazów
- 3 klasy: Artificial, Deepfake, Real
- Zaktualizowany Luty 2025
"""

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os

OUTPUT_DIR = Path("./data/ai_detection_2025")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("="*70)
    print("📥 DOWNLOADING: AI vs Deepfake vs Real (HuggingFace)")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset from HuggingFace...")
    dataset = load_dataset("prithivMLmods/AI-vs-Deepfake-vs-Real")
    
    print(f"\nDataset info:")
    print(dataset)
    
    # Get the train split (usually the main one)
    if "train" in dataset:
        data = dataset["train"]
    else:
        data = dataset[list(dataset.keys())[0]]
    
    print(f"\nTotal samples: {len(data)}")
    print(f"Features: {data.features}")
    
    # Check label distribution
    if "label" in data.features:
        labels = data["label"]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        for l, c in zip(unique, counts):
            print(f"  {l}: {c}")
    
    # Save images by class
    print("\nSaving images...")
    
    for i, item in enumerate(tqdm(data, desc="Saving")):
        # Get image and label
        img = item.get("image") or item.get("Image")
        label = item.get("label") or item.get("Label")
        
        if img is None:
            continue
        
        # Convert label to folder name
        if isinstance(label, int):
            label_map = {0: "Artificial", 1: "Deepfake", 2: "Real"}
            label_name = label_map.get(label, str(label))
        else:
            label_name = str(label)
        
        # Create folder
        class_dir = OUTPUT_DIR / label_name
        class_dir.mkdir(exist_ok=True)
        
        # Save image
        if isinstance(img, Image.Image):
            img.save(class_dir / f"{i:06d}.jpg", "JPEG", quality=95)
        
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    # Show folder contents
    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg")))
            print(f"  {folder.name}: {count} images")

if __name__ == "__main__":
    import numpy as np
    main()
