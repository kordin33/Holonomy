"""
prepare_from_downloaded.py - Przygotuj dane z pobranego datasetu

Tworzy strukturę:
A_standardized_224/
├── train/real, train/fake
├── val/real, val/fake  
└── test_A/real, test_A/fake

B_standardized_224/
└── test_B/real, test_B/fake
"""

import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

# Konfiguracja
DATA_ROOT = Path("./data")
SOURCE = DATA_ROOT / "deepfake_vs_real"
OUTPUT_A = DATA_ROOT / "A_standardized_224"
OUTPUT_B = DATA_ROOT / "B_standardized_224"

IMG_SIZE = 224
MAX_PER_CLASS_A = 3000  # Real i Fake dla A
MAX_PER_CLASS_B = 1000  # Real i Fake dla B

random.seed(42)

print("="*60)
print("Przygotowanie danych z pobranego datasetu")
print("="*60)

# Znajdź obrazy
real_folder = SOURCE / "Real"
fake_folder = SOURCE / "Deepfake"

real_images = list(real_folder.glob("*.jpg")) + list(real_folder.glob("*.png"))
fake_images = list(fake_folder.glob("*.jpg")) + list(fake_folder.glob("*.png"))

print(f"\nZnalezione obrazy:")
print(f"  Real: {len(real_images)}")
print(f"  Fake: {len(fake_images)}")

# Shuffle
random.shuffle(real_images)
random.shuffle(fake_images)

# Split for Dataset A and B
# A: 3000 per class (train/val/test_A)
# B: 1000 per class (test_B - cross-domain)

real_for_a = real_images[:MAX_PER_CLASS_A]
fake_for_a = fake_images[:MAX_PER_CLASS_A]

real_for_b = real_images[MAX_PER_CLASS_A:MAX_PER_CLASS_A + MAX_PER_CLASS_B]
fake_for_b = fake_images[MAX_PER_CLASS_A:MAX_PER_CLASS_A + MAX_PER_CLASS_B]

print(f"\nPodział:")
print(f"  Dataset A: {len(real_for_a)} real, {len(fake_for_a)} fake")
print(f"  Dataset B: {len(real_for_b)} real, {len(fake_for_b)} fake")

# Split A into train/val/test (70/15/15)
def split_70_15_15(images):
    n = len(images)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    return {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test_A': images[val_end:],
    }

real_splits = split_70_15_15(real_for_a)
fake_splits = split_70_15_15(fake_for_a)

# Create folders
for split in ['train', 'val', 'test_A']:
    for label in ['real', 'fake']:
        (OUTPUT_A / split / label).mkdir(parents=True, exist_ok=True)

(OUTPUT_B / "test_B" / "real").mkdir(parents=True, exist_ok=True)
(OUTPUT_B / "test_B" / "fake").mkdir(parents=True, exist_ok=True)

# Copy and resize function
def copy_and_resize(images, dest_folder, prefix="img"):
    count = 0
    for img_path in tqdm(images, desc=str(dest_folder.relative_to(DATA_ROOT))):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(dest_folder / f"{prefix}_{count:05d}.jpg", "JPEG", quality=95)
            count += 1
        except Exception as e:
            continue
    return count

# Process Dataset A
print("\n" + "="*50)
print("Przetwarzanie Dataset A...")
print("="*50)

for split in ['train', 'val', 'test_A']:
    print(f"\n{split}:")
    copy_and_resize(real_splits[split], OUTPUT_A / split / "real", "real")
    copy_and_resize(fake_splits[split], OUTPUT_A / split / "fake", "fake")

# Process Dataset B
print("\n" + "="*50)
print("Przetwarzanie Dataset B...")
print("="*50)

copy_and_resize(real_for_b, OUTPUT_B / "test_B" / "real", "real")
copy_and_resize(fake_for_b, OUTPUT_B / "test_B" / "fake", "fake")

# Summary
print("\n" + "="*60)
print("✅ GOTOWE!")
print("="*60)

print("\nStruktura danych:")
for split in ['train', 'val', 'test_A']:
    for label in ['real', 'fake']:
        path = OUTPUT_A / split / label
        count = len(list(path.glob("*.jpg")))
        print(f"  A/{split}/{label}: {count}")

for label in ['real', 'fake']:
    path = OUTPUT_B / "test_B" / label
    count = len(list(path.glob("*.jpg")))
    print(f"  B/test_B/{label}: {count}")

print("\n🚀 Teraz uruchom Stage 1:")
print("   python run_stage1_local.py --visualize")
