"""
prepare_from_downloaded_fixed.py - Przygotuj dane BEZ DATA LEAKAGE

Usuwa duplikaty i zapewnia rozłączność train/test.
"""

import random
import hashlib
import shutil
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict

# Konfiguracja
DATA_ROOT = Path("./data")
SOURCE = DATA_ROOT / "deepfake_vs_real"
OUTPUT_A = DATA_ROOT / "A_standardized_224_fixed"
OUTPUT_B = DATA_ROOT / "B_standardized_224_fixed"

IMG_SIZE = 224
MAX_PER_CLASS_A = 3000
MAX_PER_CLASS_B = 1000

random.seed(42)

print("="*60)
print("Przygotowanie danych BEZ DATA LEAKAGE")
print("="*60)

def get_file_hash(path):
    """Oblicz MD5 hash pliku"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def deduplicate_images(image_paths):
    """Usuń duplikaty na podstawie contentu"""
    print(f"  Sprawdzanie {len(image_paths)} obrazów...")
    
    hash_to_path = {}
    duplicates = 0
    
    for path in tqdm(image_paths, desc="Deduplication"):
        try:
            file_hash = get_file_hash(path)
            if file_hash not in hash_to_path:
                hash_to_path[file_hash] = path
            else:
                duplicates += 1
        except:
            continue
    
    unique_paths = list(hash_to_path.values())
    print(f"  Znaleziono {duplicates} duplikatów")
    print(f"  Unikalne obrazy: {len(unique_paths)}")
    
    return unique_paths

# Znajdź obrazy
real_folder = SOURCE / "Real"
fake_folder = SOURCE / "Deepfake"

real_images = list(real_folder.glob("*.jpg")) + list(real_folder.glob("*.png"))
fake_images = list(fake_folder.glob("*.jpg")) + list(fake_folder.glob("*.png"))

print(f"\nZnalezione obrazy:")
print(f"  Real: {len(real_images)}")
print(f"  Fake: {len(fake_images)}")

# Deduplikacja
print("\n" + "="*50)
print("Usuwanie duplikatów...")
print("="*50)

print("\nReal images:")
real_images = deduplicate_images(real_images)

print("\nFake images:")
fake_images = deduplicate_images(fake_images)

# Shuffle
random.shuffle(real_images)
random.shuffle(fake_images)

# Podział: użyj pierwszych obrazów do A, resztę do B
# To gwarantuje zero overlap!
total_a = MAX_PER_CLASS_A
total_b = MAX_PER_CLASS_B

real_for_a = real_images[:total_a]
real_for_b = real_images[total_a:total_a + total_b]

fake_for_a = fake_images[:total_a]
fake_for_b = fake_images[total_a:total_a + total_b]

print(f"\nPodział (DISJOINT!):")
print(f"  Dataset A: {len(real_for_a)} real, {len(fake_for_a)} fake")
print(f"  Dataset B: {len(real_for_b)} real, {len(fake_for_b)} fake")

# Dodatkowa weryfikacja
real_a_hashes = {get_file_hash(p) for p in real_for_a}
real_b_hashes = {get_file_hash(p) for p in real_for_b}
overlap = real_a_hashes & real_b_hashes
print(f"\n  Overlap A/B real: {len(overlap)} (powinno być 0)")

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

# Weryfikacja braku overlap w A
train_real_hashes = {get_file_hash(p) for p in real_splits['train']}
test_real_hashes = {get_file_hash(p) for p in real_splits['test_A']}
train_test_overlap = train_real_hashes & test_real_hashes
print(f"  Overlap train/test_A real: {len(train_test_overlap)} (powinno być 0)")

# Create folders
for split in ['train', 'val', 'test_A']:
    for label in ['real', 'fake']:
        (OUTPUT_A / split / label).mkdir(parents=True, exist_ok=True)

(OUTPUT_B / "test_B" / "real").mkdir(parents=True, exist_ok=True)
(OUTPUT_B / "test_B" / "fake").mkdir(parents=True, exist_ok=True)

# Copy and resize function
def copy_and_resize(images, dest_folder, prefix="img"):
    count = 0
    for i, img_path in enumerate(tqdm(images, desc=str(dest_folder.name))):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(dest_folder / f"{prefix}_{i:05d}.jpg", "JPEG", quality=95)
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

# Final verification
print("\n" + "="*60)
print("WERYFIKACJA KOŃCOWA")
print("="*60)

def verify_no_leakage(train_folder, test_folder):
    train_files = list(Path(train_folder).glob("*.jpg"))
    test_files = list(Path(test_folder).glob("*.jpg"))
    
    train_hashes = {get_file_hash(p) for p in train_files}
    test_hashes = {get_file_hash(p) for p in test_files}
    
    overlap = train_hashes & test_hashes
    return len(overlap)

real_leak = verify_no_leakage(OUTPUT_A / "train" / "real", OUTPUT_A / "test_A" / "real")
fake_leak = verify_no_leakage(OUTPUT_A / "train" / "fake", OUTPUT_A / "test_A" / "fake")

print(f"Train/Test_A real overlap: {real_leak}")
print(f"Train/Test_A fake overlap: {fake_leak}")

if real_leak == 0 and fake_leak == 0:
    print("\n✅ NO DATA LEAKAGE!")
else:
    print("\n🚨 WARNING: Data leakage detected!")

# Summary
print("\n" + "="*60)
print("✅ GOTOWE!")
print("="*60)

print("\nStruktura danych:")
for split in ['train', 'val', 'test_A']:
    for label in ['real', 'fake']:
        path = OUTPUT_A / split / label
        if path.exists():
            count = len(list(path.glob("*.jpg")))
            print(f"  A/{split}/{label}: {count}")

for label in ['real', 'fake']:
    path = OUTPUT_B / "test_B" / label
    if path.exists():
        count = len(list(path.glob("*.jpg")))
        print(f"  B/test_B/{label}: {count}")

print("\n🚀 Uruchom Stage 1 z nowymi danymi:")
print("   python run_stage1_local.py --data-root ./data --max-images 2000")
print("   (zmień ścieżki w skrypcie na A_standardized_224_fixed)")
