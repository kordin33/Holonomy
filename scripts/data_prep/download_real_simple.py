"""
download_real_simple.py - Pobierz REAL obrazy prostą metodą

Używa bezpośredniego pobrania małymi porcjami.
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

# Pobierz najpierw cały dataset do cache, potem przetwarzaj
print("Pobieranie datasetu (to może chwilę potrwać)...")

try:
    from datasets import load_dataset
except:
    os.system("pip install datasets")
    from datasets import load_dataset

DATA_ROOT = Path("./data")
REAL_DIR = DATA_ROOT / "real_faces_temp"
REAL_DIR.mkdir(parents=True, exist_ok=True)

MAX_REAL = 2000
IMG_SIZE = 224

# Sprawdź co już mamy
existing = list(REAL_DIR.glob("*.jpg"))
print(f"Już mamy: {len(existing)} real obrazów")

if len(existing) >= MAX_REAL:
    print("Wystarczająco! Skipping...")
else:
    needed = MAX_REAL - len(existing)
    print(f"Potrzebujemy jeszcze: {needed}")
    
    print("\nPobieram dataset (cache'owany lokalnie)...")
    
    # Pobierz do cache - to ściągnie cały dataset raz
    ds = load_dataset(
        "prithivMLmods/Deepfake-vs-Real-v2",
        split="train",
        # NIE streaming - cały dataset do cache
    )
    
    print(f"Dataset załadowany: {len(ds)} samples")
    
    # Filtruj tylko real (label=1)
    real_indices = [i for i, item in enumerate(ds) if item.get("label", 0) == 1]
    print(f"Real samples w datasecie: {len(real_indices)}")
    
    # Zapisz
    count = len(existing)
    
    for i in tqdm(real_indices[:needed], desc="Zapisuję REAL"):
        try:
            item = ds[i]
            img = item["image"].convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(REAL_DIR / f"real_{count:05d}.jpg", "JPEG", quality=95)
            count += 1
        except Exception as e:
            continue
    
    print(f"\n✅ Zapisano {count} real obrazów")

# Teraz przygotuj strukturę A_standardized_224
print("\n" + "="*50)
print("Przygotowuję strukturę danych...")
print("="*50)

import random
random.seed(42)

# Fake z rozpakowanych ZIP-ów
DEEPFAKE_DIR = DATA_ROOT / "DeepFakeFace"
temp_dirs = list(DEEPFAKE_DIR.glob("_temp_*"))

all_fake = []
for temp_dir in temp_dirs:
    all_fake.extend(list(temp_dir.rglob("*.jpg")))
    all_fake.extend(list(temp_dir.rglob("*.png")))

print(f"Fake obrazów: {len(all_fake)}")

# Real 
all_real = list(REAL_DIR.glob("*.jpg"))
print(f"Real obrazów: {len(all_real)}")

if len(all_real) == 0:
    print("❌ Brak real obrazów!")
    sys.exit(1)

# Shuffle
random.shuffle(all_fake)
random.shuffle(all_real)

# Ogranicz do 2000
all_fake = all_fake[:2000]
all_real = all_real[:2000]

# Split 70/15/15
def split_list(lst):
    n = len(lst)
    return {
        'train': lst[:int(0.7*n)],
        'val': lst[int(0.7*n):int(0.85*n)],
        'test_A': lst[int(0.85*n):],
    }

fake_splits = split_list(all_fake)
real_splits = split_list(all_real)

# Output dir
OUT_DIR = DATA_ROOT / "A_standardized_224"

for split in ['train', 'val', 'test_A']:
    for label in ['real', 'fake']:
        (OUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

# Copy and resize
def copy_images(images, out_dir, label, split):
    dst = out_dir / split / label
    print(f"  {split}/{label}: {len(images)} obrazów")
    
    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(dst / f"{label}_{i:05d}.jpg", "JPEG", quality=95)
        except:
            continue

for split in ['train', 'val', 'test_A']:
    copy_images(real_splits[split], OUT_DIR, "real", split)
    copy_images(fake_splits[split], OUT_DIR, "fake", split)

# Dataset B
print("\n" + "="*50)
print("Przygotowuję Dataset B...")
print("="*50)

OUT_B = DATA_ROOT / "B_standardized_224" / "test_B"
(OUT_B / "real").mkdir(parents=True, exist_ok=True)
(OUT_B / "fake").mkdir(parents=True, exist_ok=True)

# Użyj pozostałych obrazów z datasetu
try:
    real_for_b = [ds[i] for i in real_indices[2000:3000]]
    fake_indices = [i for i, item in enumerate(ds) if item.get("label", 0) == 0]
    fake_for_b = [ds[i] for i in fake_indices[:1000]]
    
    print(f"  B/real: zapisuję {len(real_for_b)}...")
    for i, item in enumerate(tqdm(real_for_b, desc="B/real")):
        try:
            img = item["image"].convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(OUT_B / "real" / f"real_{i:05d}.jpg", "JPEG", quality=95)
        except:
            continue
    
    print(f"  B/fake: zapisuję {len(fake_for_b)}...")
    for i, item in enumerate(tqdm(fake_for_b, desc="B/fake")):
        try:
            img = item["image"].convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(OUT_B / "fake" / f"fake_{i:05d}.jpg", "JPEG", quality=95)
        except:
            continue
            
except Exception as e:
    print(f"Błąd przy B: {e}")

# Podsumowanie
print("\n" + "="*60)
print("✅ GOTOWE!")
print("="*60)

for split in ['train', 'val', 'test_A']:
    for label in ['real', 'fake']:
        path = OUT_DIR / split / label
        if path.exists():
            count = len(list(path.glob("*.jpg")))
            print(f"  A/{split}/{label}: {count}")

for label in ['real', 'fake']:
    path = OUT_B / label
    if path.exists():
        count = len(list(path.glob("*.jpg")))
        print(f"  B/test_B/{label}: {count}")

print("\n🚀 Uruchom teraz:")
print("   python run_stage1_local.py")
