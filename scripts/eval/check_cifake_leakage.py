"""
check_cifake_leakage.py - Sprawdzenie data leakage w CIFAKE
"""

import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

print("="*70)
print("🔍 CIFAKE DATA LEAKAGE CHECK")
print("="*70)

data_dir = Path("./data/cifake")

def get_image_hash(path):
    try:
        img = Image.open(path)
        return hashlib.md5(np.array(img).tobytes()).hexdigest()
    except:
        return None

# Collect hashes
train_hashes = {}
test_hashes = {}

print("\nHashing train images...")
for cls in ["REAL", "FAKE"]:
    files = list((data_dir/"train"/cls).glob("*.jpg"))[:5000]
    for p in tqdm(files, desc=f"Train {cls}"):
        h = get_image_hash(p)
        if h:
            train_hashes[h] = p

print("\nHashing test images...")
for cls in ["REAL", "FAKE"]:
    files = list((data_dir/"test"/cls).glob("*.jpg"))[:2000]
    for p in tqdm(files, desc=f"Test {cls}"):
        h = get_image_hash(p)
        if h:
            test_hashes[h] = p

# Check overlap
overlap = set(train_hashes.keys()) & set(test_hashes.keys())

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)
print(f"Train unique hashes: {len(train_hashes)}")
print(f"Test unique hashes: {len(test_hashes)}")
print(f"Overlap (leakage): {len(overlap)}")

if len(overlap) > 0:
    print("\n🚨 DATA LEAKAGE DETECTED!")
    print("Example overlapping files:")
    for h in list(overlap)[:5]:
        print(f"  Train: {train_hashes[h]}")
        print(f"  Test:  {test_hashes[h]}")
else:
    print("\n✅ NO DATA LEAKAGE - Train and Test are completely separate!")

# Check duplicates within train
train_count = len(list((data_dir/"train"/"REAL").glob("*.jpg"))[:5000]) + len(list((data_dir/"train"/"FAKE").glob("*.jpg"))[:5000])
train_dupes = train_count - len(train_hashes)
print(f"\nDuplicates within train: {train_dupes}")
