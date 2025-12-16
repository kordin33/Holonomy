"""
download_dataset.py - Pobierz cały dataset Deepfake-vs-Real-v2

Pobiera za pomocą huggingface_hub.snapshot_download który jest
szybszy i niezawodny.
"""

import os
from pathlib import Path

print("="*60)
print("Pobieranie datasetu Deepfake-vs-Real-v2")
print("="*60)

from huggingface_hub import snapshot_download

# Pobierz cały dataset
local_dir = Path("./data/deepfake_vs_real")
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Pobieranie do: {local_dir}")
print("To może potrwać kilka minut...")

path = snapshot_download(
    repo_id="prithivMLmods/Deepfake-vs-Real-v2",
    repo_type="dataset",
    local_dir=str(local_dir),
    resume_download=True,  # Wznów jeśli przerwane
)

print(f"\n✅ Pobrano do: {path}")

# Sprawdź co mamy
print("\nStruktura:")
for item in local_dir.iterdir():
    if item.is_dir():
        count = len(list(item.rglob("*")))
        print(f"  {item.name}/: {count} plików")
    else:
        size = item.stat().st_size / 1e6
        print(f"  {item.name}: {size:.1f} MB")
