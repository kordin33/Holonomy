"""
prepare_data_local.py - Przygotuj dane lokalnie z obsługą rate limiting

Pobiera i przetwarza datasety deepfake z HuggingFace.
Z retry i delays żeby obejść 429 errors.
"""

import os
import sys
import zipfile
import shutil
import random
import time
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    print("Instaluję datasets...")
    os.system("pip install datasets")
    from datasets import load_dataset


# ============================================
# Rate Limit Handler
# ============================================

def download_with_retry(download_func, max_retries=5, initial_delay=5):
    """
    Wrapper z retry i exponential backoff dla 429 errors.
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return download_func()
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                print(f"\n⚠️ Rate limited (429). Czekam {delay}s przed ponowną próbą... (próba {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")


def extract_zip_with_structure(zip_path: Path, extract_to: Path):
    """Rozpakuj ZIP i znajdź obrazy"""
    print(f"Rozpakowuję {zip_path.name}...")
    
    temp_dir = extract_to / f"_temp_{zip_path.stem}"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Znajdź wszystkie obrazy
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        images.extend(temp_dir.rglob(ext))
    
    print(f"  Znaleziono {len(images)} obrazów")
    return images, temp_dir


def prepare_dataset_a_from_zips(
    data_root: Path,
    output_dir: Path,
    max_per_class: int = 2000,
    img_size: int = 224,
):
    """
    Przygotuj Dataset A z pobranych plików ZIP (FAKE) 
    i pobierz REAL z datasets z obsługą rate limiting.
    """
    deepfake_face_dir = data_root / "DeepFakeFace"
    
    if not deepfake_face_dir.exists():
        print(f"❌ Folder {deepfake_face_dir} nie istnieje!")
        return None
    
    # Znajdź pliki ZIP (FAKE images)
    zip_files = list(deepfake_face_dir.glob("*.zip"))
    print(f"\nZnaleziono {len(zip_files)} plików ZIP (FAKE images):")
    for z in zip_files:
        print(f"  - {z.name} ({z.stat().st_size / 1e9:.2f} GB)")
    
    if not zip_files:
        print("❌ Brak plików ZIP!")
        return None
    
    # Zbierz FAKE images z ZIP-ów
    all_fake_images = []
    temp_dirs = []
    
    for zip_path in zip_files:
        images, temp_dir = extract_zip_with_structure(zip_path, deepfake_face_dir)
        all_fake_images.extend(images)
        temp_dirs.append(temp_dir)
    
    print(f"\n📊 Łącznie FAKE obrazów: {len(all_fake_images)}")
    
    # Pobierz REAL images z datasets z retry
    print("\n" + "="*50)
    print("Pobieram REAL images z HuggingFace (z retry dla 429)...")
    print("="*50)
    
    all_real_images = download_real_faces_with_retry(
        data_root=data_root,
        max_count=max_per_class,
        img_size=img_size,
    )
    
    print(f"\n📊 Łącznie REAL obrazów: {len(all_real_images)}")
    
    if len(all_real_images) == 0:
        print("❌ Nie udało się pobrać REAL obrazów!")
        print("   Spróbuj później lub użyj własnych REAL obrazów.")
        return None
    
    # Ogranicz liczbę
    random.shuffle(all_fake_images)
    random.shuffle(all_real_images)
    
    all_fake_images = all_fake_images[:max_per_class]
    all_real_images = all_real_images[:max_per_class]
    
    # Podział 70/15/15
    def split_data(images):
        n = len(images)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        return {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test_A': images[val_end:],
        }
    
    fake_splits = split_data(all_fake_images)
    real_splits = split_data(all_real_images)
    
    # Twórz strukturę folderów
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test_A']:
        for label in ['real', 'fake']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Kopiuj i resize
    def copy_and_resize(images, dst_dir, label, split, img_size):
        dst = dst_dir / split / label
        print(f"  Przetwarzam {len(images)} obrazów do {split}/{label}...")
        
        count = 0
        for i, img_path in enumerate(tqdm(images, desc=f"{split}/{label}")):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size), Image.LANCZOS)
                img.save(dst / f"{label}_{i:05d}.jpg", "JPEG", quality=95)
                count += 1
            except Exception as e:
                continue
        return count
    
    stats = {}
    for split in ['train', 'val', 'test_A']:
        stats[f"{split}_real"] = copy_and_resize(real_splits[split], output_dir, "real", split, img_size)
        stats[f"{split}_fake"] = copy_and_resize(fake_splits[split], output_dir, "fake", split, img_size)
    
    # Cleanup temp dirs
    print("\n🧹 Czyszczenie plików tymczasowych...")
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return stats


def download_real_faces_with_retry(
    data_root: Path, 
    max_count: int = 2000,
    img_size: int = 224,
    batch_size: int = 100,
    delay_between_batches: float = 2.0,
):
    """
    Pobierz prawdziwe twarze z obsługą rate limiting.
    Pobiera w małych batchach z opóźnieniami.
    """
    real_dir = data_root / "real_faces_temp"
    real_dir.mkdir(parents=True, exist_ok=True)
    
    # Sprawdź czy już mamy pobrane
    existing = list(real_dir.glob("*.jpg"))
    if len(existing) >= max_count:
        print(f"  ✓ Już mamy {len(existing)} REAL obrazów")
        return existing[:max_count]
    
    print(f"  Pobieram do {real_dir}...")
    print(f"  Batch size: {batch_size}, delay: {delay_between_batches}s")
    
    try:
        # Streaming mode - nie ładuje wszystkiego do pamięci
        # i lepiej radzi sobie z rate limiting
        ds = load_dataset(
            "prithivMLmods/Deepfake-vs-Real-v2",
            split="train",
            streaming=True,  # Streaming mode!
        )
        
        images = list(existing)  # Start with existing
        count = len(existing)
        batch_count = 0
        
        pbar = tqdm(total=max_count, initial=count, desc="Pobieranie REAL")
        
        for item in ds:
            if count >= max_count:
                break
            
            # label==1 to REAL
            if item.get("label", 0) == 1:
                try:
                    img = item["image"]
                    if img is not None:
                        img = img.convert("RGB")
                        img = img.resize((img_size, img_size), Image.LANCZOS)
                        
                        path = real_dir / f"real_{count:05d}.jpg"
                        img.save(path, "JPEG", quality=95)
                        images.append(path)
                        count += 1
                        batch_count += 1
                        pbar.update(1)
                        
                        # Delay every batch to avoid 429
                        if batch_count >= batch_size:
                            pbar.set_description(f"REAL (pause {delay_between_batches}s)")
                            time.sleep(delay_between_batches)
                            batch_count = 0
                            pbar.set_description("Pobieranie REAL")
                            
                except Exception as e:
                    if "429" in str(e):
                        print(f"\n⚠️ 429 error. Czekam 30s...")
                        time.sleep(30)
                    continue
        
        pbar.close()
        return images
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        # Zwróć co mamy
        return list(real_dir.glob("*.jpg"))


def prepare_dataset_b_with_retry(
    data_root: Path,
    output_dir: Path,
    max_per_class: int = 1000,
    img_size: int = 224,
    batch_size: int = 50,
    delay_between_batches: float = 2.0,
):
    """
    Przygotuj Dataset B z obsługą rate limiting.
    """
    print("\n" + "="*50)
    print("Przygotowuję Dataset B (z retry dla 429)...")
    print("="*50)
    
    output_b = output_dir / "test_B"
    (output_b / "real").mkdir(parents=True, exist_ok=True)
    (output_b / "fake").mkdir(parents=True, exist_ok=True)
    
    # Sprawdź co już mamy
    existing_real = len(list((output_b / "real").glob("*.jpg")))
    existing_fake = len(list((output_b / "fake").glob("*.jpg")))
    
    if existing_real >= max_per_class and existing_fake >= max_per_class:
        print(f"  ✓ Już mamy wystarczająco: real={existing_real}, fake={existing_fake}")
        return {"test_B_real": existing_real, "test_B_fake": existing_fake}
    
    try:
        ds = load_dataset(
            "prithivMLmods/Deepfake-vs-Real-v2",
            split="train",
            streaming=True,
        )
        
        real_count = existing_real
        fake_count = existing_fake
        batch_count = 0
        
        pbar = tqdm(total=max_per_class*2, initial=real_count+fake_count, desc="Dataset B")
        
        for item in ds:
            if real_count >= max_per_class and fake_count >= max_per_class:
                break
            
            label = item.get("label", 0)
            
            if label == 1 and real_count < max_per_class:
                label_str = "real"
                idx = real_count
                real_count += 1
            elif label == 0 and fake_count < max_per_class:
                label_str = "fake"
                idx = fake_count
                fake_count += 1
            else:
                continue
            
            try:
                img = item["image"].convert("RGB")
                img = img.resize((img_size, img_size), Image.LANCZOS)
                img.save(output_b / label_str / f"{label_str}_{idx:05d}.jpg", "JPEG", quality=95)
                pbar.update(1)
                batch_count += 1
                
                if batch_count >= batch_size:
                    pbar.set_description(f"B (pause {delay_between_batches}s)")
                    time.sleep(delay_between_batches)
                    batch_count = 0
                    pbar.set_description("Dataset B")
                    
            except Exception as e:
                if "429" in str(e):
                    print(f"\n⚠️ 429 error. Czekam 30s...")
                    time.sleep(30)
                continue
        
        pbar.close()
        
        print(f"  test_B/real: {real_count}")
        print(f"  test_B/fake: {fake_count}")
        
        return {"test_B_real": real_count, "test_B_fake": fake_count}
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Przygotuj dane do deepfake detection")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--max-per-class-a", type=int, default=2000)
    parser.add_argument("--max-per-class-b", type=int, default=1000)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=100, help="Images per batch before delay")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between batches (seconds)")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    print("="*60)
    print("🔧 PRZYGOTOWANIE DANYCH DO DEEPFAKE DETECTION")
    print("   (z obsługą rate limiting)")
    print("="*60)
    
    # Dataset A
    print("\n" + "="*50)
    print("Przygotowuję Dataset A...")
    print("="*50)
    
    output_a = data_root / "A_standardized_224"
    stats_a = prepare_dataset_a_from_zips(
        data_root=data_root,
        output_dir=output_a,
        max_per_class=args.max_per_class_a,
        img_size=args.img_size,
    )
    
    # Dataset B
    output_b = data_root / "B_standardized_224"
    stats_b = prepare_dataset_b_with_retry(
        data_root=data_root,
        output_dir=output_b,
        max_per_class=args.max_per_class_b,
        img_size=args.img_size,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
    )
    
    print("\n" + "="*60)
    print("✅ DANE PRZYGOTOWANE!")
    print("="*60)
    
    # Podsumowanie
    print("\nStruktura danych:")
    for split in ['train', 'val', 'test_A']:
        for label in ['real', 'fake']:
            path = output_a / split / label
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                print(f"  A/{split}/{label}: {count}")
    
    test_b_path = output_b / "test_B"
    if test_b_path.exists():
        for label in ['real', 'fake']:
            path = test_b_path / label
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                print(f"  B/test_B/{label}: {count}")
    
    print("\n💡 Gotowe! Uruchom Stage 1:")
    print("   python run_stage1_local.py")


if __name__ == "__main__":
    main()
