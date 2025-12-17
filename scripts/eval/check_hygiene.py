"""
check_hygiene.py - Sprawdzenie higieny: leakage + CV stability

1. Sprawdzenie leakage w CIFAKE i GenImage
2. Test CV z StratifiedKFold + random_state
3. Sprawdzenie powtarzalności wyników
"""

import sys
from pathlib import Path
import numpy as np
import hashlib
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parents[2]))

RANDOM_STATE = 42


def compute_image_hash(image_path: Path) -> str:
    """Oblicza hash pliku obrazu."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def check_cifake_leakage():
    """Sprawdza leakage w CIFAKE (train vs test)."""
    print("\n" + "="*70)
    print("🔍 SPRAWDZANIE LEAKAGE: CIFAKE")
    print("="*70)
    
    cifake_dir = Path("./data/cifake")
    
    if not cifake_dir.exists():
        print("  ⚠️ CIFAKE nie znaleziony!")
        return
    
    # Collect hashes
    train_hashes = {}
    test_hashes = {}
    
    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            dir_path = cifake_dir / split / cls
            if not dir_path.exists():
                continue
            
            files = list(dir_path.glob("*.jpg"))[:500]  # sample
            for f in tqdm(files, desc=f"{split}/{cls}", leave=False):
                h = compute_image_hash(f)
                if split == "train":
                    train_hashes[h] = f
                else:
                    test_hashes[h] = f
    
    # Check overlap
    overlap = set(train_hashes.keys()) & set(test_hashes.keys())
    
    print(f"\n  Train samples checked: {len(train_hashes)}")
    print(f"  Test samples checked: {len(test_hashes)}")
    print(f"  Overlapping (leakage): {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"  ❌ LEAKAGE DETECTED! {len(overlap)} identical images!")
        for h in list(overlap)[:5]:
            print(f"     - Train: {train_hashes[h]}")
            print(f"       Test:  {test_hashes[h]}")
    else:
        print(f"  ✅ No leakage detected!")
    
    return len(overlap)


def check_genimage_leakage():
    """Sprawdza leakage w GenImage (real_pool vs fake pools)."""
    print("\n" + "="*70)
    print("🔍 SPRAWDZANIE LEAKAGE: GENIMAGE")
    print("="*70)
    
    genimage_dir = Path("./data/genimage")
    
    if not genimage_dir.exists():
        print("  ⚠️ GenImage nie znaleziony!")
        return
    
    pools = ["real_pool", "sd_pool", "mj_pool", "gan_pool"]
    pool_hashes = {}
    
    for pool in pools:
        pool_dir = genimage_dir / pool
        if not pool_dir.exists():
            print(f"  ⚠️ {pool} nie znaleziony!")
            continue
        
        files = list(pool_dir.glob("**/*.JPEG"))[:300]
        files += list(pool_dir.glob("**/*.jpg"))[:300]
        files += list(pool_dir.glob("**/*.png"))[:300]
        files = files[:500]  # limit
        
        pool_hashes[pool] = {}
        for f in tqdm(files, desc=pool, leave=False):
            h = compute_image_hash(f)
            pool_hashes[pool][h] = f
        
        print(f"  {pool}: {len(pool_hashes[pool])} samples")
    
    # Check overlaps between pools
    total_leakage = 0
    for i, pool1 in enumerate(pools):
        if pool1 not in pool_hashes:
            continue
        for pool2 in pools[i+1:]:
            if pool2 not in pool_hashes:
                continue
            overlap = set(pool_hashes[pool1].keys()) & set(pool_hashes[pool2].keys())
            if len(overlap) > 0:
                print(f"  ❌ LEAKAGE: {pool1} ↔ {pool2}: {len(overlap)} images!")
                total_leakage += len(overlap)
    
    if total_leakage == 0:
        print(f"  ✅ No leakage between pools!")
    
    return total_leakage


def check_cv_stability():
    """Sprawdza stabilność CV z StratifiedKFold."""
    print("\n" + "="*70)
    print("🔍 SPRAWDZANIE CV STABILITY")
    print("="*70)
    
    # Generate dummy data
    np.random.seed(42)
    X = np.random.randn(200, 36)
    y = np.array([1]*100 + [0]*100)
    
    # Test 1: Z random_state (powinno być identyczne)
    print("\n  Test 1: CV z random_state=42")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    scores1 = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    scores2 = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    scores3 = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    
    print(f"    Run 1: {scores1.mean():.4f}")
    print(f"    Run 2: {scores2.mean():.4f}")
    print(f"    Run 3: {scores3.mean():.4f}")
    
    if np.allclose(scores1, scores2) and np.allclose(scores2, scores3):
        print(f"    ✅ Wyniki identyczne (deterministyczne)")
    else:
        print(f"    ❌ Wyniki różne! CV nie jest deterministyczne")
    
    # Test 2: Bez random_state (powinno być różne)
    print("\n  Test 2: CV bez random_state")
    
    cv_no_seed = StratifiedKFold(n_splits=5, shuffle=True)  # brak seed!
    
    scores_a = cross_val_score(pipe, X, y, cv=cv_no_seed, scoring='roc_auc')
    scores_b = cross_val_score(pipe, X, y, cv=cv_no_seed, scoring='roc_auc')
    
    print(f"    Run A: {scores_a.mean():.4f}")
    print(f"    Run B: {scores_b.mean():.4f}")
    
    if np.allclose(scores_a, scores_b):
        print(f"    ⚠️ Wyniki identyczne mimo braku seed")
    else:
        print(f"    ℹ️ Wyniki różne (expected bez seed)")


def check_current_tests_cv():
    """Sprawdza czy nasze testy używają poprawnego CV."""
    print("\n" + "="*70)
    print("🔍 SPRAWDZANIE CV W NASZYCH TESTACH")
    print("="*70)
    
    test_files = list(Path("./scripts/eval").glob("test_*.py"))
    
    issues = []
    
    for test_file in test_files:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for GridSearchCV
        if "GridSearchCV" in content:
            # Check if cv is StratifiedKFold with random_state
            if "StratifiedKFold" in content and "random_state" in content:
                status = "✅"
            elif "cv=5" in content or "cv=3" in content:
                # Using integer cv (default StratifiedKFold, no shuffle)
                status = "⚠️ używa cv=int (OK ale bez shuffle)"
                issues.append((test_file.name, "cv=int zamiast StratifiedKFold"))
            else:
                status = "?"
        else:
            status = "ℹ️ bez GridSearchCV"
        
        print(f"  {test_file.name}: {status}")
    
    if issues:
        print(f"\n  ⚠️ Znaleziono {len(issues)} potencjalnych problemów:")
        for name, issue in issues:
            print(f"     - {name}: {issue}")
    else:
        print(f"\n  ✅ Wszystkie testy OK!")


def main():
    print("="*70)
    print("🧹 SPRAWDZANIE HIGIENY PROJEKTU")
    print("="*70)
    
    # 1. Leakage
    cifake_leak = check_cifake_leakage()
    genimage_leak = check_genimage_leakage()
    
    # 2. CV stability
    check_cv_stability()
    
    # 3. Check our tests
    check_current_tests_cv()
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE HIGIENY")
    print("="*70)
    
    if cifake_leak == 0 and (genimage_leak is None or genimage_leak == 0):
        print("  ✅ Brak leakage")
    else:
        print("  ❌ WYKRYTO LEAKAGE!")
    
    print("\n  Zalecenia:")
    print("  1. Używaj StratifiedKFold(random_state=42)")
    print("  2. Używaj train_test_split(random_state=42)")
    print("  3. Sprawdzaj leakage przy każdym nowym datasecie")
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
