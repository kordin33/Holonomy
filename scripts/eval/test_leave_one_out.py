"""
test_leave_one_out.py - Właściwy eksperyment Leave-One-Out Cross-Domain

Dla każdej metody generowania:
1. Trenuj na TEJ metodzie
2. Testuj na WSZYSTKICH metodach

To daje pełną macierz generalizacji!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import json
from datetime import datetime
import random

from deepfake_guard.embeddings.stage1_baseline import (
    Stage1BaselineDetector,
    Stage1Config,
)


def prepare_method_data(method_folder: Path, max_per_class: int = 1000):
    """Przygotuj dane z jednej metody (fake) + real z głównego datasetu"""
    fake_paths = list(method_folder.rglob("*.jpg")) + list(method_folder.rglob("*.png"))
    random.shuffle(fake_paths)
    return fake_paths[:max_per_class]


def test_detector(detector, test_paths: list, expected_label: str):
    """Test detector na obrazach - zwróć accuracy"""
    correct = 0
    total = 0
    
    for path in test_paths:
        try:
            img = Image.open(path).convert("RGB")
            result = detector.predict(img, method="knn")
            
            if result.prediction == expected_label:
                correct += 1
            total += 1
        except:
            continue
    
    return correct / total if total > 0 else 0


def run_leave_one_out_experiment():
    """Główny eksperyment Leave-One-Out"""
    
    print("="*70)
    print("🔬 LEAVE-ONE-OUT CROSS-DOMAIN EXPERIMENT")
    print("="*70)
    
    data_root = Path("./data")
    deepfake_face = data_root / "DeepFakeFace"
    
    # Metody generowania (wszystkie są FAKE)
    methods = {
        "inpainting": deepfake_face / "_temp_inpainting",
        "insight": deepfake_face / "_temp_insight",
        "text2img": deepfake_face / "_temp_text2img",
        "wiki": deepfake_face / "_temp_wiki",
    }
    
    # Sprawdź które istnieją
    available_methods = {k: v for k, v in methods.items() if v.exists()}
    print(f"\nAvailable methods: {list(available_methods.keys())}")
    
    if len(available_methods) < 2:
        print("❌ Need at least 2 methods for cross-domain testing!")
        return
    
    # Real images z głównego datasetu
    real_source = data_root / "deepfake_vs_real/Real"
    if not real_source.exists():
        print("❌ Real images not found!")
        return
    
    real_paths = list(real_source.glob("*.jpg")) + list(real_source.glob("*.png"))
    random.seed(42)
    random.shuffle(real_paths)
    
    # Podział real: 70% train, 30% test
    real_train = real_paths[:2000]
    real_test = real_paths[2000:3000]
    
    print(f"Real images: {len(real_train)} train, {len(real_test)} test")
    
    # Macierz wyników: [train_method][test_method]
    results_matrix = {}
    
    # Dla każdej metody jako TRAIN
    for train_method, train_folder in available_methods.items():
        print(f"\n{'='*60}")
        print(f"TRAINING on: {train_method}")
        print(f"{'='*60}")
        
        # Przygotuj train fake
        fake_train = prepare_method_data(train_folder, max_per_class=2000)
        print(f"  Train: {len(real_train)} real, {len(fake_train)} fake")
        
        # Utwórz nowy detektor
        config = Stage1Config(
            encoder_variant="ViT-B/32",
            k_neighbors=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        detector = Stage1BaselineDetector(config)
        
        # Buduj bazę wektorową
        # Encode real
        print("  Encoding real...")
        real_images = []
        for p in tqdm(real_train[:2000], desc="Loading real"):
            try:
                real_images.append(Image.open(p).convert("RGB"))
            except:
                continue
        
        real_embeddings = detector.encoder.encode_batch(real_images, show_progress=True)
        real_labels = ["real"] * len(real_embeddings)
        
        # Encode fake
        print("  Encoding fake...")
        fake_images = []
        for p in tqdm(fake_train[:2000], desc="Loading fake"):
            try:
                fake_images.append(Image.open(p).convert("RGB"))
            except:
                continue
        
        fake_embeddings = detector.encoder.encode_batch(fake_images, show_progress=True)
        fake_labels = ["fake"] * len(fake_embeddings)
        
        # Add to database
        detector.db.add(real_embeddings, real_labels)
        detector.db.add(fake_embeddings, fake_labels)
        detector.is_fitted = True
        
        print(f"  Database: {detector.db.count()} embeddings")
        
        # Testuj na KAŻDEJ metodzie
        results_matrix[train_method] = {}
        
        # Test na real
        real_test_sample = real_test[:500]
        real_acc = test_detector(detector, real_test_sample, "real")
        results_matrix[train_method]["real"] = real_acc
        print(f"\n  Test REAL: {real_acc:.2%} accuracy")
        
        # Test na każdej fake method
        for test_method, test_folder in available_methods.items():
            fake_test = prepare_method_data(test_folder, max_per_class=500)
            
            fake_acc = test_detector(detector, fake_test, "fake")
            results_matrix[train_method][test_method] = fake_acc
            
            is_same = "✓" if test_method == train_method else "→"
            print(f"  Test {test_method}: {fake_acc:.2%} {is_same}")
    
    # Podsumowanie
    print("\n" + "="*70)
    print("📊 RESULTS MATRIX (Fake Detection Rate)")
    print("="*70)
    
    # Header
    header = ["Train\\Test"] + list(available_methods.keys()) + ["real"]
    print("\n{:<15}".format(header[0]) + "".join(["{:<12}".format(h) for h in header[1:]]))
    print("-"*70)
    
    # Rows
    for train_method in available_methods.keys():
        row = [train_method]
        for test_method in list(available_methods.keys()) + ["real"]:
            acc = results_matrix[train_method].get(test_method, 0)
            row.append(f"{acc:.1%}")
        print("{:<15}".format(row[0]) + "".join(["{:<12}".format(r) for r in row[1:]]))
    
    # Analiza
    print("\n" + "="*70)
    print("📈 CROSS-DOMAIN GENERALIZATION ANALYSIS")
    print("="*70)
    
    # Same-domain accuracy (diagonal)
    same_domain = []
    cross_domain = []
    
    for train_method in available_methods.keys():
        for test_method in available_methods.keys():
            acc = results_matrix[train_method].get(test_method, 0)
            if train_method == test_method:
                same_domain.append(acc)
            else:
                cross_domain.append(acc)
    
    avg_same = np.mean(same_domain) if same_domain else 0
    avg_cross = np.mean(cross_domain) if cross_domain else 0
    generalization_gap = avg_same - avg_cross
    
    print(f"\nSame-domain accuracy (diagonal): {avg_same:.2%}")
    print(f"Cross-domain accuracy (off-diagonal): {avg_cross:.2%}")
    print(f"Generalization gap: {generalization_gap:.2%}")
    
    # Real detection
    real_accs = [results_matrix[m].get("real", 0) for m in available_methods.keys()]
    avg_real = np.mean(real_accs)
    print(f"Average real detection: {avg_real:.2%}")
    
    # Overall balanced accuracy
    balanced_acc = (avg_cross + avg_real) / 2
    print(f"\nBalanced cross-domain accuracy: {balanced_acc:.2%}")
    
    # Interpretation
    print("\n" + "="*70)
    print("💡 INTERPRETATION")
    print("="*70)
    
    if avg_cross >= 0.85:
        print("🏆 EXCELLENT! CLIP generalizes very well across generation methods!")
        print("   This is publication-worthy baseline performance.")
    elif avg_cross >= 0.70:
        print("✅ GOOD! Decent generalization. Stage 2/3 improvements recommended.")
        print("   - Try SVM/MLP classifiers on embeddings")
        print("   - Consider LoRA fine-tuning")
    elif avg_cross >= 0.50:
        print("⚠️ MODERATE. Significant generalization gap.")
        print("   - Fine-tuning is necessary")
        print("   - Consider frequency domain features")
    else:
        print("❌ POOR. Model overfits to generation method.")
        print("   - Need fundamentally different approach")
    
    # Save
    final_results = {
        "results_matrix": results_matrix,
        "summary": {
            "same_domain_accuracy": avg_same,
            "cross_domain_accuracy": avg_cross,
            "generalization_gap": generalization_gap,
            "real_detection_accuracy": avg_real,
            "balanced_accuracy": balanced_acc,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open("leave_one_out_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print("\n✓ Results saved to leave_one_out_results.json")
    
    return final_results


if __name__ == "__main__":
    random.seed(42)
    run_leave_one_out_experiment()
