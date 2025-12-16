"""
test_cross_methods.py - Test generalizacji na różnych metodach generowania

Trenuje na jednym datasecie, testuje na 4 różnych metodach generowania deepfake.
To jest PRAWDZIWY test cross-domain!
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

from deepfake_guard.embeddings.stage1_baseline import (
    Stage1BaselineDetector,
    Stage1Config,
)


def prepare_test_set_from_folder(folder_path: Path, max_images: int = 500):
    """Przygotuj obrazy testowe z folderu (wszystkie są FAKE)"""
    images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
    images = images[:max_images]
    return images


def test_on_method(detector, method_folder: Path, max_images: int = 500):
    """Test detektora na jednej metodzie generowania"""
    
    # Wszystkie obrazy w tym folderze są FAKE
    fake_paths = list(method_folder.rglob("*.jpg")) + list(method_folder.rglob("*.png"))
    fake_paths = fake_paths[:max_images]
    
    if len(fake_paths) == 0:
        return None
    
    print(f"\nTesting on {method_folder.name}: {len(fake_paths)} images")
    
    # Predict
    correct = 0
    total = 0
    
    for path in tqdm(fake_paths, desc=method_folder.name):
        try:
            img = Image.open(path).convert("RGB")
            result = detector.predict(img, method="knn")
            
            # Wszystkie powinny być "fake"
            if result.prediction == "fake":
                correct += 1
            total += 1
            
        except Exception as e:
            continue
    
    if total == 0:
        return None
    
    # Fake detection rate (TPR for fake class)
    tpr = correct / total
    
    return {
        "method": method_folder.name,
        "total_tested": total,
        "correctly_detected_fake": correct,
        "true_positive_rate": tpr,
        "false_negative_rate": 1 - tpr,
    }


def main():
    print("="*60)
    print("🔬 CROSS-METHOD GENERALIZATION TEST")
    print("="*60)
    
    # Load detector trained on Deepfake-vs-Real-v2
    print("\nLoading detector trained on Deepfake-vs-Real-v2...")
    
    config = Stage1Config(
        encoder_variant="ViT-B/32",
        k_neighbors=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    detector = Stage1BaselineDetector(config)
    
    # Fit on our dataset
    data_root = Path("./data")
    train_real = data_root / "A_standardized_224_fixed/train/real"
    train_fake = data_root / "A_standardized_224_fixed/train/fake"
    
    if not train_real.exists():
        print("❌ Training data not found! Run prepare_from_downloaded_fixed.py first.")
        return
    
    print("\nBuilding database from Deepfake-vs-Real-v2...")
    detector.fit_from_folder(
        real_folder=str(train_real),
        fake_folder=str(train_fake),
        max_images=2000,
    )
    
    # Test on each DeepFakeFace method
    deepfake_face = data_root / "DeepFakeFace"
    
    methods = [
        deepfake_face / "_temp_inpainting",
        deepfake_face / "_temp_insight", 
        deepfake_face / "_temp_text2img",
        deepfake_face / "_temp_wiki",
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("CROSS-METHOD TESTING")
    print("="*60)
    
    for method_folder in methods:
        if method_folder.exists():
            result = test_on_method(detector, method_folder, max_images=500)
            if result:
                results[method_folder.name] = result
                print(f"\n{method_folder.name}:")
                print(f"  True Positive Rate (Fake correctly detected): {result['true_positive_rate']:.2%}")
                print(f"  False Negative Rate (Fake missed): {result['false_negative_rate']:.2%}")
    
    # Also test on real images from our dataset to check FPR
    print("\n" + "="*50)
    print("Testing on REAL images (checking FPR)...")
    print("="*50)
    
    test_real = data_root / "A_standardized_224_fixed/test_A/real"
    if test_real.exists():
        real_paths = list(test_real.glob("*.jpg"))[:500]
        
        correct = 0
        total = 0
        
        for path in tqdm(real_paths, desc="Testing real"):
            try:
                img = Image.open(path).convert("RGB")
                result = detector.predict(img, method="knn")
                
                if result.prediction == "real":
                    correct += 1
                total += 1
            except:
                continue
        
        tnr = correct / total if total > 0 else 0
        fpr = 1 - tnr
        
        results["real_images_test"] = {
            "total_tested": total,
            "correctly_detected_real": correct,
            "true_negative_rate": tnr,
            "false_positive_rate": fpr,
        }
        
        print(f"\nReal images:")
        print(f"  True Negative Rate (Real correctly detected): {tnr:.2%}")
        print(f"  False Positive Rate (Real misclassified): {fpr:.2%}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    print("\n{:<20} {:<15} {:<15}".format("Method", "Fake TPR", "Count"))
    print("-"*50)
    
    for name, r in results.items():
        if "true_positive_rate" in r:
            print("{:<20} {:<15.2%} {:<15}".format(
                name.replace("_temp_", ""), 
                r["true_positive_rate"],
                r["total_tested"]
            ))
    
    if "real_images_test" in results:
        r = results["real_images_test"]
        print("{:<20} {:<15.2%} {:<15}".format(
            "real (TNR)", 
            r["true_negative_rate"],
            r["total_tested"]
        ))
    
    # Calculate overall
    all_tpr = [r["true_positive_rate"] for r in results.values() if "true_positive_rate" in r]
    avg_tpr = np.mean(all_tpr) if all_tpr else 0
    
    print("\n" + "-"*50)
    print(f"Average Fake Detection (TPR): {avg_tpr:.2%}")
    
    if "real_images_test" in results:
        tnr = results["real_images_test"]["true_negative_rate"]
        balanced_acc = (avg_tpr + tnr) / 2
        print(f"Real Detection (TNR): {tnr:.2%}")
        print(f"Balanced Accuracy: {balanced_acc:.2%}")
    
    # Interpretation
    print("\n" + "="*60)
    print("💡 INTERPRETATION")
    print("="*60)
    
    if avg_tpr >= 0.85:
        print("🏆 EXCELLENT! Model generalizes well across different generation methods!")
    elif avg_tpr >= 0.70:
        print("✅ GOOD! Model shows decent generalization. Stage 2/3 can improve.")
    elif avg_tpr >= 0.50:
        print("⚠️ MODERATE. Significant room for improvement with fine-tuning.")
    else:
        print("❌ POOR. Need different approach or fine-tuning.")
    
    # Save results
    results["summary"] = {
        "average_tpr": avg_tpr,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open("cross_method_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Results saved to cross_method_results.json")


if __name__ == "__main__":
    main()
