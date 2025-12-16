"""
run_stage1_fixed.py - Uruchom Stage 1 z naprawionymi danymi (bez data leakage)
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.stage1_baseline import (
    Stage1BaselineDetector,
    Stage1Config,
)

def main():
    print("\n" + "="*60)
    print("🔬 STAGE 1 - FIXED DATA (NO LEAKAGE)")
    print("="*60)
    
    # Config
    config = Stage1Config(
        encoder_name="clip",
        encoder_variant="ViT-B/32",
        k_neighbors=10,
        db_backend="numpy",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    detector = Stage1BaselineDetector(config)
    
    # Paths - FIXED data
    data_root = Path("./data")
    train_real = data_root / "A_standardized_224_fixed/train/real"
    train_fake = data_root / "A_standardized_224_fixed/train/fake"
    test_a_real = data_root / "A_standardized_224_fixed/test_A/real"
    test_a_fake = data_root / "A_standardized_224_fixed/test_A/fake"
    test_b_real = data_root / "B_standardized_224_fixed/test_B/real"
    test_b_fake = data_root / "B_standardized_224_fixed/test_B/fake"
    
    # Build database
    print("\n" + "="*50)
    print("BUILDING DATABASE")
    print("="*50)
    
    start_time = time.time()
    
    train_stats = detector.fit_from_folder(
        real_folder=str(train_real),
        fake_folder=str(train_fake),
        max_images=2000,
        batch_size=32,
    )
    
    build_time = time.time() - start_time
    print(f"\n⏱️ Database built in {build_time:.1f} seconds")
    
    # Evaluate on Test A
    print("\n" + "="*50)
    print("EVALUATION - DATASET A (In-Domain)")
    print("="*50)
    
    results_A = detector.evaluate_from_folder(
        real_folder=str(test_a_real),
        fake_folder=str(test_a_fake),
        max_images=400,
    )
    
    # Evaluate on Test B (cross-domain)
    results_B = None
    if test_b_real.exists() and test_b_fake.exists():
        print("\n" + "="*50)
        print("EVALUATION - DATASET B (Cross-Domain)")
        print("="*50)
        
        results_B = detector.evaluate_from_folder(
            real_folder=str(test_b_real),
            fake_folder=str(test_b_fake),
            max_images=500,
        )
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS (NO DATA LEAKAGE)")
    print("="*60)
    
    print(f"\n📈 Dataset A (In-Domain):")
    for method, metrics in results_A.items():
        print(f"   {method.upper()}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
    
    if results_B:
        print(f"\n📈 Dataset B (Cross-Domain):")
        for method, metrics in results_B.items():
            print(f"   {method.upper()}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
    
    # Analysis
    best_acc_a = max(m['accuracy'] for m in results_A.values())
    
    print("\n" + "="*60)
    print("💡 ANALYSIS")
    print("="*60)
    
    if best_acc_a >= 0.90:
        print("🏆 Excellent! CLIP embeddings are highly effective for this task.")
    elif best_acc_a >= 0.75:
        print("✅ Good results. CLIP provides useful features.")
    elif best_acc_a >= 0.60:
        print("⚠️ Moderate results. Consider Stage 2 or Stage 3.")
    else:
        print("❌ Low accuracy. Need better approach (fine-tuning, different features).")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
