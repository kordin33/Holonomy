"""
run_stage1_local.py - Uruchom Stage 1 lokalnie na GPU

Zoptymalizowany dla RTX 4060 Ti 8GB (i podobnych kart)

Użycie:
    python run_stage1_local.py --data-root ./data --max-images 2000
    python run_stage1_local.py --encoder ViT-L/14 --batch-size 16
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_research.utils.cuda_utils import (
    setup_cuda_optimizations,
    print_cuda_memory_stats,
    clear_cuda_cache,
)
from deepfake_research.embeddings.stage1_baseline import (
    Stage1BaselineDetector,
    Stage1Config,
)
from deepfake_research.embeddings.visualization import EmbeddingVisualizer


def check_gpu():
    """Sprawdź GPU i wyświetl info"""
    print("\n" + "="*60)
    print("🖥️ GPU CHECK")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA niedostępna! Uruchom na CPU (wolniej).")
        return "cpu", 0
    
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {vram:.1f} GB")
    
    # Rekomendacje dla karty
    if "4060" in gpu_name or "4070" in gpu_name or "3060" in gpu_name:
        print(f"\n🎯 Rekomendacje dla {gpu_name}:")
        print("   Encoder: ViT-B/32 lub ViT-L/14")
        print("   Batch size: 32-64 (ViT-B/32) lub 16-32 (ViT-L/14)")
    
    return "cuda", vram


def run_local_experiment(args):
    """Główna funkcja eksperymentu"""
    
    # Check GPU
    device, vram = check_gpu()
    
    # Setup CUDA optimizations
    if device == "cuda":
        print("\n" + "="*60)
        print("⚡ CUDA OPTIMIZATIONS")
        print("="*60)
        
        setup_cuda_optimizations(
            use_compile=False,  # CLIP nie wymaga kompilacji
            use_cudnn_benchmark=True,
            use_tf32=True,
            verbose=True,
        )
        print_cuda_memory_stats()
    
    # Prepare paths
    data_root = Path(args.data_root)
    train_real = data_root / "A_standardized_224/train/real"
    train_fake = data_root / "A_standardized_224/train/fake"
    test_a_real = data_root / "A_standardized_224/test_A/real"
    test_a_fake = data_root / "A_standardized_224/test_A/fake"
    test_b_real = data_root / "B_standardized_224/test_B/real"
    test_b_fake = data_root / "B_standardized_224/test_B/fake"
    
    # Check data exists
    print("\n" + "="*60)
    print("📁 DATA CHECK")
    print("="*60)
    
    for path, name in [
        (train_real, "Train Real"),
        (train_fake, "Train Fake"),
        (test_a_real, "Test A Real"),
        (test_a_fake, "Test A Fake"),
    ]:
        if path.exists():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"✅ {name}: {count} images")
        else:
            print(f"❌ {name}: NOT FOUND at {path}")
            print("\n⚠️ Najpierw przygotuj dane! Uruchom:")
            print("   python efficientnet_b0_deepfake.py --prepare --data-root ./data")
            return
    
    # Initialize detector
    print("\n" + "="*60)
    print("🚀 STAGE 1: BASELINE DETECTOR")
    print("="*60)
    
    # Auto-adjust batch size based on VRAM
    if args.batch_size is None:
        if "L/14" in args.encoder:
            args.batch_size = 16 if vram < 10 else 32
        else:
            args.batch_size = 32 if vram < 10 else 64
    
    print(f"\nKonfiguracja:")
    print(f"  Encoder: CLIP {args.encoder}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  K neighbors: {args.k}")
    print(f"  Max images: {args.max_images}")
    
    config = Stage1Config(
        encoder_name="clip",
        encoder_variant=args.encoder,
        k_neighbors=args.k,
        db_backend="numpy",
        device=device,
    )
    
    detector = Stage1BaselineDetector(config)
    
    # Build database
    print("\n" + "="*60)
    print("📚 BUILDING VECTOR DATABASE")
    print("="*60)
    
    start_time = time.time()
    
    train_stats = detector.fit_from_folder(
        real_folder=str(train_real),
        fake_folder=str(train_fake),
        max_images=args.max_images,
        batch_size=args.batch_size,
    )
    
    build_time = time.time() - start_time
    print(f"\n⏱️ Database built in {build_time:.1f} seconds")
    print(f"   Speed: {train_stats['total_count'] / build_time:.1f} images/sec")
    
    # Clear cache
    if device == "cuda":
        clear_cuda_cache()
    
    # Evaluate on Test A
    print("\n" + "="*60)
    print("📊 EVALUATION - DATASET A (In-Domain)")
    print("="*60)
    
    results_A = detector.evaluate_from_folder(
        real_folder=str(test_a_real),
        fake_folder=str(test_a_fake),
        max_images=args.max_test,
    )
    
    # Evaluate on Test B (if exists)
    results_B = None
    if test_b_real.exists() and test_b_fake.exists():
        print("\n" + "="*60)
        print("📊 EVALUATION - DATASET B (Cross-Domain)")
        print("="*60)
        
        results_B = detector.evaluate_from_folder(
            real_folder=str(test_b_real),
            fake_folder=str(test_b_fake),
            max_images=args.max_test,
        )
    
    # Visualization
    if args.visualize:
        print("\n" + "="*60)
        print("🎨 VISUALIZATION")
        print("="*60)
        
        embeddings, labels = detector.db.get_all_embeddings()
        visualizer = EmbeddingVisualizer(figsize=(10, 8))
        
        # Output dir
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating t-SNE...")
        visualizer.plot_tsne(
            embeddings=embeddings,
            labels=labels,
            title=f"t-SNE: CLIP {args.encoder} Embeddings",
            save_path=str(output_dir / "stage1_tsne.png"),
        )
        
        print("Generating cluster analysis...")
        cluster_metrics = visualizer.plot_cluster_analysis(
            embeddings=embeddings,
            labels=labels,
            title="Cluster Analysis",
            save_path=str(output_dir / "stage1_clusters.png"),
        )
    else:
        cluster_metrics = {}
    
    # Summary
    print("\n" + "="*60)
    print("📋 FINAL SUMMARY")
    print("="*60)
    
    print(f"\n🔧 Configuration:")
    print(f"   GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"   Encoder: CLIP {args.encoder}")
    print(f"   Database: {detector.db.count()} embeddings")
    print(f"   Build time: {build_time:.1f}s")
    
    print(f"\n📊 Results - Dataset A:")
    for method, metrics in results_A.items():
        print(f"   {method.upper()}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
    
    if results_B:
        print(f"\n📊 Results - Dataset B:")
        for method, metrics in results_B.items():
            print(f"   {method.upper()}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
    
    if cluster_metrics:
        print(f"\n📐 Cluster Metrics:")
        print(f"   Silhouette: {cluster_metrics.get('silhouette_score', 0):.4f}")
        print(f"   Separation Ratio: {cluster_metrics.get('separation_ratio', 0):.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "encoder": args.encoder,
            "k_neighbors": args.k,
            "batch_size": args.batch_size,
            "max_images": args.max_images,
        },
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "build_time_seconds": build_time,
        "database_size": detector.db.count(),
        "results_A": results_A,
        "results_B": results_B,
        "cluster_metrics": cluster_metrics,
    }
    
    results_path = output_dir / "stage1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Save detector
    detector.save(str(output_dir / "detector"))
    
    print("\n" + "="*60)
    print("✅ STAGE 1 COMPLETE!")
    print("="*60)
    
    # Next steps
    best_acc = max(m['accuracy'] for m in results_A.values())
    
    if best_acc < 0.70:
        print("\n💡 Next: Accuracy is low. Try Stage 2 (better classifiers) or Stage 3 (LoRA).")
    elif best_acc < 0.80:
        print("\n💡 Next: Good start! Stage 2 may improve results with SVM/MLP.")
    else:
        print("\n💡 Great results! You can proceed to Stage 3 for even better performance.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: CLIP Embedding-based Deepfake Detection (Local)"
    )
    
    # Data
    parser.add_argument("--data-root", type=str, default="./data",
                       help="Root folder with data")
    parser.add_argument("--max-images", type=int, default=2000,
                       help="Max images per class for training database")
    parser.add_argument("--max-test", type=int, default=500,
                       help="Max images per class for testing")
    
    # Model
    parser.add_argument("--encoder", type=str, default="ViT-B/32",
                       choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                       help="CLIP encoder variant")
    parser.add_argument("--k", type=int, default=10,
                       help="K for k-NN classification")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (auto-detected if not set)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./stage1_results",
                       help="Output directory")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations (t-SNE, clusters)")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.set_defaults(visualize=True)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔬 STAGE 1: EMBEDDING-BASED DEEPFAKE DETECTION")
    print("="*60)
    
    run_local_experiment(args)


if __name__ == "__main__":
    main()
