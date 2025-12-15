"""
run_experiments.py - Główny skrypt do uruchamiania eksperymentów badawczych

Uruchamia pełne porównanie wszystkich architektur:
1. Baseline (EfficientNet, ViT)
2. Frequency Branch (EfficientNet + FFT/DCT)  
3. Attention models (CBAM, Artifact Attention)
4. Hybrid (Spatial + Frequency + Attention)
5. Ultimate Detector (pełna architektura)

Użycie:
    python run_experiments.py --experiment all --epochs 20
    python run_experiments.py --experiment baseline --epochs 5 --debug
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Import from our package
import sys
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_research.config import ExperimentConfig, EXPERIMENT_CONFIGS, DataConfig, ModelConfig, TrainingConfig
from deepfake_research.models.factory import create_model, list_models
from deepfake_research.data.datasets import create_dataloaders
from deepfake_research.training.trainer import Trainer
from deepfake_research.training.optimizers import get_optimizer, get_scheduler
from deepfake_research.training.losses import DeepfakeLoss
from deepfake_research.evaluation.benchmark import Benchmark
from deepfake_research.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_cross_dataset_heatmap,
    plot_model_comparison,
    plot_training_history,
)
from deepfake_research.utils.cuda_utils import (
    setup_cuda_optimizations,
    compile_model,
    print_cuda_memory_stats,
)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model_name: str,
    config: ExperimentConfig,
    dataloaders: dict,
    device: str,
    use_compile: bool = False,
) -> tuple:
    """
    Train a single model.
    
    Returns:
        Tuple of (trained_model, training_time, history)
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = create_model(model_name, config.model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Compile model for speedup (PyTorch 2.0+)
    if use_compile and device == "cuda":
        model = compile_model(model, mode="default", verbose=True)
    
    # Optimizer & Scheduler
    optimizer = get_optimizer(
        model,
        optimizer_name="adamw",
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=config.training.scheduler,
        epochs=config.training.epochs,
        warmup_epochs=config.training.warmup_epochs,
    )
    
    # Loss
    criterion = DeepfakeLoss(
        loss_type='smooth' if config.training.label_smoothing > 0 else 'ce',
        label_smoothing=config.training.label_smoothing,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=config.training.epochs,
        use_amp=config.training.use_amp,
        early_stopping=config.training.early_stopping,
        patience=config.training.patience,
        save_dir=config.output_dir / model_name,
        experiment_name=model_name,
        use_wandb=config.log_wandb,
    )
    
    # Train
    import time
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    
    return model, training_time, results['history']


def run_all_experiments(args):
    """Run all experiments and compare results"""
    
    # Setup
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")
    
    # CUDA Optimizations
    if device == "cuda":
        cuda_config = setup_cuda_optimizations(
            use_compile=args.compile,
            use_cudnn_benchmark=True,
            use_tf32=True,
            deterministic=False,
            verbose=True,
        )
        print_cuda_memory_stats()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_sbi=args.use_sbi,
        sbi_probability=0.3,
    )
    
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test A samples: {len(dataloaders['test_A'].dataset)}")
    print(f"Test B samples: {len(dataloaders['test_B'].dataset)}")
    
    # Models to train
    if args.experiment == "all":
        models_to_train = [
            "baseline_efficientnet",
            "baseline_vit",
            "freq_efficientnet",
            "attention_efficientnet",
            "hybrid",
            "ultimate",
        ]
    elif args.experiment == "baseline":
        models_to_train = ["baseline_efficientnet", "baseline_vit"]
    elif args.experiment == "advanced":
        models_to_train = ["freq_efficientnet", "attention_efficientnet", "hybrid"]
    elif args.experiment == "ultimate":
        models_to_train = ["ultimate"]
    else:
        models_to_train = [args.experiment]
    
    # Training config
    base_config = ExperimentConfig(
        data=DataConfig(
            data_root=Path(args.data_root),
            img_size=args.img_size,
            batch_size=args.batch_size,
            use_sbi=args.use_sbi,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            device=device,
            early_stopping=not args.no_early_stopping,
        ),
        output_dir=output_dir,
        log_wandb=args.wandb,
    )
    
    # Train all models
    trained_models = {}
    training_times = {}
    histories = {}
    
    for model_name in models_to_train:
        try:
            model, train_time, history = train_model(
                model_name=model_name,
                config=base_config,
                dataloaders=dataloaders,
                device=device,
                use_compile=args.compile,
            )
            trained_models[model_name] = model
            training_times[model_name] = train_time
            histories[model_name] = history
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            if args.debug:
                raise
            continue
    
    # Benchmark
    print("\n" + "="*60)
    print("BENCHMARKING ALL MODELS")
    print("="*60)
    
    test_loaders = {
        'test_A': dataloaders['test_A'],
        'test_B': dataloaders['test_B'],
    }
    
    benchmark = Benchmark(
        dataloaders=test_loaders,
        device=device,
        output_dir=output_dir / "benchmark",
    )
    
    for model_name, model in trained_models.items():
        benchmark.add_model(
            model=model,
            model_name=model_name,
            training_time=training_times.get(model_name, 0),
        )
    
    # Print comparison
    benchmark.print_comparison()
    
    # Save results
    benchmark.save_results("full_benchmark.json")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Cross-dataset heatmap
    results_for_heatmap = {}
    for model_name, result in benchmark.results.items():
        results_for_heatmap[model_name] = {
            ds: m.accuracy for ds, m in result.metrics_per_dataset.items()
        }
    
    try:
        plot_cross_dataset_heatmap(
            results_for_heatmap,
            save_path=str(output_dir / "cross_dataset_heatmap.png"),
        )
        
        plot_model_comparison(
            results_for_heatmap,
            save_path=str(output_dir / "model_comparison.png"),
        )
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Generate report
    report = benchmark.generate_report()
    report_path = output_dir / "BENCHMARK_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save experiment config
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'models': models_to_train,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'use_sbi': args.use_sbi,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    
    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Research Experiments")
    
    # Experiment selection
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="all",
        choices=["all", "baseline", "advanced", "ultimate"] + list(list_models().keys()),
        help="Which experiment(s) to run",
    )
    
    # Data
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-sbi", action="store_true", help="Use Self-Blended Images")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for 20-40%% speedup (PyTorch 2.0+)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./experiments")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--wandb-project", type=str, default="msc-deepfake-detection", help="W&B project name")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    # Print config
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION RESEARCH")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Use SBI: {args.use_sbi}")
    print(f"Output: {args.output_dir}")
    
    run_all_experiments(args)


if __name__ == "__main__":
    main()
