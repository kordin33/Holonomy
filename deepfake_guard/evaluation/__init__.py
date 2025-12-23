"""
Evaluation Module - Metrics, Benchmarking & Visualization
==========================================================

Comprehensive evaluation toolkit for deepfake detection models.

COMPONENTS:
-----------

1. metrics.py - Evaluation Metrics
   - EvaluationMetrics: Dataclass for all metrics
   - compute_metrics(): Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, EER
   - MetricsComputer: Model evaluation wrapper
   - Cross-dataset evaluation utilities

2. benchmark.py - Benchmarking Framework
   - Benchmark: Multi-model comparison
   - BenchmarkResult: Per-model results container
   - Automatic cross-dataset testing
   - JSON/Markdown report generation

3. visualization.py - Result Visualization
   - ROC curves, PR curves
   - Confusion matrices (heatmap)
   - Training history plots
   - Cross-dataset comparison heatmaps

USAGE:
------
    from deepfake_guard.evaluation import (
        compute_metrics,
        Benchmark,
        plot_roc_curve,
        plot_confusion_matrix
    )
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"AUC-ROC: {metrics.auc_roc:.4f}")
    print(f"EER: {metrics.eer:.4f}")
    
    # Benchmark multiple models
    benchmark = Benchmark(dataloaders, device="cuda")
    benchmark.add_model(model1, "HolonomyV18")
    benchmark.add_model(model2, "Baseline")
    benchmark.print_comparison()
    benchmark.save_results("benchmark_results.json")
    
    # Generate report
    report = benchmark.generate_report()  # Markdown format

METRICS COMPUTED:
-----------------
    - Accuracy: Overall classification accuracy
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)  
    - F1 Score: 2 * P * R / (P + R)
    - AUC-ROC: Area under ROC curve
    - AUC-PR: Area under Precision-Recall curve
    - EER: Equal Error Rate (where FPR = FNR)
    - Confusion Matrix: [[TN, FP], [FN, TP]]

AUTHOR: Konrad Kenczuk
VERSION: 1.0.0
"""

from .metrics import compute_metrics, MetricsComputer, EvaluationMetrics, compute_eer
from .benchmark import Benchmark, BenchmarkResult, run_benchmark
from .visualization import plot_confusion_matrix, plot_roc_curve, plot_training_history

__all__ = [
    # Metrics
    "compute_metrics",
    "MetricsComputer", 
    "EvaluationMetrics",
    "compute_eer",
    
    # Benchmarking
    "Benchmark",
    "BenchmarkResult",
    "run_benchmark",
    
    # Visualization
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
]
