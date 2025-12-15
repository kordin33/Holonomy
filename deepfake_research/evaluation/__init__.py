"""
Evaluation package - Metrics, benchmarking, and visualization
"""

from .metrics import compute_metrics, MetricsComputer
from .benchmark import Benchmark, run_benchmark
from .visualization import plot_confusion_matrix, plot_roc_curve, plot_training_history

__all__ = [
    "compute_metrics",
    "MetricsComputer",
    "Benchmark",
    "run_benchmark",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
]
