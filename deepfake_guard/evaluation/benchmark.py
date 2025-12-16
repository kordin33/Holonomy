"""
benchmark.py - Benchmarking Framework

Porównuje różne modele na tych samych datasetach.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .metrics import MetricsComputer, EvaluationMetrics


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single model"""
    model_name: str
    metrics_per_dataset: Dict[str, EvaluationMetrics]
    training_time: float = 0.0
    inference_time: float = 0.0  # per sample
    model_params: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        accs = [m.accuracy for m in self.metrics_per_dataset.values()]
        aucs = [m.auc_roc for m in self.metrics_per_dataset.values()]
        
        return {
            'model_name': self.model_name,
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'training_time': self.training_time,
            'inference_time_ms': self.inference_time * 1000,
            'params_millions': self.model_params / 1e6,
        }


class Benchmark:
    """
    Benchmarking framework for comparing deepfake detection models.
    
    Features:
    - Multiple model comparison
    - Cross-dataset evaluation
    - Timing analysis
    - Automatic result saving
    """
    
    def __init__(
        self,
        dataloaders: Dict[str, DataLoader],
        device: str = "cuda",
        output_dir: Path = Path("./benchmark_results"),
    ):
        self.dataloaders = dataloaders
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, BenchmarkResult] = {}
    
    def add_model(
        self,
        model: nn.Module,
        model_name: str,
        training_time: float = 0.0,
    ) -> BenchmarkResult:
        """
        Add and evaluate a model.
        
        Args:
            model: PyTorch model
            model_name: Name for identification
            training_time: Training time in seconds
            
        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*50}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*50}")
        
        model = model.to(self.device)
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Evaluate on all datasets
        metrics_computer = MetricsComputer(model, self.device)
        metrics_per_dataset = {}
        
        total_inference_time = 0.0
        total_samples = 0
        
        for dataset_name, loader in self.dataloaders.items():
            print(f"\n  Evaluating on {dataset_name}...")
            
            start_time = time.time()
            metrics = metrics_computer.evaluate(loader, return_per_sample=True)
            eval_time = time.time() - start_time
            
            metrics_per_dataset[dataset_name] = metrics
            
            num_samples = len(loader.dataset)
            total_inference_time += eval_time
            total_samples += num_samples
            
            print(f"    Accuracy:  {metrics.accuracy:.4f}")
            print(f"    F1 Score:  {metrics.f1:.4f}")
            print(f"    AUC-ROC:   {metrics.auc_roc:.4f}")
            print(f"    EER:       {metrics.eer:.4f}")
        
        # Calculate average inference time per sample
        avg_inference_time = total_inference_time / total_samples if total_samples > 0 else 0
        
        result = BenchmarkResult(
            model_name=model_name,
            metrics_per_dataset=metrics_per_dataset,
            training_time=training_time,
            inference_time=avg_inference_time,
            model_params=num_params,
        )
        
        self.results[model_name] = result
        
        return result
    
    def compare(self) -> Dict[str, Any]:
        """
        Compare all benchmarked models.
        
        Returns:
            Comparison summary
        """
        if not self.results:
            raise ValueError("No models benchmarked yet!")
        
        comparison = {
            'models': {},
            'best_per_dataset': {},
            'best_overall': None,
        }
        
        # Collect all metrics
        for model_name, result in self.results.items():
            comparison['models'][model_name] = result.get_summary()
        
        # Find best model per dataset
        dataset_names = list(next(iter(self.results.values())).metrics_per_dataset.keys())
        
        for dataset in dataset_names:
            best_model = None
            best_acc = 0
            
            for model_name, result in self.results.items():
                acc = result.metrics_per_dataset[dataset].accuracy
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name
            
            comparison['best_per_dataset'][dataset] = {
                'model': best_model,
                'accuracy': best_acc,
            }
        
        # Find overall best model (by mean accuracy)
        mean_accs = {
            name: result.get_summary()['mean_accuracy']
            for name, result in self.results.items()
        }
        best_overall = max(mean_accs, key=mean_accs.get)
        comparison['best_overall'] = {
            'model': best_overall,
            'mean_accuracy': mean_accs[best_overall],
        }
        
        return comparison
    
    def print_comparison(self) -> None:
        """Print comparison table"""
        comparison = self.compare()
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        
        # Header
        datasets = list(next(iter(self.results.values())).metrics_per_dataset.keys())
        header = f"{'Model':<25} | {'Params':>8} | {'Mean Acc':>8} | "
        header += " | ".join([f"{d[:8]:>8}" for d in datasets])
        print(header)
        print("-" * len(header))
        
        # Rows
        for model_name, result in self.results.items():
            summary = result.get_summary()
            row = f"{model_name:<25} | {summary['params_millions']:>7.1f}M | {summary['mean_accuracy']:>8.4f} | "
            row += " | ".join([
                f"{result.metrics_per_dataset[d].accuracy:>8.4f}"
                for d in datasets
            ])
            print(row)
        
        print("-" * len(header))
        print(f"\nBest overall: {comparison['best_overall']['model']} "
              f"(Mean Acc: {comparison['best_overall']['mean_accuracy']:.4f})")
        
        print("\nBest per dataset:")
        for dataset, info in comparison['best_per_dataset'].items():
            print(f"  {dataset}: {info['model']} ({info['accuracy']:.4f})")
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save results to JSON"""
        output = {
            'comparison': self.compare(),
            'detailed_results': {},
        }
        
        for model_name, result in self.results.items():
            output['detailed_results'][model_name] = {
                'summary': result.get_summary(),
                'per_dataset': {
                    dataset: metrics.to_dict()
                    for dataset, metrics in result.metrics_per_dataset.items()
                }
            }
        
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {path}")
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        comparison = self.compare()
        
        report = ["# Deepfake Detection Benchmark Report\n"]
        report.append(f"## Summary\n")
        report.append(f"**Best Overall Model:** {comparison['best_overall']['model']}\n")
        report.append(f"**Mean Accuracy:** {comparison['best_overall']['mean_accuracy']:.4f}\n\n")
        
        # Table
        report.append("## Results Table\n")
        datasets = list(next(iter(self.results.values())).metrics_per_dataset.keys())
        
        header = "| Model | Params | Mean Acc | " + " | ".join(datasets) + " |"
        separator = "|" + "|".join(["---"] * (3 + len(datasets))) + "|"
        report.append(header)
        report.append(separator)
        
        for model_name, result in self.results.items():
            summary = result.get_summary()
            row = f"| {model_name} | {summary['params_millions']:.1f}M | {summary['mean_accuracy']:.4f} | "
            row += " | ".join([
                f"{result.metrics_per_dataset[d].accuracy:.4f}"
                for d in datasets
            ]) + " |"
            report.append(row)
        
        report.append("\n## Analysis\n")
        report.append("### Cross-Dataset Generalization\n")
        
        for model_name, result in self.results.items():
            accs = [m.accuracy for m in result.metrics_per_dataset.values()]
            report.append(f"- **{model_name}**: Mean {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        
        return "\n".join(report)


def run_benchmark(
    models: Dict[str, nn.Module],
    dataloaders: Dict[str, DataLoader],
    device: str = "cuda",
    output_dir: str = "./benchmark_results",
) -> Benchmark:
    """
    Convenience function to run benchmark on multiple models.
    
    Args:
        models: Dict of model_name -> model
        dataloaders: Dict of dataset_name -> DataLoader
        device: Device to use
        output_dir: Output directory
        
    Returns:
        Benchmark object with results
    """
    benchmark = Benchmark(dataloaders, device, Path(output_dir))
    
    for model_name, model in models.items():
        benchmark.add_model(model, model_name)
    
    benchmark.print_comparison()
    benchmark.save_results()
    
    return benchmark
