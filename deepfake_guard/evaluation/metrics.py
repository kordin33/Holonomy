"""
metrics.py - Evaluation Metrics
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float  # Area under PR curve
    eer: float  # Equal Error Rate
    confusion_matrix: np.ndarray
    
    # Per-sample data for visualization
    y_true: np.ndarray = None
    y_pred: np.ndarray = None
    y_prob: np.ndarray = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'eer': self.eer,
        }
    
    def __str__(self) -> str:
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1 Score:  {self.f1:.4f}\n"
            f"AUC-ROC:   {self.auc_roc:.4f}\n"
            f"AUC-PR:    {self.auc_pr:.4f}\n"
            f"EER:       {self.eer:.4f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for positive class)
        
    Returns:
        EvaluationMetrics object
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Probability-based metrics
    if y_prob is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except:
            auc_pr = 0.5
        
        # Equal Error Rate
        eer = compute_eer(y_true, y_prob)
    else:
        auc_roc = 0.5
        auc_pr = 0.5
        eer = 0.5
    
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        eer=eer,
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def compute_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    EER is the point where False Positive Rate = False Negative Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    
    # Find the point where FPR = FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return eer


class MetricsComputer:
    """
    Class for computing metrics on a model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        positive_class: int = 1,  # Index of "real" class
    ):
        self.model = model
        self.device = device
        self.positive_class = positive_class
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_per_sample: bool = False,
    ) -> EvaluationMetrics:
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            return_per_sample: Whether to return per-sample predictions
            
        Returns:
            EvaluationMetrics object
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, self.positive_class].cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        if not return_per_sample:
            metrics.y_true = None
            metrics.y_pred = None
            metrics.y_prob = None
        
        return metrics
    
    def evaluate_cross_dataset(
        self,
        dataloaders: Dict[str, DataLoader],
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate on multiple datasets.
        
        Args:
            dataloaders: Dict of name -> DataLoader
            
        Returns:
            Dict of name -> EvaluationMetrics
        """
        results = {}
        
        for name, loader in dataloaders.items():
            print(f"Evaluating on {name}...")
            metrics = self.evaluate(loader, return_per_sample=True)
            results[name] = metrics
            print(f"  Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")
        
        return results


def compute_cross_dataset_matrix(
    results: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Create cross-dataset evaluation matrix.
    
    Args:
        results: Dict of model_name -> {dataset_name: accuracy}
        
    Returns:
        Matrix and dataset names
    """
    models = list(results.keys())
    datasets = list(results[models[0]].keys())
    
    matrix = np.zeros((len(models), len(datasets)))
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            matrix[i, j] = results[model][dataset]
    
    return matrix, models, datasets


def compute_generalization_gap(
    in_domain_acc: float,
    out_domain_accs: List[float],
) -> Dict[str, float]:
    """
    Compute generalization metrics.
    
    Args:
        in_domain_acc: Accuracy on training domain
        out_domain_accs: Accuracies on other domains
        
    Returns:
        Dict with generalization metrics
    """
    out_domain_mean = np.mean(out_domain_accs)
    out_domain_std = np.std(out_domain_accs)
    gap = in_domain_acc - out_domain_mean
    
    return {
        'in_domain_acc': in_domain_acc,
        'out_domain_mean': out_domain_mean,
        'out_domain_std': out_domain_std,
        'generalization_gap': gap,
        'relative_gap': gap / in_domain_acc if in_domain_acc > 0 else 0,
    }
