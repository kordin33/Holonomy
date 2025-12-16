"""
visualization.py - Visualization utilities
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Skipping plot.")
        return
    
    if class_names is None:
        class_names = ['Fake', 'Real']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot ROC curve.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Skipping plot.")
        return
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_multiple_roc_curves(
    results: Dict[str, tuple],  # model_name -> (y_true, y_prob)
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot ROC curves for multiple models.
    """
    if not HAS_MATPLOTLIB:
        return
    
    from sklearn.metrics import roc_auc_score
    
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (model_name, (y_true, y_prob)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', color=color, linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot Precision-Recall curve.
    """
    if not HAS_MATPLOTLIB:
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    history: List[Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> None:
    """
    Plot training history.
    """
    if not HAS_MATPLOTLIB:
        return
    
    if metrics is None:
        metrics = ['train_loss', 'val_loss', 'val_acc']
    
    epochs = [h['epoch'] for h in history]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [h.get(metric, 0) for h in history]
        ax.plot(epochs, values, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_cross_dataset_heatmap(
    results: Dict[str, Dict[str, float]],  # model -> {dataset: accuracy}
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot cross-dataset evaluation heatmap.
    """
    if not HAS_MATPLOTLIB:
        return
    
    models = list(results.keys())
    datasets = list(results[models[0]].keys())
    
    matrix = np.zeros((len(models), len(datasets)))
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            matrix[i, j] = results[model][dataset]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=datasets,
        yticklabels=models,
        vmin=0.5,
        vmax=1.0,
    )
    plt.title(f'Cross-Dataset {metric_name}')
    plt.xlabel('Test Dataset')
    plt.ylabel('Model')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Bar plot comparing models across datasets.
    """
    if not HAS_MATPLOTLIB:
        return
    
    models = list(results.keys())
    datasets = list(results[models[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, model in enumerate(models):
        values = [results[model][d] for d in datasets]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    title: str = "Attention Map",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> None:
    """
    Visualize attention map overlaid on image.
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attention, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
