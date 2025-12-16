"""
config.py - Centralna konfiguracja projektu
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class DataConfig:
    """Konfiguracja danych"""
    data_root: Path = Path("./data")
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    
    # Datasety
    datasets: List[str] = field(default_factory=lambda: [
        "A_standardized_224",  # Dataset A
        "B_standardized_224",  # Dataset B
    ])
    
    # Augmentacja
    use_sbi: bool = True  # Self-Blended Images
    sbi_probability: float = 0.5
    
    # Normalizacja ImageNet
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Konfiguracja modeli"""
    # Baseline
    backbone: str = "efficientnet_b0"  # lub "vit_b_16", "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.3
    
    # Frequency Branch
    use_frequency_branch: bool = True
    freq_channels: int = 128
    
    # Attention
    use_attention: bool = True
    attention_type: str = "spatial"  # "spatial", "channel", "cbam"
    
    # Face detection
    use_face_crop: bool = False
    
    # Ensemble
    use_ensemble: bool = False


@dataclass
class TrainingConfig:
    """Konfiguracja treningu"""
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    
    # Mixed precision
    use_amp: bool = True
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Seed
    seed: int = 42
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """Pełna konfiguracja eksperymentu"""
    name: str = "baseline_experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Output
    output_dir: Path = Path("./outputs")
    save_checkpoints: bool = True
    log_wandb: bool = False
    wandb_project: str = "deepfake-detection-research"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj do słownika dla logowania"""
        return {
            "name": self.name,
            "data": {
                "img_size": self.data.img_size,
                "batch_size": self.data.batch_size,
                "use_sbi": self.data.use_sbi,
            },
            "model": {
                "backbone": self.model.backbone,
                "use_frequency_branch": self.model.use_frequency_branch,
                "use_attention": self.model.use_attention,
            },
            "training": {
                "epochs": self.training.epochs,
                "lr": self.training.lr,
                "label_smoothing": self.training.label_smoothing,
            }
        }


# Predefiniowane konfiguracje eksperymentów
EXPERIMENT_CONFIGS = {
    # Baseline models
    "baseline_efficientnet": ExperimentConfig(
        name="baseline_efficientnet",
        model=ModelConfig(
            backbone="efficientnet_b0",
            use_frequency_branch=False,
            use_attention=False,
        ),
    ),
    "baseline_vit": ExperimentConfig(
        name="baseline_vit",
        model=ModelConfig(
            backbone="vit_b_16",
            use_frequency_branch=False,
            use_attention=False,
        ),
    ),
    
    # Z Frequency Branch
    "efficientnet_freq": ExperimentConfig(
        name="efficientnet_freq",
        model=ModelConfig(
            backbone="efficientnet_b0",
            use_frequency_branch=True,
            use_attention=False,
        ),
    ),
    "vit_freq": ExperimentConfig(
        name="vit_freq",
        model=ModelConfig(
            backbone="vit_b_16",
            use_frequency_branch=True,
            use_attention=False,
        ),
    ),
    
    # Z Attention
    "efficientnet_attention": ExperimentConfig(
        name="efficientnet_attention",
        model=ModelConfig(
            backbone="efficientnet_b0",
            use_frequency_branch=False,
            use_attention=True,
            attention_type="cbam",
        ),
    ),
    
    # Hybrid (wszystko)
    "hybrid_full": ExperimentConfig(
        name="hybrid_full",
        model=ModelConfig(
            backbone="efficientnet_b0",
            use_frequency_branch=True,
            use_attention=True,
            attention_type="cbam",
        ),
        data=DataConfig(use_sbi=True),
    ),
    
    # Ultimate model
    "ultimate_detector": ExperimentConfig(
        name="ultimate_detector",
        model=ModelConfig(
            backbone="efficientnet_b0",
            use_frequency_branch=True,
            use_attention=True,
            attention_type="cbam",
            use_face_crop=True,
        ),
        data=DataConfig(use_sbi=True, sbi_probability=0.5),
        training=TrainingConfig(
            epochs=30,
            lr=5e-5,
            label_smoothing=0.1,
        ),
    ),
}
