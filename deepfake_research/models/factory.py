"""
factory.py - Model Factory

Centralne miejsce do tworzenia modeli.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import torch.nn as nn

from ..config import ModelConfig


# Registry wszystkich dostępnych modeli
MODEL_REGISTRY = {
    "baseline_efficientnet": "EfficientNet-B0 baseline",
    "baseline_vit": "ViT-B/16 baseline",
    "baseline_xception": "Xception (FaceForensics++ baseline)",
    "freq_efficientnet": "EfficientNet + Frequency Branch",
    "freq_vit": "ViT + Frequency Branch",
    "attention_efficientnet": "EfficientNet + Attention",
    "hybrid": "Hybrid (Spatial + Frequency + Attention)",
    "xray": "Face X-ray Detector",
    "ensemble": "Ensemble of multiple models",
    "ultimate": "Ultimate Detector (wszystkie komponenty)",
}


def create_model(
    model_name: str,
    config: Optional[ModelConfig] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function do tworzenia modeli.
    
    Args:
        model_name: Nazwa modelu z MODEL_REGISTRY
        config: ModelConfig (opcjonalnie)
        **kwargs: Dodatkowe argumenty
        
    Returns:
        Zainicjalizowany model
    """
    if config is None:
        config = ModelConfig()
    
    model_name = model_name.lower()
    
    # ==================== BASELINES ====================
    if model_name == "baseline_efficientnet":
        from .backbones import get_backbone
        model, _ = get_backbone(
            "efficientnet_b0",
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )
        return model
    
    elif model_name == "baseline_vit":
        from .backbones import get_backbone
        model, _ = get_backbone(
            "vit_b_16",
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )
        return model
    
    elif model_name == "baseline_xception":
        from .backbones import Xception
        return Xception(num_classes=config.num_classes)
    
    # ==================== FREQUENCY MODELS ====================
    elif model_name == "freq_efficientnet":
        from .hybrid import HybridDeepfakeDetector
        return HybridDeepfakeDetector(
            backbone="efficientnet_b0",
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            use_frequency=True,
            use_attention=False,
        )
    
    elif model_name == "freq_vit":
        from .hybrid import HybridDeepfakeDetector
        return HybridDeepfakeDetector(
            backbone="vit_b_16",
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            use_frequency=True,
            use_attention=False,
        )
    
    # ==================== ATTENTION MODELS ====================
    elif model_name == "attention_efficientnet":
        from .hybrid import HybridDeepfakeDetector
        return HybridDeepfakeDetector(
            backbone="efficientnet_b0",
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            use_frequency=False,
            use_attention=True,
            attention_type=config.attention_type,
        )
    
    # ==================== HYBRID ====================
    elif model_name == "hybrid":
        from .hybrid import HybridDeepfakeDetector
        return HybridDeepfakeDetector(
            backbone=config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            use_frequency=config.use_frequency_branch,
            use_attention=config.use_attention,
            attention_type=config.attention_type,
            dropout=config.dropout,
        )
    
    # ==================== XRAY ====================
    elif model_name == "xray":
        from .xray import FaceXrayDetector
        return FaceXrayDetector(
            backbone=config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )
    
    # ==================== ENSEMBLE ====================
    elif model_name == "ensemble":
        from .ensemble import DiverseEnsemble
        return DiverseEnsemble(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        )
    
    # ==================== ULTIMATE ====================
    elif model_name == "ultimate":
        from .hybrid import UltimateDeepfakeDetector
        return UltimateDeepfakeDetector(
            backbone=config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Zwraca informacje o modelu"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return {
        "name": model_name,
        "description": MODEL_REGISTRY[model_name],
    }


def list_models() -> Dict[str, str]:
    """Lista wszystkich dostępnych modeli"""
    return MODEL_REGISTRY.copy()
