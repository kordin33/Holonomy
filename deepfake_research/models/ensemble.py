"""
ensemble.py - Ensemble Methods for Deepfake Detection

Implementacje:
- Simple Average Ensemble
- Learned Weighted Ensemble  
- Stacking Ensemble
"""

from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleDetector(nn.Module):
    """
    Ensemble of multiple deepfake detectors.
    
    Łączy predykcje z różnych modeli dla lepszej dokładności.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_type: str = "average",  # "average", "weighted", "stacking"
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_type = ensemble_type
        self.num_models = len(models)
        
        if ensemble_type == "weighted":
            # Learnable weights dla każdego modelu
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
            
        elif ensemble_type == "stacking":
            # Meta-learner
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * num_classes, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_individual: Return predictions from each model
            
        Returns:
            Dict with ensemble predictions
        """
        outputs = {}
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                out = model(x)
                if isinstance(out, dict):
                    logits = out["logits"]
                else:
                    logits = out
                predictions.append(logits)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=1)  # [B, num_models, num_classes]
        
        # Ensemble
        if self.ensemble_type == "average":
            # Simple average of softmax probabilities
            probs = F.softmax(stacked, dim=-1)
            ensemble_probs = probs.mean(dim=1)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.ensemble_type == "weighted":
            # Weighted average
            weights = F.softmax(self.weights, dim=0)  # Normalize weights
            probs = F.softmax(stacked, dim=-1)
            ensemble_probs = (probs * weights.view(1, -1, 1)).sum(dim=1)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.ensemble_type == "stacking":
            # Meta-learner
            flat = stacked.view(stacked.size(0), -1)
            ensemble_logits = self.meta_learner(flat)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        outputs["logits"] = ensemble_logits
        
        if return_individual:
            outputs["individual_logits"] = stacked
        
        return outputs
    
    def freeze_base_models(self):
        """Freeze all base models (train only ensemble weights)"""
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_base_models(self):
        """Unfreeze all base models"""
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = True


class DiverseEnsemble(nn.Module):
    """
    Ensemble z różnorodnymi modelami.
    
    Automatycznie tworzy ensemble z:
    - EfficientNet
    - ViT
    - Model z frequency analysis
    - Model z attention
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        
        from .hybrid import HybridDeepfakeDetector
        from .backbones import get_backbone
        
        # Model 1: EfficientNet baseline
        eff_backbone, eff_dim = get_backbone("efficientnet_b0", pretrained, num_classes)
        self.efficientnet = eff_backbone
        
        # Model 2: ViT baseline
        vit_backbone, vit_dim = get_backbone("vit_b_16", pretrained, num_classes)
        self.vit = vit_backbone
        
        # Model 3: Hybrid (EfficientNet + Frequency)
        self.hybrid = HybridDeepfakeDetector(
            backbone="efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
            use_frequency=True,
            use_attention=True,
        )
        
        # Learned fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        # EfficientNet predictions
        eff_logits = self.efficientnet(x)
        
        # ViT predictions
        vit_logits = self.vit(x)
        
        # Hybrid predictions
        hybrid_out = self.hybrid(x)
        hybrid_logits = hybrid_out["logits"]
        
        # Fusion
        combined = torch.cat([eff_logits, vit_logits, hybrid_logits], dim=1)
        ensemble_logits = self.fusion(combined)
        
        return {
            "logits": ensemble_logits,
            "efficientnet_logits": eff_logits,
            "vit_logits": vit_logits,
            "hybrid_logits": hybrid_logits,
        }


class ModelAverager:
    """
    Utility class do uśredniania predykcji z modeli bez gradientów.
    
    Używane podczas inference.
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Average predictions from all models.
        
        Args:
            x: Input images
            return_uncertainty: Return prediction variance
        """
        all_probs = []
        
        for model in self.models:
            model.eval()
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        
        # Stack and average
        stacked = torch.stack(all_probs, dim=0)  # [num_models, B, num_classes]
        mean_probs = stacked.mean(dim=0)
        
        result = {
            "probs": mean_probs,
            "predictions": mean_probs.argmax(dim=-1),
        }
        
        if return_uncertainty:
            # Variance as uncertainty measure
            variance = stacked.var(dim=0)
            result["uncertainty"] = variance.mean(dim=-1)
        
        return result
