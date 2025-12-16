"""
losses.py - Loss Functions for Deepfake Detection
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy with Label Smoothing.
    
    Label smoothing helps prevent overconfident predictions
    and improves generalization.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        
        self.confidence = 1.0 - smoothing
        self.smooth_value = smoothing / (num_classes - 1)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predictions [B, C]
            targets: Ground truth labels [B]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smooth_value)
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute KL divergence
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses on hard examples by down-weighting easy ones.
    
    Paper: "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predictions [B, C]
            targets: Ground truth labels [B]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DeepfakeLoss(nn.Module):
    """
    Combined loss for deepfake detection.
    
    Łączy:
    - Classification loss (CE lub Focal)
    - Label smoothing
    - Optional auxiliary losses
    """
    
    def __init__(
        self,
        loss_type: str = 'ce',  # 'ce', 'focal', 'smooth'
        label_smoothing: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.3,
        xray_weight: float = 0.5,
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.aux_weight = aux_weight
        self.xray_weight = xray_weight
        
        if loss_type == 'focal':
            self.main_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'smooth' or label_smoothing > 0:
            self.main_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        else:
            self.main_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: dict | torch.Tensor,
        targets: torch.Tensor,
        xray_target: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            outputs: Model outputs (dict or tensor)
            targets: Ground truth labels [B]
            xray_target: Optional X-ray ground truth [B, 1, H, W]
            
        Returns:
            Dict with individual and total losses
        """
        losses = {}
        
        # Main classification loss
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        losses['main'] = self.main_loss(logits, targets)
        
        # Auxiliary losses
        if isinstance(outputs, dict):
            if 'aux_spatial' in outputs:
                aux_loss = self.main_loss(outputs['aux_spatial'], targets)
                losses['aux_spatial'] = aux_loss * self.aux_weight
            
            if 'aux_freq' in outputs:
                aux_loss = self.main_loss(outputs['aux_freq'], targets)
                losses['aux_freq'] = aux_loss * self.aux_weight
            
            if 'xray' in outputs and xray_target is not None:
                xray_loss = self._xray_loss(outputs['xray'], xray_target)
                losses['xray'] = xray_loss * self.xray_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _xray_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute X-ray prediction loss (BCE + Dice)"""
        # Binary cross-entropy
        bce = F.binary_cross_entropy(pred, target, reduction='mean')
        
        # Dice loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        
        return bce + dice


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for representation learning.
    
    Może być użyte do uczenia lepszych features.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: Feature embeddings [B, D]
            labels: Labels [B]
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_neg = 1 - mask_pos
        
        # Remove diagonal
        mask_diag = torch.eye(features.size(0), device=features.device)
        mask_pos = mask_pos - mask_diag
        
        # Contrastive loss
        exp_sim = torch.exp(similarity)
        
        # Positive pairs
        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        
        # Negative pairs
        neg_sim = (exp_sim * mask_neg).sum(dim=1)
        
        # Loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-6))
        
        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center Loss - penalizuje odległość features od centrum klasy.
    
    Pomaga w uczeniu bardziej dyskryminatywnych features.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        feature_dim: int = 256,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: Feature embeddings [B, D]
            labels: Labels [B]
        """
        batch_size = features.size(0)
        
        # Get centers for each sample
        centers_batch = self.centers[labels]
        
        # Compute L2 distance
        distances = (features - centers_batch).pow(2).sum(dim=1)
        
        return distances.mean()
    
    def update_centers(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update centers with moving average"""
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = (labels == i)
                if mask.sum() > 0:
                    class_features = features[mask].mean(dim=0)
                    self.centers[i] = (
                        self.alpha * self.centers[i] +
                        (1 - self.alpha) * class_features
                    )
