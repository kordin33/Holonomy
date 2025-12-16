"""
xray.py - Face X-ray Implementation

Implementacja metody Face X-ray z CVPR 2020:
"Face X-ray for More General Face Forgery Detection"

X-ray wykrywa blending boundaries między oryginalną i sfałszowaną częścią.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone


class FaceXrayDetector(nn.Module):
    """
    Face X-ray Detector.
    
    Generuje "X-ray" obrazu - mapę wskazującą granice blendingu.
    Dla prawdziwych obrazów X-ray jest puste (same zera).
    Dla fake obrazów X-ray pokazuje granice manipulacji.
    
    Paper: "Face X-ray for More General Face Forgery Detection" (CVPR 2020)
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        
        # Backbone do ekstrakcji features
        self.backbone, self.feature_dim = get_backbone(
            backbone, pretrained=pretrained, return_features=True
        )
        
        # X-ray generator (U-Net style decoder)
        self.xray_decoder = XrayDecoder(
            in_channels=self.feature_dim,
            out_channels=1,  # Grayscale X-ray map
        )
        
        # Binary classifier based on X-ray
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Final classifier combining backbone features + X-ray features
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_xray: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_xray: Whether to return X-ray map
            
        Returns:
            Dict with:
            - logits: Classification predictions
            - xray: X-ray map (optional)
        """
        outputs = {}
        
        # Extract backbone features
        if hasattr(self.backbone, 'features'):
            feature_maps = self.backbone.features(x)
            backbone_features = F.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
        else:
            # ViT doesn't have intermediate feature maps, use direct output
            backbone_features = self.backbone(x)
            feature_maps = backbone_features.unsqueeze(-1).unsqueeze(-1)
        
        # Generate X-ray
        xray = self.xray_decoder(feature_maps, target_size=x.shape[-2:])
        
        # Encode X-ray
        xray_features = self.xray_encoder(xray).flatten(1)
        
        # Combine and classify
        combined = torch.cat([backbone_features, xray_features], dim=1)
        logits = self.classifier(combined)
        
        outputs["logits"] = logits
        
        if return_xray:
            outputs["xray"] = xray
        
        return outputs
    
    def generate_xray(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate only X-ray map (for visualization).
        """
        with torch.no_grad():
            if hasattr(self.backbone, 'features'):
                feature_maps = self.backbone.features(x)
            else:
                feature_maps = self.backbone(x).unsqueeze(-1).unsqueeze(-1)
            
            xray = self.xray_decoder(feature_maps, target_size=x.shape[-2:])
        
        return xray


class XrayDecoder(nn.Module):
    """
    Decoder do generowania X-ray map.
    
    Upsampluje feature maps z backbone do rozdzielczości wejściowej.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        mid_channels: int = 256,
    ):
        super().__init__()
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels // 2, mid_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels // 4, mid_channels // 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels // 8),
            nn.ReLU(inplace=True),
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(mid_channels // 8, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            x: Feature maps from backbone
            target_size: Desired output size (H, W)
        """
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.output(x)
        
        # Ensure correct output size
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class SelfBlendedXray(nn.Module):
    """
    Moduł do treningu Face X-ray z Self-Blended Images.
    
    Zamiast potrzebować prawdziwych fake images do supervised training,
    generujemy syntetyczne blending boundaries z prawdziwych obrazów.
    """
    
    def __init__(self):
        super().__init__()
    
    def generate_synthetic_xray(
        self,
        mask: torch.Tensor,
        blur_sigma: float = 10.0,
    ) -> torch.Tensor:
        """
        Generuje synthetic X-ray ground truth z maski blendingu.
        
        Args:
            mask: Binary blending mask [B, 1, H, W]
            blur_sigma: Blur amount for soft boundaries
            
        Returns:
            X-ray ground truth [B, 1, H, W]
        """
        # Detect edges using Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=mask.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=mask.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        edge_x = F.conv2d(mask, sobel_x, padding=1)
        edge_y = F.conv2d(mask, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        # Normalize
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        
        return edge_magnitude


class XrayLoss(nn.Module):
    """
    Loss function dla Face X-ray training.
    
    Łączy:
    - Binary CE dla klasyfikacji
    - Dice loss dla X-ray map
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        xray_weight: float = 0.5,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.xray_weight = xray_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """Dice loss dla segmentacji X-ray"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pred_xray: Optional[torch.Tensor] = None,
        target_xray: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Classification predictions [B, 2]
            labels: Ground truth labels [B]
            pred_xray: Predicted X-ray [B, 1, H, W]
            target_xray: Ground truth X-ray [B, 1, H, W]
        """
        losses = {}
        
        # Classification loss
        ce = self.ce_loss(logits, labels)
        losses["ce_loss"] = ce * self.classification_weight
        
        # X-ray loss (if provided)
        if pred_xray is not None and target_xray is not None:
            dice = self.dice_loss(pred_xray, target_xray)
            losses["xray_loss"] = dice * self.xray_weight
        
        # Total
        losses["total"] = sum(losses.values())
        
        return losses
