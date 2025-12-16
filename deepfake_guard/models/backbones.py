"""
backbones.py - Backbone networks (EfficientNet, ViT, ConvNeXt, etc.)
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class BackboneFactory:
    """Factory do tworzenia backbone networks"""
    
    SUPPORTED_BACKBONES = [
        "efficientnet_b0",
        "efficientnet_b3",
        "efficientnet_b4",
        "vit_b_16",
        "vit_b_32",
        "convnext_tiny",
        "convnext_small",
        "resnet50",
        "xception",  # Custom implementation
    ]
    
    @staticmethod
    def create(
        name: str,
        pretrained: bool = True,
        num_classes: int = 2,
        return_features: bool = False,
    ) -> Tuple[nn.Module, int]:
        """
        Tworzy backbone i zwraca (model, feature_dim).
        
        Args:
            name: Nazwa backbone
            pretrained: Czy użyć pretrained weights
            num_classes: Liczba klas wyjściowych
            return_features: Jeśli True, zwraca model bez klasyfikatora
            
        Returns:
            Tuple[model, feature_dimension]
        """
        name = name.lower()
        
        if name == "efficientnet_b0":
            return BackboneFactory._create_efficientnet(
                "efficientnet_b0", pretrained, num_classes, return_features
            )
        elif name == "efficientnet_b3":
            return BackboneFactory._create_efficientnet(
                "efficientnet_b3", pretrained, num_classes, return_features
            )
        elif name == "efficientnet_b4":
            return BackboneFactory._create_efficientnet(
                "efficientnet_b4", pretrained, num_classes, return_features
            )
        elif name == "vit_b_16":
            return BackboneFactory._create_vit(
                "vit_b_16", pretrained, num_classes, return_features
            )
        elif name == "vit_b_32":
            return BackboneFactory._create_vit(
                "vit_b_32", pretrained, num_classes, return_features
            )
        elif name == "convnext_tiny":
            return BackboneFactory._create_convnext(
                "convnext_tiny", pretrained, num_classes, return_features
            )
        elif name == "convnext_small":
            return BackboneFactory._create_convnext(
                "convnext_small", pretrained, num_classes, return_features
            )
        elif name == "resnet50":
            return BackboneFactory._create_resnet(
                "resnet50", pretrained, num_classes, return_features
            )
        elif name == "xception":
            return BackboneFactory._create_xception(
                pretrained, num_classes, return_features
            )
        else:
            raise ValueError(f"Unknown backbone: {name}. Supported: {BackboneFactory.SUPPORTED_BACKBONES}")
    
    @staticmethod
    def _create_efficientnet(
        variant: str,
        pretrained: bool,
        num_classes: int,
        return_features: bool,
    ) -> Tuple[nn.Module, int]:
        """Tworzy EfficientNet backbone"""
        if variant == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        elif variant == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            feature_dim = 1536
        elif variant == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            feature_dim = 1792
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")
        
        if return_features:
            # Usuń klasyfikator, zwróć tylko features
            model.classifier = nn.Identity()
        else:
            model.classifier[1] = nn.Linear(feature_dim, num_classes)
        
        return model, feature_dim
    
    @staticmethod
    def _create_vit(
        variant: str,
        pretrained: bool,
        num_classes: int,
        return_features: bool,
    ) -> Tuple[nn.Module, int]:
        """Tworzy Vision Transformer backbone"""
        if variant == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            feature_dim = 768
        elif variant == "vit_b_32":
            weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
            model = models.vit_b_32(weights=weights)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown ViT variant: {variant}")
        
        if return_features:
            model.heads.head = nn.Identity()
        else:
            model.heads.head = nn.Linear(feature_dim, num_classes)
        
        return model, feature_dim
    
    @staticmethod
    def _create_convnext(
        variant: str,
        pretrained: bool,
        num_classes: int,
        return_features: bool,
    ) -> Tuple[nn.Module, int]:
        """Tworzy ConvNeXt backbone"""
        if variant == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            feature_dim = 768
        elif variant == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = models.convnext_small(weights=weights)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")
        
        if return_features:
            model.classifier[2] = nn.Identity()
        else:
            model.classifier[2] = nn.Linear(feature_dim, num_classes)
        
        return model, feature_dim
    
    @staticmethod
    def _create_resnet(
        variant: str,
        pretrained: bool,
        num_classes: int,
        return_features: bool,
    ) -> Tuple[nn.Module, int]:
        """Tworzy ResNet backbone"""
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = 2048
        
        if return_features:
            model.fc = nn.Identity()
        else:
            model.fc = nn.Linear(feature_dim, num_classes)
        
        return model, feature_dim
    
    @staticmethod
    def _create_xception(
        pretrained: bool,
        num_classes: int,
        return_features: bool,
    ) -> Tuple[nn.Module, int]:
        """
        Tworzy Xception backbone.
        Xception jest popularny w detekcji deepfake (FaceForensics++ baseline).
        """
        model = Xception(num_classes=num_classes)
        feature_dim = 2048
        
        if pretrained:
            # Załaduj pretrained weights jeśli dostępne
            try:
                from timm import create_model
                timm_model = create_model('xception', pretrained=True)
                model.load_state_dict(timm_model.state_dict(), strict=False)
            except:
                print("Warning: Could not load pretrained Xception weights. Using random init.")
        
        if return_features:
            model.fc = nn.Identity()
        
        return model, feature_dim


class Xception(nn.Module):
    """
    Xception architecture - popularny baseline w FaceForensics++
    
    Simplified implementation based on the paper:
    "Xception: Deep Learning with Depthwise Separable Convolutions"
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1 = XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2)
        self.block3 = XceptionBlock(256, 728, reps=2, stride=2)
        
        # Middle flow (8 blocks)
        self.middle_blocks = nn.Sequential(*[
            XceptionBlock(728, 728, reps=3, stride=1) for _ in range(8)
        ])
        
        # Exit flow
        self.block4 = XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.middle_blocks(x)
        
        # Exit flow
        x = self.block4(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Ekstrakcja features bez klasyfikatora"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle_blocks(x)
        x = self.block4(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.pool(x)
        x = x.flatten(1)
        
        return x


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Blok Xception z residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reps: int,
        stride: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ):
        super().__init__()
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        # Main path
        layers = []
        channels = in_channels
        
        for i in range(reps):
            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=True))
            
            if grow_first:
                next_channels = out_channels if i == 0 else out_channels
            else:
                next_channels = in_channels if i < reps - 1 else out_channels
            
            layers.append(SeparableConv2d(channels, next_channels))
            layers.append(nn.BatchNorm2d(next_channels))
            channels = next_channels
        
        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


def get_backbone(
    name: str,
    pretrained: bool = True,
    num_classes: int = 2,
    return_features: bool = False,
) -> Tuple[nn.Module, int]:
    """Convenience function do tworzenia backbone"""
    return BackboneFactory.create(name, pretrained, num_classes, return_features)
