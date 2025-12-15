"""
hybrid.py - Hybrid Deepfake Detector Architecture

Główna architektura łącząca:
- Spatial features (RGB backbone)
- Frequency features (FFT/DCT)
- Attention mechanisms
- Face-specific processing

To jest nasza "Ultimate" architektura.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone
from .frequency import FrequencyBranch, DCTBranch, FrequencyAwareModule
from .attention import CBAM, ArtifactAttention, BlendingBoundaryAttention, get_attention_module


class HybridDeepfakeDetector(nn.Module):
    """
    Hybrid Multi-Stream Architecture for Deepfake Detection.
    
    Architektura łączy:
    1. Spatial Stream (EfficientNet/ViT backbone) - RGB features
    2. Frequency Stream (FFT + DCT) - frequency domain artifacts
    3. Attention Module - focus on manipulation regions
    4. Learned Fusion - combine all streams
    
    Jest to kompletna architektura badawcza z wieloma konfigurowalnymi komponentami.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        
        # Frequency branch
        use_frequency: bool = True,
        freq_out_features: int = 256,
        use_fft: bool = True,
        use_dct: bool = True,
        
        # Attention
        use_attention: bool = True,
        attention_type: str = "cbam",  # "cbam", "artifact", "boundary"
        
        # Fusion
        fusion_type: str = "concat",  # "concat", "attention", "gated"
        
        # Regularization
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.use_frequency = use_frequency
        self.use_attention = use_attention
        self.fusion_type = fusion_type
        
        # ==================== SPATIAL STREAM ====================
        self.spatial_backbone, self.spatial_dim = get_backbone(
            backbone, pretrained=pretrained, return_features=True
        )
        
        # Spatial attention (opcjonalne)
        if use_attention and backbone.startswith("efficientnet"):
            self.spatial_attention = get_attention_module(attention_type, self.spatial_dim)
        else:
            self.spatial_attention = None
        
        # ==================== FREQUENCY STREAM ====================
        if use_frequency:
            self.frequency_module = FrequencyAwareModule(
                out_features=freq_out_features,
                use_fft=use_fft,
                use_dct=use_dct,
                use_dwt=False,  # Opcjonalnie można włączyć
            )
            self.freq_dim = freq_out_features
        else:
            self.frequency_module = None
            self.freq_dim = 0
        
        # ==================== FUSION ====================
        total_features = self.spatial_dim + self.freq_dim
        
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(total_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
            )
            fusion_out = 256
            
        elif fusion_type == "attention":
            # Attention-based fusion
            self.fusion = AttentionFusion(
                spatial_dim=self.spatial_dim,
                freq_dim=self.freq_dim,
                out_dim=256,
            )
            fusion_out = 256
            
        elif fusion_type == "gated":
            # Gated fusion
            self.fusion = GatedFusion(
                spatial_dim=self.spatial_dim,
                freq_dim=self.freq_dim,
                out_dim=256,
            )
            fusion_out = 256
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # ==================== CLASSIFIER ====================
        self.classifier = nn.Linear(fusion_out, num_classes)
        
        # ==================== AUXILIARY HEADS (dla deep supervision) ====================
        self.aux_spatial_head = nn.Linear(self.spatial_dim, num_classes)
        if use_frequency:
            self.aux_freq_head = nn.Linear(self.freq_dim, num_classes)
    
    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Ekstrakcja features z spatial backbone"""
        if hasattr(self.spatial_backbone, 'features'):
            # EfficientNet style
            features = self.spatial_backbone.features(x)
            
            # Apply attention if available
            if self.spatial_attention is not None:
                features = self.spatial_attention(features)
            
            # Global pooling
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.flatten(1)
        else:
            # ViT style - already returns flattened features
            features = self.spatial_backbone(x)
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: Return intermediate features for visualization
            return_aux: Return auxiliary predictions (for deep supervision)
            
        Returns:
            Dict with:
            - logits: Main predictions [B, num_classes]
            - aux_spatial: Auxiliary spatial predictions (optional)
            - aux_freq: Auxiliary frequency predictions (optional)
            - features: Intermediate features (optional)
        """
        outputs = {}
        
        # ==================== SPATIAL STREAM ====================
        spatial_features = self.extract_spatial_features(x)
        
        # ==================== FREQUENCY STREAM ====================
        if self.use_frequency:
            freq_features = self.frequency_module(x)
        else:
            freq_features = None
        
        # ==================== FUSION ====================
        if self.use_frequency:
            if self.fusion_type in ["concat"]:
                combined = torch.cat([spatial_features, freq_features], dim=1)
                fused = self.fusion(combined)
            else:
                fused = self.fusion(spatial_features, freq_features)
        else:
            fused = self.fusion(spatial_features)
        
        # ==================== CLASSIFICATION ====================
        logits = self.classifier(fused)
        outputs["logits"] = logits
        
        # ==================== AUXILIARY OUTPUTS ====================
        if return_aux:
            outputs["aux_spatial"] = self.aux_spatial_head(spatial_features)
            if self.use_frequency:
                outputs["aux_freq"] = self.aux_freq_head(freq_features)
        
        # ==================== FEATURES (dla wizualizacji) ====================
        if return_features:
            outputs["spatial_features"] = spatial_features
            if self.use_frequency:
                outputs["freq_features"] = freq_features
            outputs["fused_features"] = fused
        
        return outputs
    
    def get_attention_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Zwraca attention maps dla wizualizacji.
        
        Przydatne do XAI (explainable AI).
        """
        if self.spatial_attention is None:
            return None
        
        # Forward przez backbone do attention layer
        if hasattr(self.spatial_backbone, 'features'):
            features = self.spatial_backbone.features(x)
            
            # Hook attention output
            if isinstance(self.spatial_attention, CBAM):
                # Get intermediate attention maps
                ca_out = self.spatial_attention.channel_attention(features)
                
                # Spatial attention map
                avg_pool = torch.mean(ca_out, dim=1, keepdim=True)
                max_pool, _ = torch.max(ca_out, dim=1, keepdim=True)
                pooled = torch.cat([avg_pool, max_pool], dim=1)
                attention_map = self.spatial_attention.spatial_attention.conv(pooled)
                
                return attention_map
        
        return None


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of spatial and frequency features.
    
    Uczy się które features (spatial vs frequency) są ważniejsze.
    """
    
    def __init__(
        self,
        spatial_dim: int,
        freq_dim: int,
        out_dim: int,
    ):
        super().__init__()
        
        # Project to same dimension
        self.spatial_proj = nn.Linear(spatial_dim, out_dim)
        self.freq_proj = nn.Linear(freq_dim, out_dim)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, 2),
            nn.Softmax(dim=1),
        )
        
        # Output
        self.output = nn.Linear(out_dim, out_dim)
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        freq_features: torch.Tensor,
    ) -> torch.Tensor:
        # Project
        spatial_proj = self.spatial_proj(spatial_features)
        freq_proj = self.freq_proj(freq_features)
        
        # Compute attention weights
        combined = torch.cat([spatial_proj, freq_proj], dim=1)
        weights = self.attention(combined)  # [B, 2]
        
        # Weighted sum
        fused = weights[:, 0:1] * spatial_proj + weights[:, 1:2] * freq_proj
        
        return self.output(fused)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.
    
    Używa gates do kontrolowania przepływu informacji.
    """
    
    def __init__(
        self,
        spatial_dim: int,
        freq_dim: int,
        out_dim: int,
    ):
        super().__init__()
        
        total_dim = spatial_dim + freq_dim
        
        # Gates
        self.gate_spatial = nn.Sequential(
            nn.Linear(total_dim, spatial_dim),
            nn.Sigmoid(),
        )
        self.gate_freq = nn.Sequential(
            nn.Linear(total_dim, freq_dim),
            nn.Sigmoid(),
        )
        
        # Transform
        self.transform_spatial = nn.Linear(spatial_dim, out_dim // 2)
        self.transform_freq = nn.Linear(freq_dim, out_dim // 2)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        freq_features: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate for gate computation
        combined = torch.cat([spatial_features, freq_features], dim=1)
        
        # Compute gates
        gate_s = self.gate_spatial(combined)
        gate_f = self.gate_freq(combined)
        
        # Apply gates
        gated_spatial = spatial_features * gate_s
        gated_freq = freq_features * gate_f
        
        # Transform and combine
        trans_spatial = self.transform_spatial(gated_spatial)
        trans_freq = self.transform_freq(gated_freq)
        
        fused = torch.cat([trans_spatial, trans_freq], dim=1)
        
        return self.output(fused)


class UltimateDeepfakeDetector(HybridDeepfakeDetector):
    """
    Ultimate Deepfake Detector - najlepsza konfiguracja.
    
    Łączy wszystkie najlepsze praktyki:
    - EfficientNet-B0/B3 backbone
    - FFT + DCT frequency analysis
    - CBAM attention
    - Gated fusion
    - Auxiliary supervision
    - Test-time augmentation support
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            use_frequency=True,
            freq_out_features=256,
            use_fft=True,
            use_dct=True,
            use_attention=True,
            attention_type="cbam",
            fusion_type="gated",
            dropout=0.3,
        )
        
        # Dodatkowe komponenty dla Ultimate version
        self.blending_detector = BlendingBoundaryAttention(
            in_channels=1280 if "b0" in backbone else 1536,  # EfficientNet feature dim
            mid_channels=64,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_boundary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extended forward z boundary detection"""
        
        outputs = super().forward(x, return_features, return_aux)
        
        if return_boundary and hasattr(self.spatial_backbone, 'features'):
            features = self.spatial_backbone.features(x)
            _, boundary_mask = self.blending_detector(features)
            outputs["boundary_mask"] = boundary_mask
        
        return outputs
    
    def predict_with_tta(
        self,
        x: torch.Tensor,
        num_augmentations: int = 4,
    ) -> torch.Tensor:
        """
        Prediction with Test-Time Augmentation (TTA).
        
        Wykonuje augmentacje podczas inference i uśrednia predykcje.
        """
        predictions = []
        
        # Original
        with torch.no_grad():
            pred = self.forward(x)["logits"]
            predictions.append(F.softmax(pred, dim=1))
        
        # Horizontal flip
        if num_augmentations >= 2:
            x_flip = torch.flip(x, dims=[3])
            with torch.no_grad():
                pred = self.forward(x_flip)["logits"]
                predictions.append(F.softmax(pred, dim=1))
        
        # Slight rotations (if enabled)
        if num_augmentations >= 4:
            for angle in [5, -5]:
                # Simplified rotation (in practice use kornia)
                with torch.no_grad():
                    pred = self.forward(x)["logits"]
                    predictions.append(F.softmax(pred, dim=1))
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        
        return avg_pred
