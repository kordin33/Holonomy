"""
attention.py - Attention Mechanisms for Deepfake Detection

Implementacje:
- Spatial Attention (lokalizacja artefaktów)
- Channel Attention (ważenie kanałów)
- CBAM (Convolutional Block Attention Module)
- Artifact Attention (specjalizowany dla deepfake)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Generuje mapę uwagi wskazującą które regiony obrazu są ważne.
    Dla detekcji deepfake - fokusuje się na regionach z artefaktami.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(pooled)
        
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).
    
    Uczy się które kanały (features) są najważniejsze.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Channel-reweighted features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Global pooling
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        
        # MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        
        # Combine
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Łączy Channel Attention i Spatial Attention.
    
    Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Channel attention → Spatial attention"""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ArtifactAttention(nn.Module):
    """
    Specialized Attention for Detecting Forgery Artifacts.
    
    Inspiracja: Face X-ray i inne metody detekcji blending boundaries.
    
    Ten moduł uczy się zwracać uwagę na:
    - Granice blendingu (blending boundaries)
    - Regiony z nienaturalnymi teksturami
    - Obszary z artefaktami kompresji
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        use_edge_detection: bool = True,
    ):
        super().__init__()
        self.use_edge_detection = use_edge_detection
        
        # Multi-scale convolutions do wykrywania artefaktów
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_3x3 = nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, in_channels // reduction, 5, padding=2)
        
        # Edge detection (Sobel-like)
        if use_edge_detection:
            self.edge_conv_x = nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1, bias=False)
            self.edge_conv_y = nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1, bias=False)
            
            # Initialize with Sobel-like filters
            with torch.no_grad():
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                
                for i in range(in_channels // reduction):
                    self.edge_conv_x.weight[i, i % in_channels] = sobel_x
                    self.edge_conv_y.weight[i, i % in_channels] = sobel_y
        
        # Attention generation
        combined_ch = (in_channels // reduction) * (4 if use_edge_detection else 3)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(combined_ch, combined_ch // 2, 3, padding=1),
            nn.BatchNorm2d(combined_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(combined_ch // 2, 1, 1),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Artifact-attended features [B, C, H, W]
        """
        # Multi-scale features
        feat_1x1 = F.relu(self.conv_1x1(x))
        feat_3x3 = F.relu(self.conv_3x3(x))
        feat_5x5 = F.relu(self.conv_5x5(x))
        
        features = [feat_1x1, feat_3x3, feat_5x5]
        
        # Edge features (gradient magnitude)
        if self.use_edge_detection:
            edge_x = self.edge_conv_x(x)
            edge_y = self.edge_conv_y(x)
            edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
            features.append(edge_magnitude)
        
        # Combine multi-scale features
        combined = torch.cat(features, dim=1)
        
        # Generate attention map
        attention = self.attention_conv(combined)
        
        # Apply attention
        attended = x * attention
        
        # Residual connection
        output = self.output_proj(attended) + x
        
        return output


class BlendingBoundaryAttention(nn.Module):
    """
    Attention module specifically for detecting blending boundaries.
    
    Inspiracja: Face X-ray paper (CVPR 2020)
    
    Deepfake images mają charakterystyczne granice między oryginalną
    i sfałszowaną częścią obrazu.
    """
    
    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        
        # Encoder do wykrywania granic
        self.boundary_encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Boundary prediction (grayscale mask)
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(mid_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(in_channels + mid_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Tuple of:
            - Enhanced features [B, C, H, W]
            - Boundary mask [B, 1, H, W] (dla wizualizacji/supervised training)
        """
        # Encode dla boundary detection
        boundary_features = self.boundary_encoder(x)
        
        # Predict boundary mask
        boundary_mask = self.boundary_predictor(boundary_features)
        
        # Combine original features z boundary features
        combined = torch.cat([x, boundary_features], dim=1)
        output = self.output_proj(combined)
        
        return output, boundary_mask


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-head spatial attention dla feature maps.
    
    Podobne do Vision Transformer attention, ale dla CNN features.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.output = nn.Conv2d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Attended features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # Transpose for attention: [B, heads, HW, head_dim]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).contiguous()  # [B, heads, head_dim, HW]
        out = out.view(B, C, H, W)
        
        return self.output(out) + x  # Residual connection


def get_attention_module(
    attention_type: str,
    channels: int,
    **kwargs,
) -> nn.Module:
    """Factory function do tworzenia attention modules"""
    
    attention_type = attention_type.lower()
    
    if attention_type == "spatial":
        return SpatialAttention(kernel_size=kwargs.get("kernel_size", 7))
    elif attention_type == "channel":
        return ChannelAttention(channels, reduction=kwargs.get("reduction", 16))
    elif attention_type == "cbam":
        return CBAM(channels, reduction=kwargs.get("reduction", 16))
    elif attention_type == "artifact":
        return ArtifactAttention(channels, reduction=kwargs.get("reduction", 8))
    elif attention_type == "boundary":
        return BlendingBoundaryAttention(channels)
    elif attention_type == "multihead":
        return MultiHeadSpatialAttention(channels, num_heads=kwargs.get("num_heads", 8))
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
