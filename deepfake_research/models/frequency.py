"""
frequency.py - Frequency Domain Analysis Modules

Implementacje różnych metod analizy w dziedzinie częstotliwości:
- FFT (Fast Fourier Transform)
- DCT (Discrete Cosine Transform)  
- DWT (Discrete Wavelet Transform)

Te moduły wykrywają artefakty niewidoczne w domenie przestrzennej (RGB).
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyBranch(nn.Module):
    """
    Główny moduł analizy częstotliwości oparty na FFT.
    
    Deepfake zostawia charakterystyczne "fingerprints" w high-frequency spectrum.
    Ten moduł analizuje magnitude i phase spectrum obrazu.
    
    Paper reference: FreqNet (AAAI 2024)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_features: int = 256,
        use_phase: bool = True,
        use_high_freq_filter: bool = True,
    ):
        super().__init__()
        self.use_phase = use_phase
        self.use_high_freq_filter = use_high_freq_filter
        
        # Liczba kanałów wejściowych: magnitude + opcjonalnie phase
        input_ch = in_channels * 2 if use_phase else in_channels
        
        # CNN do analizy spektrum częstotliwości
        self.freq_encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(256, out_features)
        
        # High-frequency filter (opcjonalny)
        if use_high_freq_filter:
            self.register_buffer('high_pass_filter', self._create_high_pass_filter(224))
    
    def _create_high_pass_filter(self, size: int) -> torch.Tensor:
        """Tworzy high-pass filter w dziedzinie częstotliwości"""
        # Tworzymy Gaussian low-pass i odejmujemy od 1
        center = size // 2
        y, x = torch.meshgrid(
            torch.arange(size) - center,
            torch.arange(size) - center,
            indexing='ij'
        )
        radius = torch.sqrt(x.float()**2 + y.float()**2)
        
        # Sigma kontroluje cutoff frequency
        sigma = size // 8
        gaussian = torch.exp(-radius**2 / (2 * sigma**2))
        high_pass = 1 - gaussian
        
        return high_pass.unsqueeze(0).unsqueeze(0)
    
    def extract_frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ekstrakcja features z domeny częstotliwości.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Frequency spectrum tensor [B, C*2, H, W] (magnitude + phase)
        """
        B, C, H, W = x.shape
        
        # 2D FFT dla każdego kanału
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # Magnitude spectrum (log-scaled dla lepszej wizualizacji)
        magnitude = torch.log1p(torch.abs(fft_shifted))
        
        # Opcjonalnie zastosuj high-pass filter
        if self.use_high_freq_filter and hasattr(self, 'high_pass_filter'):
            # Resize filter jeśli potrzeba
            if self.high_pass_filter.shape[-1] != W:
                hp_filter = F.interpolate(
                    self.high_pass_filter,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                hp_filter = self.high_pass_filter
            magnitude = magnitude * hp_filter
        
        if self.use_phase:
            # Phase spectrum (normalized to [-1, 1])
            phase = torch.angle(fft_shifted) / np.pi
            return torch.cat([magnitude, phase], dim=1)
        else:
            return magnitude
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Frequency features [B, out_features]
        """
        # Ekstrakcja frequency features
        freq_features = self.extract_frequency_features(x)
        
        # Encode przez CNN
        encoded = self.freq_encoder(freq_features)
        encoded = encoded.flatten(1)
        
        # Final projection
        out = self.fc(encoded)
        
        return out


class DCTBranch(nn.Module):
    """
    Discrete Cosine Transform Branch.
    
    DCT jest używany w kompresji JPEG i często ujawnia artefakty
    wprowadzone przez generatory deepfake.
    
    Paper reference: "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues"
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_features: int = 256,
        block_size: int = 8,  # DCT block size (jak w JPEG)
    ):
        super().__init__()
        self.block_size = block_size
        
        # DCT basis functions (precomputed)
        self.register_buffer('dct_matrix', self._create_dct_matrix(block_size))
        
        # CNN do analizy DCT coefficients
        self.dct_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(256, out_features)
    
    def _create_dct_matrix(self, n: int) -> torch.Tensor:
        """Tworzy macierz transformacji DCT"""
        dct_m = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_m[k, i] = 1 / np.sqrt(n)
                else:
                    dct_m[k, i] = np.sqrt(2/n) * np.cos(np.pi * (2*i + 1) * k / (2*n))
        return dct_m
    
    def apply_dct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zastosuj block-wise DCT jak w kompresji JPEG.
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            DCT coefficients [B, C, H, W]
        """
        B, C, H, W = x.shape
        bs = self.block_size
        
        # Pad do wielokrotności block_size
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad = x.shape
        
        # Reshape do bloków [B, C, H//bs, bs, W//bs, bs]
        x = x.view(B, C, H_pad // bs, bs, W_pad // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H//bs, W//bs, bs, bs]
        
        # Apply 2D DCT: D * x * D^T
        dct_m = self.dct_matrix
        x = torch.einsum('ij,...jk->...ik', dct_m, x)
        x = torch.einsum('...ij,kj->...ik', x, dct_m)
        
        # Reshape back
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H//bs, bs, W//bs, bs]
        x = x.contiguous().view(B, C, H_pad, W_pad)
        
        # Crop back to original size
        x = x[:, :, :H, :W]
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Apply DCT
        dct_coeffs = self.apply_dct(x)
        
        # Log transform dla lepszej dystrybucji
        dct_coeffs = torch.log1p(torch.abs(dct_coeffs)) * torch.sign(dct_coeffs)
        
        # Encode
        encoded = self.dct_encoder(dct_coeffs)
        encoded = encoded.flatten(1)
        
        return self.fc(encoded)


class WaveletBranch(nn.Module):
    """
    Discrete Wavelet Transform Branch.
    
    Wavelet decomposition pozwala analizować obraz na różnych skalach
    i orientacjach, co jest użyteczne do wykrywania artefaktów.
    
    Paper reference: FSBI (2024) - używa DWT do analizy SBI
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_features: int = 256,
        wavelet: str = "haar",
        levels: int = 2,
    ):
        super().__init__()
        self.levels = levels
        
        # Haar wavelet filters
        if wavelet == "haar":
            low = torch.tensor([1, 1], dtype=torch.float32) / np.sqrt(2)
            high = torch.tensor([1, -1], dtype=torch.float32) / np.sqrt(2)
        else:
            raise ValueError(f"Unsupported wavelet: {wavelet}")
        
        self.register_buffer('low_filter', low)
        self.register_buffer('high_filter', high)
        
        # CNN dla każdego poziomu dekompozycji
        # Po każdym poziomie mamy 4x więcej kanałów (LL, LH, HL, HH)
        total_channels = in_channels * (3 * levels + 1)  # Approximation + details
        
        self.wavelet_encoder = nn.Sequential(
            nn.Conv2d(total_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(256, out_features)
    
    def haar_dwt_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D Haar Wavelet Transform.
        
        Returns:
            LL (approximation), LH (horizontal detail), HL (vertical detail), HH (diagonal detail)
        """
        B, C, H, W = x.shape
        
        # Ensure even dimensions
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
            W += 1
        
        # Reshape for downsampling
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        
        # Compute LL, LH, HL, HH
        ll = (x[:, :, :, 0, :, 0] + x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / 2
        lh = (x[:, :, :, 0, :, 0] + x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 0] - x[:, :, :, 1, :, 1]) / 2
        hl = (x[:, :, :, 0, :, 0] - x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 0] - x[:, :, :, 1, :, 1]) / 2
        hh = (x[:, :, :, 0, :, 0] - x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / 2
        
        return ll, lh, hl, hh
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass z multi-level wavelet decomposition"""
        coefficients = []
        current = x
        
        for _ in range(self.levels):
            ll, lh, hl, hh = self.haar_dwt_2d(current)
            
            # Zbierz detail coefficients
            coefficients.extend([lh, hl, hh])
            
            # Kontynuuj z approximation
            current = ll
        
        # Dodaj final approximation
        coefficients.append(current)
        
        # Resize wszystkich do tego samego rozmiaru i concat
        target_size = coefficients[-1].shape[-2:]
        resized = []
        for coeff in coefficients:
            if coeff.shape[-2:] != target_size:
                coeff = F.interpolate(coeff, size=target_size, mode='bilinear', align_corners=False)
            resized.append(coeff)
        
        combined = torch.cat(resized, dim=1)
        
        # Encode
        encoded = self.wavelet_encoder(combined)
        encoded = encoded.flatten(1)
        
        return self.fc(encoded)


class FrequencyAwareModule(nn.Module):
    """
    Łączy wszystkie metody analizy częstotliwości.
    
    Można używać jako drop-in module do dowolnego backbone.
    """
    
    def __init__(
        self,
        out_features: int = 256,
        use_fft: bool = True,
        use_dct: bool = True,
        use_dwt: bool = False,  # Opcjonalne, wolniejsze
    ):
        super().__init__()
        self.use_fft = use_fft
        self.use_dct = use_dct
        self.use_dwt = use_dwt
        
        branches = []
        branch_features = 0
        
        if use_fft:
            self.fft_branch = FrequencyBranch(out_features=128)
            branch_features += 128
        
        if use_dct:
            self.dct_branch = DCTBranch(out_features=128)
            branch_features += 128
        
        if use_dwt:
            self.dwt_branch = WaveletBranch(out_features=128)
            branch_features += 128
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(branch_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = []
        
        if self.use_fft:
            features.append(self.fft_branch(x))
        
        if self.use_dct:
            features.append(self.dct_branch(x))
        
        if self.use_dwt:
            features.append(self.dwt_branch(x))
        
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)
