"""
encoders.py - Image Encoders for Embedding Extraction

Wspierane enkodery:
- CLIP (OpenAI) - ViT-B/32, ViT-L/14
- DINOv2 (Meta) - dla przyszłych stopni
"""

from __future__ import annotations
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm

# Lazy imports dla kompatybilności
_clip_model = None
_clip_preprocess = None


def get_encoder(
    encoder_name: str = "clip",
    model_variant: str = "ViT-B/32",
    device: str = "cuda",
) -> "BaseEncoder":
    """
    Factory function do tworzenia enkoderów.
    
    Args:
        encoder_name: "clip" lub "dinov2"
        model_variant: Wariant modelu
        device: "cuda" lub "cpu"
        
    Returns:
        Encoder instance
    """
    encoder_name = encoder_name.lower()
    
    if encoder_name == "clip":
        return CLIPEncoder(model_variant, device)
    elif encoder_name == "dinov2":
        return DINOv2Encoder(device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


class BaseEncoder(nn.Module):
    """Bazowa klasa dla enkoderów"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.embedding_dim = 512
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def encode_batch(
        self, 
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def dim(self) -> int:
        return self.embedding_dim


class CLIPEncoder(BaseEncoder):
    """
    CLIP Encoder - OpenAI's Contrastive Language-Image Pretraining
    
    Modele:
    - ViT-B/32: 512-dim, szybki, dobry baseline
    - ViT-L/14: 768-dim, lepszy, wolniejszy
    - ViT-L/14@336px: najlepszy, najwolniejszy
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
    ):
        super().__init__(device)
        
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Lazy loading CLIP"""
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
            )
        
        print(f"Loading CLIP {self.model_name}...")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        if "L/14" in self.model_name:
            self.embedding_dim = 768
        else:
            self.embedding_dim = 512
        
        print(f"  ✓ CLIP loaded: {self.model_name}")
        print(f"  ✓ Embedding dim: {self.embedding_dim}")
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode single image tensor.
        
        Args:
            image: [1, 3, H, W] or [3, H, W]
            
        Returns:
            Embedding [1, dim] or [dim]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        embedding = self.model.encode_image(image)
        
        # Normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.float()
    
    @torch.no_grad()
    def encode_pil(self, image: Image.Image) -> np.ndarray:
        """
        Encode PIL Image.
        
        Args:
            image: PIL Image
            
        Returns:
            Embedding [dim]
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.encode_image(image_tensor)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def encode_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode batch of PIL Images.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Embeddings [N, dim]
        """
        all_embeddings = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")
        
        for i in iterator:
            batch = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = torch.stack([
                self.preprocess(img) for img in batch
            ]).to(self.device)
            
            # Encode
            embeddings = self.model.encode_image(batch_tensors)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def encode_dataloader(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode all images from a DataLoader.
        
        Args:
            dataloader: DataLoader returning (images, labels)
            show_progress: Show progress bar
            
        Returns:
            Tuple of (embeddings [N, dim], labels [N])
        """
        all_embeddings = []
        all_labels = []
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Encoding dataset")
        
        for images, labels in iterator:
            images = images.to(self.device)
            
            embeddings = self.model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        return np.vstack(all_embeddings), np.array(all_labels)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (for zero-shot classification)"""
        import clip
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        return text_embedding.cpu().numpy().flatten()


class DINOv2Encoder(BaseEncoder):
    """
    DINOv2 Encoder - Meta's Self-Supervised Vision Transformer
    
    Dla przyszłych stopni - może dawać lepsze wyniki niż CLIP
    dla pure-vision tasks.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        
        print("Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model = self.model.to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.embedding_dim = 768
        print(f"  ✓ DINOv2 loaded, embedding dim: {self.embedding_dim}")
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        embedding = self.model(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.float()
