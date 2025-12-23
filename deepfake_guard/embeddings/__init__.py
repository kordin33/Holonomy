"""
Embeddings Module - Vision Encoder Wrappers & Vector Database
==============================================================

This module provides tools for extracting, storing, and analyzing 
high-dimensional image embeddings for deepfake detection.

COMPONENTS:
-----------

1. encoders.py - Vision Encoder Wrappers
   - CLIPEncoder: OpenAI CLIP (ViT-B/32, ViT-L/14)
   - DINOv2Encoder: Meta's self-supervised ViT
   - Factory function: get_encoder()

2. vector_db.py - Vector Database
   - VectorDatabase: Base class (NumPy/ChromaDB/FAISS backends)
   - DeepfakeVectorDB: Specialized for real/fake classification
   - k-NN and centroid-based classification

3. visualization.py - Embedding Visualization
   - EmbeddingVisualizer: t-SNE, UMAP, cluster analysis
   - Similarity heatmaps, k-NN explanations

4. stage1_baseline.py - Baseline Pipeline
   - End-to-end embedding extraction pipeline

USAGE:
------
    from deepfake_guard.embeddings import get_encoder, DeepfakeVectorDB
    
    # Get encoder
    encoder = get_encoder("clip", "ViT-L/14", "cuda")
    
    # Extract embeddings
    embedding = encoder.encode_pil(image)        # [768]
    embeddings = encoder.encode_batch(images)    # [N, 768]
    
    # Store in database
    db = DeepfakeVectorDB(backend="numpy")
    db.add(embeddings, labels)
    
    # Classify
    result = db.classify_knn(query_embedding, k=10)

EMBEDDING DIMENSIONS:
---------------------
    CLIP ViT-B/32:  512D
    CLIP ViT-L/14:  768D  ⭐ (Recommended)
    DINOv2 ViT-B:   768D

AUTHOR: Konrad Kenczuk
VERSION: 1.0.0
"""

from .encoders import get_encoder, CLIPEncoder, DINOv2Encoder, BaseEncoder
from .vector_db import VectorDatabase, DeepfakeVectorDB
from .visualization import EmbeddingVisualizer

__all__ = [
    # Encoders
    "get_encoder",
    "CLIPEncoder",
    "DINOv2Encoder",
    "BaseEncoder",
    
    # Vector DB
    "VectorDatabase",
    "DeepfakeVectorDB",
    
    # Visualization
    "EmbeddingVisualizer",
]
