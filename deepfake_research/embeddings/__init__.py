"""
Embeddings Package - Deepfake Detection via Embeddings & Vector Database

Stopniowy system detekcji:
- Stage 1: Pretrained CLIP + Vector DB (baseline)
- Stage 2: Tuning algorytmów klasyfikacji (TODO)
- Stage 3: LoRA fine-tuning (TODO)
- Stage 4: Multimodal RGB + Frequency (TODO)
"""

from .encoders import CLIPEncoder, get_encoder
from .vector_db import VectorDatabase, DeepfakeVectorDB
from .stage1_baseline import Stage1BaselineDetector
from .visualization import EmbeddingVisualizer

__all__ = [
    "CLIPEncoder",
    "get_encoder",
    "VectorDatabase",
    "DeepfakeVectorDB",
    "Stage1BaselineDetector",
    "EmbeddingVisualizer",
]
