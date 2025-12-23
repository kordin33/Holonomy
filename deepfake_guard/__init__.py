"""
DeepFake Guard - Loop Holonomy Feature Extraction Framework
==============================================================

A novel approach to AI-generated image detection using concepts from 
differential geometry (holonomy, curvature, torsion) applied to neural 
embedding spaces.

MAIN MODULES:
-------------
• embeddings/ - CLIP/DINOv2 encoder wrappers
• features/production/ - Production-ready feature extractors
• evaluation/ - Metrics, benchmarking, visualization
• models/ - Neural network architectures (optional end-to-end training)

QUICK START:
------------
    from deepfake_guard.embeddings.encoders import get_encoder
    from deepfake_guard.features.production import HolonomyV18
    
    encoder = get_encoder("clip", "ViT-L/14", "cuda")
    extractor = HolonomyV18()
    
    features = extractor.extract_features(encoder, image)  # [126D]

KEY INNOVATION:
---------------
Instead of training end-to-end classifiers, we analyze how images respond
to sequential degradation operations (JPEG, blur, rescaling) in the embedding
space of pre-trained vision models. The geometric properties of these 
"degradation trajectories" (holonomy, curvature, path length) serve as 
discriminative features for real vs. fake classification.

PERFORMANCE:
------------
• SOTA V18 Model: 0.8961 AUC on CIFAKE benchmark
• 126-dimensional feature vector per image
• No training required (uses pre-trained CLIP)
• Simple SVM classifier achieves best results

AUTHOR: Konrad Kenczuk
VERSION: 1.0.0
DATE: 2024-12-22
"""

__version__ = "1.0.0"
__author__ = "Konrad Kenczuk"
__email__ = "konrad@example.com"  # Update with your email

# Core exports for easy access
from deepfake_guard.features.production.holonomy_v18 import HolonomyV18
from deepfake_guard.embeddings.encoders import get_encoder, CLIPEncoder

__all__ = [
    "HolonomyV18",
    "get_encoder", 
    "CLIPEncoder",
    "__version__",
]
