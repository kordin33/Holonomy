"""
🔬 PRODUCTION FEATURE EXTRACTORS
=================================

This module contains the final, validated feature extractors for AI-generated
image detection based on Loop Holonomy theory.

PRODUCTION-READY MODELS:
========================

1. HolonomyV18 (holonomy_v18.py) - ⭐ SOTA
   - AUC: 0.8961 on CIFAKE
   - 126D features: Global (63D) + PatchMean (63D)
   - Best performing, simplest architecture
   - Recommended for production use

2. H3_NormalizedDispersionV2 (h3_dispersion.py)
   - AUC: 0.835 (solo)
   - 9D features: cosine dispersion, path holonomy, covariance trace
   - Useful for lightweight deployments

3. H2_AreaScaleLaw_Fixed (h2_scale_law.py)
   - AUC: 0.804 (solo)
   - 5D features: power-law exponent α, R², residual std
   - Captures scaling behavior of holonomy

4. BaselineFeatures (baseline.py)
   - AUC: 0.756-0.780
   - 36D features: H_raw, tortuosity, curvature, std_step
   - Foundation for all holonomy methods

MATHEMATICAL BACKGROUND:
========================

Loop Holonomy: In differential geometry, holonomy measures the failure of 
parallel transport around a closed loop to return a vector to its original
state. In our context:

    H = ||z_end - z_0||

where z_i are embeddings of the image after sequential degradations.

The key insight is that REAL and FAKE images have statistically different
holonomy distributions due to:
- Internal texture/frequency characteristics
- Artifact response to compression/blur
- Interpolation behavior differences

USAGE:
======
    from deepfake_guard.features.production.holonomy_v18 import HolonomyV18
    from deepfake_guard.embeddings.encoders import get_encoder
    
    encoder = get_encoder("clip", "ViT-L/14", "cuda")
    extractor = HolonomyV18()
    
    features = extractor.extract_features(encoder, image)  # [126]

CHANGELOG:
==========
- 2024-12-16: Initial production checkpoints
- 2024-12-17: Added H2/H3 specialized extractors  
- 2024-12-22: V18 validated as SOTA (0.8961 AUC)

⚠️ DO NOT MODIFY PRODUCTION FILES WITHOUT VALIDATION ⚠️
"""

__version__ = "1.0.0"
__author__ = "Konrad Kenczuk"

# Main production extractors
from .holonomy_v18 import HolonomyV18, compute_baseline_features
from .h3_dispersion import H3_NormalizedDispersionV2, H3_NormalizedDispersionV2Fast
from .h2_scale_law import H2_AreaScaleLaw_Fixed, H1_HolonomySpectrum_Fixed
from .baseline import extract_minimal_features, extract_batch_minimal_features

__all__ = [
    # Primary SOTA
    "HolonomyV18",
    "compute_baseline_features",
    
    # Alternative extractors
    "H3_NormalizedDispersionV2",
    "H3_NormalizedDispersionV2Fast",
    "H2_AreaScaleLaw_Fixed",
    "H1_HolonomySpectrum_Fixed",
    
    # Baseline
    "extract_minimal_features",
    "extract_batch_minimal_features",
]

# Model registry for programmatic access
PRODUCTION_MODELS = {
    "v18": {
        "class": "HolonomyV18",
        "dims": 126,
        "auc": 0.8961,
        "description": "SOTA: Global + PatchMean holonomy features",
    },
    "h3_dispersion": {
        "class": "H3_NormalizedDispersionV2",
        "dims": 9,
        "auc": 0.835,
        "description": "Normalized dispersion metrics (D_cos, D_path, D_cov)",
    },
    "h2_scale": {
        "class": "H2_AreaScaleLaw_Fixed",
        "dims": 5,
        "auc": 0.804,
        "description": "Power-law exponent of holonomy vs degradation strength",
    },
    "baseline": {
        "class": "extract_minimal_features",
        "dims": 36,
        "auc": 0.756,
        "description": "Trajectory shape features (H_raw, tortuosity, curvature)",
    },
}


def get_production_model(name: str):
    """
    Factory function to get production model by name.
    
    Args:
        name: One of 'v18', 'h3_dispersion', 'h2_scale', 'baseline'
        
    Returns:
        Model instance or function
        
    Example:
        extractor = get_production_model("v18")
        features = extractor.extract_features(encoder, image)
    """
    if name not in PRODUCTION_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(PRODUCTION_MODELS.keys())}")
    
    info = PRODUCTION_MODELS[name]
    
    if name == "v18":
        return HolonomyV18()
    elif name == "h3_dispersion":
        return H3_NormalizedDispersionV2()
    elif name == "h2_scale":
        return H2_AreaScaleLaw_Fixed()
    elif name == "baseline":
        return extract_minimal_features
    
    raise ValueError(f"Model {name} not properly registered")
