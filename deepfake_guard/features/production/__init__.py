"""
PRODUCTION CHECKPOINTS - Najlepsze modele Loop Holonomy

Ten folder zawiera TYLKO sprawdzone, stabilne wersje feature extractorów.
NIE EDYTOWAĆ - to są twarde checkpointy!

LISTA NAJLEPSZYCH:
==================

1. h3_dispersion.py - H3 Normalized Dispersion V2
   - AUC solo: 0.835 ✅ (bije baseline!)
   - L2 normalizacja, cosine path, grid patches
   
2. h2_scale_law.py - H2 Area/Scale Law Fixed
   - AUC solo: 0.804 ✅
   - 11 punktów skali, area = zmierzona siła

3. baseline.py - Loop Holonomy Baseline
   - AUC: 0.756-0.780
   - H_raw + shape features (curvature, tortuosity, std_step)

UŻYCIE:
=======
from deepfake_guard.features.production.h3_dispersion import H3_NormalizedDispersion
from deepfake_guard.features.production.h2_scale_law import H2_AreaScaleLaw
from deepfake_guard.features.production.baseline import extract_minimal_features

HISTORIA:
=========
- 2024-12-16: Initial checkpoint z H3 V2, H2 Fixed, Baseline V3
"""
