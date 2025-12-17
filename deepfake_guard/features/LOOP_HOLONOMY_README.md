# Loop Holonomy - Hypothesis Summary

## 📊 Results (2024-12-16)

### Baseline v2 (NEW) - AUC: 0.861
- **File**: `loop_holonomy_baseline.py`
- **Features**: Minimal (H_raw + shape) = 36D
  - H_raw: 9D (raw holonomy)
  - Shape: 27D (tortuosity, curvature, std_step × 9 loops)
- **Best params**: C=3, gamma=0.01

### Legacy v1 - AUC: 0.785  
- **File**: `loop_holonomy_legacy_v1.py`
- **Features**: H_raw only = 8D

## 🎯 Key Improvements (v1 → v2)
1. Shape features (tortuosity, curvature, std_step)
2. Pipeline(StandardScaler → SVM)
3. GridSearch for hyperparameters
4. NaN protection in curvature calculation

## 📈 Comparison
| Version | AUC | Improvement |
|---------|-----|-------------|
| Legacy v1 | 0.785 | - |
| Baseline v2 | 0.861 | +9.7% |

## 🔬 Mathematical Foundation
```
H_raw = ||z_end - z_0||
tortuosity = path_length / (H_raw + ε)
curvature = mean(1 - cos(Δ_i, Δ_{i+1}))
std_step = std(||Δ_i||)
```

## 📁 Files
- `loop_holonomy_baseline.py` - Current best (v2)
- `loop_holonomy_legacy_v1.py` - Original version
- `degradation_commutator_v3_fixed.py` - Development version
