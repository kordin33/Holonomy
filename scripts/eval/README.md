# 📋 Evaluation Scripts Reference

This directory contains **70+ evaluation scripts** developed during the research process for Loop Holonomy feature development and validation.

## 🔑 Key Scripts (Production-Ready)

### Benchmarking
| Script | Description | Datasets |
|--------|-------------|----------|
| `benchmark_suite.py` | **Full benchmark**: CLIP vs DINOv2 × RGB vs FFT × k-NN/SVM/RF | All available |
| `full_benchmark.py` | ViT-B/32 vs ViT-L/14 comparison | DeepFakeFace |
| `genimage_benchmark.py` | GenImage dataset (Midjourney, Stable Diffusion) | GenImage |
| `benchmark_cifake.py` | CIFAKE benchmark (Real vs AI-generated) | CIFAKE |
| `benchmark_ai_detection.py` | General AI detection benchmark | Multiple |

### V18 Evaluation (SOTA)
| Script | Description |
|--------|-------------|
| `test_final_v18.py` | Final V18 model evaluation |
| `test_v18_operational.py` | Component ablation study for V18 |
| `test_v18_refined.py` | V18 with refinements |
| `test_v18_rv2.py` | V18 revision 2 testing |
| `test_final_v18_opt.py` | Optimized V18 testing |

### Analysis & Visualization
| Script | Description |
|--------|-------------|
| `analyze_loop_holonomy.py` | In-depth loop holonomy analysis |
| `frequency_domain_comparison.py` | FFT feature comparison |
| `visualize_degradation_hypothesis.py` | Degradation hypothesis visualization |
| `visualize_hypothesis_only.py` | Pure hypothesis visualization |
| `cifake_full_analysis.py` | CIFAKE 50K sample full analysis |

## 🔬 Research Iteration Scripts

### Version Progression (test_decomp_v*.py)
These scripts track the evolution from V2 to V17:

```
v2  → Initial decomposition approach
v3  → Fixed pipeline issues
v4  → Added shape features
v5  → Optimized loops
v6  → Patch-based analysis
v7  → Curvature estimation
...
v17 → Pre-V18 refinements
```

| Script | Key Changes |
|--------|-------------|
| `test_decomp_v2.py` | Initial holonomy decomposition |
| `test_decomp_v3.py` | Pipeline fixes |
| `test_decomp_v4.py` | Shape feature addition |
| `test_decomp_v5.py` | Loop optimization |
| `test_decomp_v6.py` | Patch analysis |
| `test_decomp_v7.py` | Curvature features |
| `test_decomp_v10.py` | Normalization improvements |
| `test_decomp_v11.py` | Chordal distance |
| `test_decomp_v12.py` | Multi-scale analysis |
| `test_decomp_v13.py` | Feature selection |
| `test_decomp_v14.py` | Aggregation methods |
| `test_decomp_v15.py` | Patch mean vs std |
| `test_decomp_v16.py` | Final refinements |
| `test_decomp_v17.py` | Pre-V18 validation |

### H2/H3 Component Testing
| Script | Description |
|--------|-------------|
| `test_h2_v6.py` | H2 scale law v6 |
| `test_h2h3_v5.py` | Combined H2+H3 v5 |
| `test_h2h3_v7.py` | Combined H2+H3 v7 |
| `test_h3_fixed.py` | Fixed H3 dispersion |
| `test_h3_v2.py` | H3 v2 testing |
| `test_h3_h2_combined.py` | H3+H2 combination |
| `test_h1_h2_fixed.py` | H1+H2 spectrum analysis |

### Hypothesis Testing
| Script | Description |
|--------|-------------|
| `hypothesis_tester.py` | Statistical hypothesis testing framework |
| `test_6_hypotheses.py` | Testing 6 key hypotheses |
| `test_6_hypotheses_v2.py` | Hypothesis revision |

### Optimization Studies
| Script | Description |
|--------|-------------|
| `test_optimized_v2.py` | Optimization iteration 2 |
| `test_optimized_v3.py` | Optimization iteration 3 |
| `test_optimized_v4.py` | Optimization iteration 4 |
| `test_optimized_vs_production.py` | Optimized vs production comparison |
| `test_all_optimizations_parallel.py` | Parallel optimization testing |
| `quick_test_optimizations.py` | Quick optimization checks |

### Ablation & Component Studies
| Script | Description |
|--------|-------------|
| `test_leave_one_out.py` | Leave-one-out feature importance |
| `test_cross_methods.py` | Cross-method generalization |
| `test_residual_fusion.py` | Residual fusion experiments |
| `test_sota_combo.py` | SOTA combination testing |
| `test_patchwise.py` | Patchwise analysis |
| `test_trajectory.py` | Trajectory analysis |
| `test_baseline.py` | Baseline comparison |
| `test_combined.py` | Feature combination |
| `test_commutator.py` | Commutator features |

### Quality Assurance
| Script | Description |
|--------|-------------|
| `check_cifake_leakage.py` | Data leakage detection |
| `check_hygiene.py` | Data hygiene checks |
| `test_consistency_check.py` | Consistency validation |
| `dataset_bias_check.py` | Dataset bias analysis |
| `diagnostic_and_fix.py` | Diagnostic and repair tools |

### Future Versions
| Script | Description |
|--------|-------------|
| `test_v19_fixed.py` | V19 experiments |
| `test_v19_frontier.py` | V19 frontier testing |
| `test_v20_quick.py` | V20 quick test |
| `test_v21_quick.py` | V21 quick test |
| `test_v22_quick.py` | V22 quick test |
| `test_final_push.py` | Final push experiments |

## 🚀 Running Scripts

### Basic Usage

```bash
# Navigate to project root
cd /path/to/project

# Run specific script
python scripts/eval/test_v18_operational.py

# Run with custom parameters (if supported)
python scripts/eval/benchmark_suite.py
```

### Common Patterns

```python
# Most scripts follow this pattern:
# 1. Load encoder
encoder = get_encoder("clip", "ViT-L/14", "cuda")

# 2. Load dataset
images, labels = load_data(sample_size=200)

# 3. Extract features
features = extract_features(encoder, images)

# 4. Train and evaluate
auc = train_and_evaluate(features, labels)
```

## 📊 Output

Scripts typically output:
- Console logs with metrics
- JSON files in `./results/`
- PNG visualizations (when applicable)

---

*Total: 70+ scripts developed over 22 iterations*
