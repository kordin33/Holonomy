"""
visualize_benchmark.py - Wizualizacja wyników benchmarku
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load latest results
results_dir = Path("./results/benchmark_full")
json_files = sorted(results_dir.glob("benchmark_*.json"))

if not json_files:
    print("No benchmark results found!")
    exit(1)

latest = json_files[-1]
print(f"Loading: {latest}")

with open(latest, "r") as f:
    results = json.load(f)

# Prepare data
configs = []
overall_acc = []
method_acc = {m: [] for m in ["real", "inpainting", "text2img", "wiki"]}

for r in results:
    config = f"{r['encoder']}\n{r['features']}\n{r['classifier']}"
    configs.append(config)
    overall_acc.append(r["overall_accuracy"])
    for m in method_acc:
        method_acc[m].append(r["per_method"].get(m, 0))

# ============================================================================
# Figure 1: Overall Accuracy Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(configs))
bars = ax.bar(x, overall_acc, color=['#3498db', '#e74c3c'] * 4, edgecolor='black')

# Add value labels
for bar, val in zip(bars, overall_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Benchmark: Overall Accuracy by Configuration', fontsize=14, fontweight='bold')
ax.set_ylim(0.85, 1.02)
ax.axhline(y=max(overall_acc), color='green', linestyle='--', alpha=0.5, label=f'Best: {max(overall_acc):.1%}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "overall_accuracy.png", dpi=150)
print(f"✓ Saved: overall_accuracy.png")

# ============================================================================
# Figure 2: Per-Method Heatmap
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Create matrix
method_names = ["real", "inpainting", "text2img", "wiki"]
matrix = np.array([method_acc[m] for m in method_names])

# Heatmap
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)

# Labels
ax.set_xticks(np.arange(len(configs)))
ax.set_yticks(np.arange(len(method_names)))
ax.set_xticklabels([c.replace('\n', ' / ') for c in configs], fontsize=8, rotation=45, ha='right')
ax.set_yticklabels([m.capitalize() for m in method_names], fontsize=10)

# Add text annotations
for i in range(len(method_names)):
    for j in range(len(configs)):
        val = matrix[i, j]
        color = 'white' if val < 0.7 else 'black'
        ax.text(j, i, f'{val:.1%}', ha='center', va='center', color=color, fontsize=9, fontweight='bold')

ax.set_title('Per-Method Accuracy Heatmap', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, label='Accuracy')

plt.tight_layout()
plt.savefig(results_dir / "method_heatmap.png", dpi=150)
print(f"✓ Saved: method_heatmap.png")

# ============================================================================
# Figure 3: kNN vs SVM Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Group by classifier
knn_results = [r for r in results if r["classifier"] == "kNN"]
svm_results = [r for r in results if r["classifier"] == "SVM"]

for ax, clf_results, title, color in [
    (axes[0], knn_results, "k-NN Classifier", "#3498db"),
    (axes[1], svm_results, "SVM Classifier", "#e74c3c"),
]:
    labels = [f"{r['encoder']}\n{r['features']}" for r in clf_results]
    x = np.arange(len(labels))
    
    # Per-method bars
    width = 0.2
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
    
    for i, method in enumerate(method_names):
        values = [r["per_method"].get(method, 0) for r in clf_results]
        ax.bar(x + i*width - 1.5*width, values, width, label=method.capitalize(), color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0.4, 1.1)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Classifier Comparison: k-NN vs SVM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / "classifier_comparison.png", dpi=150)
print(f"✓ Saved: classifier_comparison.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("📊 BENCHMARK VISUALIZATION COMPLETE")
print("="*60)
print(f"\nFiles saved to: {results_dir}")

# Find best
best = max(results, key=lambda x: x["overall_accuracy"])
print(f"\n🏆 BEST CONFIG:")
print(f"   {best['encoder']} + {best['features']} + {best['classifier']}")
print(f"   Overall: {best['overall_accuracy']:.1%}")

plt.show()
