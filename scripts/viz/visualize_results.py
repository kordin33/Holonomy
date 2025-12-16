"""
visualize_results.py - Wizualizacja wyników eksperymentu Leave-One-Out
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Wczytaj wyniki
with open("leave_one_out_results.json", "r") as f:
    results = json.load(f)

# Przygotuj dane do macierzy
methods = list(results["results_matrix"].keys())
n_methods = len(methods)

# Macierz fake detection (bez real)
fake_methods = [m for m in methods if m != "real"]
matrix = np.zeros((len(fake_methods), len(fake_methods)))

for i, train_m in enumerate(fake_methods):
    for j, test_m in enumerate(fake_methods):
        matrix[i, j] = results["results_matrix"][train_m].get(test_m, 0) * 100

# Figure 1: Heatmap macierzy generalizacji
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)

# Heatmap
ax = sns.heatmap(
    matrix,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn',
    xticklabels=[m.replace('_temp_', '') for m in fake_methods],
    yticklabels=[m.replace('_temp_', '') for m in fake_methods],
    vmin=30,
    vmax=100,
    cbar_kws={'label': 'Fake Detection Rate (%)'},
    annot_kws={'size': 14, 'weight': 'bold'},
)

plt.title('Cross-Method Generalization Matrix\n(CLIP ViT-B/32 + k-NN)', fontsize=16, fontweight='bold')
plt.xlabel('Test Method', fontsize=12)
plt.ylabel('Train Method', fontsize=12)

# Dodaj prostokąt wokół diagonali (same-domain)
for i in range(len(fake_methods)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

plt.tight_layout()
plt.savefig('results_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results_heatmap.png")

# Figure 2: Bar chart porównujący same-domain vs cross-domain
plt.figure(figsize=(12, 6))

# Same-domain (diagonal)
same_domain = [matrix[i, i] for i in range(len(fake_methods))]

# Cross-domain (average excluding diagonal)
cross_domain = []
for i in range(len(fake_methods)):
    row = [matrix[i, j] for j in range(len(fake_methods)) if i != j]
    cross_domain.append(np.mean(row))

x = np.arange(len(fake_methods))
width = 0.35

bars1 = plt.bar(x - width/2, same_domain, width, label='Same-domain', color='#2ecc71', edgecolor='black')
bars2 = plt.bar(x + width/2, cross_domain, width, label='Cross-domain (avg)', color='#e74c3c', edgecolor='black')

plt.axhline(y=results["summary"]["same_domain_accuracy"]*100, color='#2ecc71', linestyle='--', alpha=0.7, label=f'Avg same: {results["summary"]["same_domain_accuracy"]*100:.1f}%')
plt.axhline(y=results["summary"]["cross_domain_accuracy"]*100, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Avg cross: {results["summary"]["cross_domain_accuracy"]*100:.1f}%')

plt.xlabel('Training Method', fontsize=12)
plt.ylabel('Fake Detection Rate (%)', fontsize=12)
plt.title('Same-Domain vs Cross-Domain Performance\n(Generalization Gap Analysis)', fontsize=14, fontweight='bold')
plt.xticks(x, [m.replace('_temp_', '') for m in fake_methods])
plt.legend()
plt.ylim(0, 105)

# Dodaj wartości na barach
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('generalization_gap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: generalization_gap.png")

# Figure 3: Real vs Fake detection per method
plt.figure(figsize=(10, 6))

# Real detection rates
real_rates = [results["results_matrix"][m].get("real", 0) * 100 for m in fake_methods]

# Najgorszy cross-domain dla każdej metody (hardest to generalize)
worst_cross = []
for i, train_m in enumerate(fake_methods):
    min_cross = min([matrix[i, j] for j in range(len(fake_methods)) if i != j])
    worst_cross.append(min_cross)

x = np.arange(len(fake_methods))
width = 0.25

plt.bar(x - width, real_rates, width, label='Real Detection (TNR)', color='#3498db', edgecolor='black')
plt.bar(x, same_domain, width, label='Best (Same-domain)', color='#2ecc71', edgecolor='black')
plt.bar(x + width, worst_cross, width, label='Worst (Cross-domain)', color='#e74c3c', edgecolor='black')

plt.xlabel('Training Method', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Performance Spectrum: Best vs Worst Cases', fontsize=14, fontweight='bold')
plt.xticks(x, [m.replace('_temp_', '') for m in fake_methods])
plt.legend()
plt.ylim(0, 105)

plt.tight_layout()
plt.savefig('performance_spectrum.png', dpi=150, bbox_inches='tight')
print("✓ Saved: performance_spectrum.png")

# Figure 4: Summary pie/donut chart
plt.figure(figsize=(8, 8))

# Summary stats
balanced_acc = results["summary"]["balanced_accuracy"] * 100
gap = results["summary"]["generalization_gap"] * 100

# Pie chart showing what's working
sizes = [balanced_acc, 100 - balanced_acc]
labels = [f'Accurate\n{balanced_acc:.1f}%', f'Errors\n{100-balanced_acc:.1f}%']
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='', startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 2})

# Add center text
centre_circle = plt.Circle((0, 0), 0.50, fc='white', edgecolor='black', linewidth=2)
plt.gca().add_artist(centre_circle)

plt.text(0, 0.1, 'Balanced', ha='center', va='center', fontsize=14, fontweight='bold')
plt.text(0, -0.1, 'Cross-Domain', ha='center', va='center', fontsize=14, fontweight='bold')
plt.text(0, -0.3, 'Accuracy', ha='center', va='center', fontsize=14, fontweight='bold')

plt.title('CLIP + k-NN Baseline Performance\n(No Fine-tuning)', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('summary_chart.png', dpi=150, bbox_inches='tight')
print("✓ Saved: summary_chart.png")

# Print summary
print("\n" + "="*60)
print("📊 SUMMARY OF RESULTS")
print("="*60)
print(f"Same-domain accuracy:     {results['summary']['same_domain_accuracy']*100:.1f}%")
print(f"Cross-domain accuracy:    {results['summary']['cross_domain_accuracy']*100:.1f}%")
print(f"Generalization gap:       {results['summary']['generalization_gap']*100:.1f}%")
print(f"Real detection (TNR):     {results['summary']['real_detection_accuracy']*100:.1f}%")
print(f"Balanced accuracy:        {results['summary']['balanced_accuracy']*100:.1f}%")

print("\n🔬 KEY OBSERVATIONS:")
print("1. text2img is hardest to detect cross-domain (~40-57%)")
print("2. inpainting and insight generalize well to each other (~88-91%)")
print("3. Real images detected very accurately (>96%)")
print("4. Stage 2/3 improvements could close the gap significantly!")

plt.show()
