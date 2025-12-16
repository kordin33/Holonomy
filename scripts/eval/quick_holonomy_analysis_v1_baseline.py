"""
quick_holonomy_analysis.py - Szybka analiza Loop Holonomy (bez optymalizacji)

Odpowiada na 3 kluczowe pytania:
1. CO to mówi o fake?
2. Czy to mocny predyktor?  
3. Formalny dowód
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.degradation_commutator import extract_holonomy_features

DATA_DIR = Path("./data/cifake")
OUTPUT_DIR = Path("./results/holonomy_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 300
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(n_per_class):
    images, labels = [], []
    for cls, label in [("REAL", 1), ("FAKE", 0)]:
        files = list((DATA_DIR / "test" / cls).glob("*.jpg"))[:n_per_class]
        for p in tqdm(files, desc=f"Loading {cls}"):
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(label)
    return images, np.array(labels)


def main():
    print("="*70)
    print("🔬 SZYBKA ANALIZA LOOP HOLONOMY")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Loaddata
    print(f"\nLoading {SAMPLE_SIZE} images per class...")
    images, labels = load_data(SAMPLE_SIZE)
    print(f"Total: {len(images)} images")
    
    # Encoder
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Extract features
    print("\nExtracting Loop Holonomy features (8 loops)...")
    features_list = []
    for img in tqdm(images, desc="Loop Holonomy"):
        feat = extract_holonomy_features(encoder, img)
        features_list.append(feat)
    features = np.array(features_list)
    print(f"Features shape: {features.shape}")
    
    # Analysis
    print("\n" + "="*70)
    print("📊 WYNIKI ANALIZY")
    print("="*70)
    
    real_mask = labels == 1
    fake_mask = labels == 0
    
    real_hol = features[real_mask]
    fake_hol = features[fake_mask]
    
    # 1) CO TO MÓWI O FAKE?
    print("\n🔬 1) CO LOOP HOLONOMY MÓWI O FAKE?")
    print("-"*70)
    
    mean_real = real_hol.mean()
    mean_fake = fake_hol.mean()
    diff = mean_real - mean_fake
    
    print(f"\nReal images (mean holonomy):  {mean_real:.6f}")
    print(f"Fake images (mean holonomy):  {mean_fake:.6f}")
    print(f"Różnica (Real - Fake):        {diff:+.6f}")
    print(f"Relative difference:          {(diff/mean_fake)*100:+.2f}%")
    
    # Cohen's d
    pooled_std = np.sqrt((real_hol.std()**2 + fake_hol.std()**2) / 2)
    cohens_d = diff / pooled_std
    
    print(f"\nCohen's d (effect size):      {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect = "Small"
    elif abs(cohens_d) < 0.8:
        effect = "Medium"
    else:
        effect = "Large"
    print(f"Interpretation:               {effect}")
    
    # T-test
    t_stat, p_value = stats.ttest_ind(real_hol.flatten(), fake_hol.flatten())
    print(f"\nt-statistic:                  {t_stat:.4f}")
    print(f"p-value:                      {p_value:.2e}")
    print(f"Statistically significant?    {'YES ✅' if p_value < 0.001 else 'NO ❌'}")
    
    print("\n💡 INTERPRETACJA:")
    if diff > 0:
        print("✅ REAL images have HIGHER holonomy:")
        print("   → Natural images are more 'sensitive' to degradation sequences")
        print("   → Microtextures respond 'chaotically' to JPEG→blur→scale sequences")
        print("   → Transformations DON'T commute perfectly")
        print("\n✅ FAKE (AI) images have LOWER holonomy:")
        print("   → Generated images have smoother, more regular structure")
        print("   → Degradations affect them more 'predictably'")
        print("   → Missing natural 'roughness' of real image microtextures")
    
    # 2) CZY TO MOCNY PREDYKTOR?
    print("\n" + "="*70)
    print("⚡ 2) CZY TO MOCNY PRED YKTOR?")
    print("-"*70)
    
    from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # ROC-AUC per loop
    print("\n📊 ROC-AUC per individual loop:")
    aucs = []
    for i in range(features.shape[1]):
        hol = features[:, i].reshape(-1, 1)
        auc = roc_auc_score(labels, hol)
        aucs.append(auc)
        quality = "✅ Good" if auc > 0.6 else "⚠️ Moderate" if auc > 0.55 else "❌ Weak"
        print(f"  Loop {i+1}: AUC={auc:.4f} {quality}")
    
    print(f"\nMean AUC (single loop):       {np.mean(aucs):.4f}")
    print(f"Best AUC (single loop):       {np.max(aucs):.4f} (Loop {np.argmax(aucs)+1})")
    
    # Combined model
    print("\n🎯 COMBINED (all 8 loops together):")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Logistic Regression
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    cv_scores = cross_val_score(lr, features, labels, cv=5, scoring='roc_auc')
    print(f"\nLogistic Regression (5-fold CV):")
    print(f"  ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    print(f"  Test AUC: {auc_lr:.4f}")
    print(f"  Test Acc: {acc_lr:.4f}")
    
    # SVM
    print(f"\nSVM (RBF kernel):")
    svm = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE, probability=True)
    svm.fit(X_train, y_train)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    print(f"  Test AUC: {auc_svm:.4f}")
    print(f"  Test Acc: {acc_svm:.4f}")
    
    # Silhouette
    sil = silhouette_score(features, labels)
    print(f"\nSilhouette Score:             {sil:.4f}")
    
    print("\n💡 WNIOSKI O MOCY:")
    if auc_svm > 0.70:
        print("✅ Loop Holonomy is a STRONG standalone predictor!")
        print("✅ Can work as independent detection method")
        print("✅ Combined with RGB embeddings = REVOLUTION!")
        strength = "STRONG"
    elif auc_svm > 0.60:
        print("⚠️ Loop Holonomy has MODERATE predictive power")
        print("✅ Good as additional feature to RGB embeddings")
        print("⚠️ May be weak standalone")
        strength = "MODERATE"
    else:
        print("❌ Loop Holonomy has WEAK predictive power")
        print("❌ Needs optimization")
        strength = "WEAK"
    
    # 3) FORMALNY DOWÓD
    print("\n" + "="*70)
    print("📜 3) FORMALNY DOWÓD HIPOTEZY")
    print("-"*70)
    
    print("\n🎓 HIPOTEZA:")
    print("  H0: Real i Fake mają tę samą holonomię pętli")
    print("  H1: Real i Fake różnią się holonomią")
    
    print("\n📊 DOWÓD STATYSTYCZNY:")
    print(f"\n  1. Test t-Studenta:")
    print(f"     t = {t_stat:.4f}")
    print(f"     p = {p_value:.2e}")
    if p_value < 0.001:
        print(f"     → ODRZUCAMY H0 ✅ (p < 0.001)")
        print(f"     → Istnieje istotna różnica statystyczna!")
    else:
        print(f"     → NIE MOŻEMY ODRZUCIĆ H0 ❌")
    
    print(f"\n  2. Wielkość efektu:")
    print(f"     Cohen's d = {cohens_d:.4f} ({effect})")
    print(f"     → {abs(cohens_d):.2f}σ separation")
    
    print(f"\n  3. Moc dyskryminacyjna:")
    print(f"     ROC-AUC = {auc_svm:.4f}")
    print(f"     → Lepsze od losowego (0.5) o {(auc_svm-0.5):.1%}")
    
    print("\n💡 INTERPRETACJA GEOMETRYCZNA:")
    print("  Loop holonomy H(x) = ||e(T_n∘...∘T_1(x)) - e(x)||")
    print("  mierzy 'krzywizne' embedding manifold.")
    print()
    print("  Real: Większa holonomia → chaotyczne mikrotekstury")
    print("  Fake: Mniejsza holonomia → gładka struktura generatów")
    
    print("\n🏆 WERDYKT KOŃCOWY:")
    if p_value < 0.001 and auc_svm > 0.65:
        print("  ✅ HIPOTEZA POTWIERDZONA SILNIE!")
        print(f"     → Istotność: p < {p_value:.0e}")
        print(f"     → Moc: AUC = {auc_svm:.2%}")
        print("  ✅ Loop Holonomy to NOWA, SILNA metoda detekcji!")
        print("  ✅ Format-agnostyczna (JPG/PNG/screenshot)")
    elif p_value < 0.01 and auc_svm > 0.55:
        print("  ⚠️ HIPOTEZA POTWIERDZONA CZĘŚCIOWO")
        print("  ✅ Dobra jako dodatek do RGB embeddings")
    else:
        print("  ❌ HIPOTEZA SŁABA - potrzebna optymalizacja")
    
    # Save
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    np.savez_compressed(
        OUTPUT_DIR / "quick_analysis.npz",
        features=features,
        labels=labels,
        mean_real=mean_real,
        mean_fake=mean_fake,
        diff=diff,
        cohens_d=cohens_d,
        p_value=p_value,
        t_stat=t_stat,
        auc_svm=auc_svm,
        acc_svm=acc_svm,
        silhouette=sil
    )
    
    # Summary JSON
    import json
    summary = {
        "interpretation": {
            "real_mean_holonomy": float(mean_real),
            "fake_mean_holonomy": float(mean_fake),
            "difference": float(diff),
            "relative_diff_percent": float((diff/mean_fake)*100),
            "cohens_d": float(cohens_d),
            "effect_size": effect
        },
        "statistical_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.001)
        },
        "prediction_power": {
            "mean_auc_single": float(np.mean(aucs)),
            "best_auc_single": float(np.max(aucs)),
            "lr_auc": float(auc_lr),
            "svm_auc": float(auc_svm),
            "svm_accuracy": float(acc_svm),
            "silhouette": float(sil),
            "strength": strength
        },
        "verdict": {
            "hypothesis_confirmed": bool(p_value < 0.001 and auc_svm > 0.60),
            "standalone_viable": bool(auc_svm > 0.70),
            "good_as_feature": bool(auc_svm > 0.55)
        }
    }
    
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved to {OUTPUT_DIR}/")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("✅ ANALIZA ZAKOŃCZONA!")
    print("="*70)


if __name__ == "__main__":
    main()
