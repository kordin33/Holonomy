"""
test_6_hypotheses.py - Test wszystkich 6 nowych hipotez vs baseline

Baseline: AUC 0.861 (Minimal = H_raw + shape, 36D)

Testuje każdą hipotezę osobno i porównuje z baseline.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from deepfake_guard.embeddings.encoders import get_encoder
from deepfake_guard.features.holonomy_hypotheses import (
    H1_HolonomySpectrum,
    H2_AreaScaleLaw,
    H3_ConsistencyIndex,
    H4_PatchCoupled,
    H5_HolonomyTensor,
    H6_Orthogonality,
    AllHypothesesExtractor
)
from deepfake_guard.features.degradation_commutator_v3_fixed import extract_minimal_features

# GENIMAGE - trudniejszy dataset!
DATA_DIR = Path("./data/genimage")
OUTPUT_DIR = Path("./results/6_hypotheses_genimage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 100  # Per generator
RANDOM_STATE = 42
BASELINE_AUC = 0.861  # baseline z CIFAKE


def load_data_genimage(n_per_class):
    """Load GenImage: real_pool vs (sd_pool + mj_pool + gan_pool)"""
    images, labels = [], []
    
    # REAL
    real_dir = DATA_DIR / "real_pool"
    files = list(real_dir.glob("**/*.jpg")) + list(real_dir.glob("**/*.png")) + list(real_dir.glob("**/*.JPEG"))
    files = files[:n_per_class * 3]  # 3x więcej real (bo mamy 3 generatory fake)
    for p in tqdm(files, desc="Loading REAL"):
        try:
            img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
            images.append(img)
            labels.append(1)
        except:
            pass
    
    # FAKE: SD + MJ + GAN
    for gen_name in ["sd_pool", "mj_pool", "gan_pool"]:
        gen_dir = DATA_DIR / gen_name
        files = list(gen_dir.glob("**/*.jpg")) + list(gen_dir.glob("**/*.png")) + list(gen_dir.glob("**/*.JPEG"))
        files = files[:n_per_class]
        for p in tqdm(files, desc=f"Loading {gen_name}"):
            try:
                img = Image.open(p).convert("RGB").resize((224, 224), Image.LANCZOS)
                images.append(img)
                labels.append(0)
            except:
                pass
    
    return images, np.array(labels)


def test_with_pipeline(features, labels, name):
    """Test z Pipeline + GridSearch."""
    # Sprawdź NaN
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
    )
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE))
    ])
    
    param_grid = {
        'svm__C': [0.3, 1, 3, 10],
        'svm__gamma': ['scale', 0.01, 0.1]
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, grid.best_estimator_.predict(X_test))
    
    return {
        'name': name,
        'auc': auc,
        'acc': acc,
        'shape': features.shape,
        'best_params': grid.best_params_
    }


def main():
    print("="*70)
    print("🔬 TEST 6 NOWYCH HIPOTEZ HOLONOMII")
    print("   Porównanie z Baseline (AUC=0.861)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load
    print(f"\nLoading {SAMPLE_SIZE} images per generator from GENIMAGE...")
    images, labels = load_data_genimage(SAMPLE_SIZE)
    
    # Encoder
    print("\nInitializing CLIP...")
    encoder = get_encoder("clip", "ViT-L/14", device)
    
    # Collect embeddings for PCA
    print("\nCollecting embeddings for PCA...")
    all_embeddings = encoder.encode_batch(images, batch_size=32, show_progress=True)
    
    # Initialize extractors
    extractors = {
        'H1_Spectrum': H1_HolonomySpectrum(),
        'H2_ScaleLaw': H2_AreaScaleLaw(),
        'H3_Consistency': H3_ConsistencyIndex(n_samples=15),  # Mniej próbek dla szybkości
        'H4_PatchCoupled': H4_PatchCoupled(n_patches=6),
        'H5_Tensor': H5_HolonomyTensor(),
        'H6_Orthogonality': H6_Orthogonality(),
    }
    
    # Fit PCA for H1 and H5
    print("\nFitting PCA...")
    extractors['H1_Spectrum'].fit_pca(all_embeddings)
    extractors['H5_Tensor'].fit_pca(all_embeddings)
    
    # Extract features for each hypothesis
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    all_features = {}
    
    for name, extractor in extractors.items():
        print(f"\n📊 Extracting: {name}")
        features_list = []
        
        for img in tqdm(images, desc=name):
            feat = extractor.extract_features(encoder, img)
            features_list.append(feat)
        
        all_features[name] = np.array(features_list, dtype=np.float32)
        print(f"   Shape: {all_features[name].shape}")
    
    # Also extract baseline for comparison
    print("\n📊 Extracting: Baseline (Minimal)")
    baseline_features = []
    for img in tqdm(images, desc="Baseline"):
        feat = extract_minimal_features(encoder, img)
        baseline_features.append(feat['minimal'])
    
    all_features['Baseline'] = np.array(baseline_features, dtype=np.float32)
    print(f"   Shape: {all_features['Baseline'].shape}")
    
    # Test each
    print("\n" + "="*70)
    print("TESTING WITH PIPELINE + GRIDSEARCH")
    print("="*70)
    
    results = []
    
    for name in ['Baseline', 'H1_Spectrum', 'H2_ScaleLaw', 'H3_Consistency', 
                 'H4_PatchCoupled', 'H5_Tensor', 'H6_Orthogonality']:
        print(f"\n🔬 Testing: {name}")
        res = test_with_pipeline(all_features[name], labels, name)
        results.append(res)
        
        delta = res['auc'] - BASELINE_AUC
        symbol = "✅" if delta >= 0 else "❌"
        print(f"  AUC: {res['auc']:.4f} ({symbol} {delta:+.4f} vs baseline)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Hipoteza':<20} {'AUC':<8} {'vs Baseline':<12} {'Wymiary':<10}")
    print("-"*70)
    
    for res in results:
        delta = res['auc'] - BASELINE_AUC
        symbol = "✅" if delta >= 0 else "⚠️" if delta > -0.05 else "❌"
        print(f"{res['name']:<20} {res['auc']:.4f}  {symbol} {delta:+.4f}      {res['shape'][1]}D")
    
    # Best new hypothesis
    new_results = [r for r in results if r['name'] != 'Baseline']
    best_new = max(new_results, key=lambda x: x['auc'])
    
    print(f"\n🏆 NAJLEPSZA NOWA HIPOTEZA: {best_new['name']} (AUC={best_new['auc']:.4f})")
    
    # Combined test
    print("\n" + "="*70)
    print("🧪 TEST KOMBINACJI: Baseline + Najlepsza nowa")
    print("="*70)
    
    combined = np.concatenate([
        all_features['Baseline'],
        all_features[best_new['name']]
    ], axis=1)
    
    res_combined = test_with_pipeline(combined, labels, "Baseline + " + best_new['name'])
    print(f"  AUC: {res_combined['auc']:.4f} (Baseline+{best_new['name']})")
    
    # All combined
    print("\n🧪 TEST: Wszystkie hipotezy razem")
    all_combined = np.concatenate([all_features[name] for name in all_features.keys()], axis=1)
    res_all = test_with_pipeline(all_combined, labels, "ALL Combined")
    print(f"  AUC: {res_all['auc']:.4f} (wszystkie {all_combined.shape[1]}D)")
    
    # Save
    np.savez_compressed(
        OUTPUT_DIR / "6_hypotheses_results.npz",
        labels=labels,
        **all_features
    )
    
    import json
    results_json = []
    for r in results + [res_combined, res_all]:
        r_copy = r.copy()
        r_copy['shape'] = list(r['shape'])
        results_json.append(r_copy)
    
    with open(OUTPUT_DIR / "6_hypotheses_summary.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_DIR}/")
    
    # Cleanup
    del encoder
    torch.cuda.empty_cache()
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
