# ⚡ Optymalizacja Loop Holonomy przed połączeniem z ViT-L/14 + SVM

## 🎯 Cel
Maksymalizować moc predykcyjną Loop Holonomy PRZED dodaniem do RGB embeddings.

---

## 📊 Obecny Stan (Baseline)
- **8 pętli** (ręcznie wybrane)
- **Silhouette**: 0.176
- **Przewidywane AUC**: 0.65-0.72
- **Wymiary**: 8D (jedna wartość holonomii per pętla)

---

## 🔧 STRATEGIA 1: Optymalizacja Pętli (Loops)

### 1.1 **Systematyczne przeszukiwanie przestrzeni pętli**

**Parametry do optymalizacji:**
- Długość pętli: 2, 3, 4, 5, 6 transformacji
- Typ transformacji: JPEG, blur, scale, noise
- Kolejność: iteracyjne vs alternating
- Intensywność: gentle vs aggressive

**Przykłady do przetestowania:**

```python
# Krótkie, agresywne
['jpeg_50', 'scale_0.5', 'blur_1.0']

# Długie, stopniowane
['jpeg_90', 'jpeg_80', 'jpeg_70', 'jpeg_60', 'jpeg_50']

# Alternating compression + artifact
['jpeg_60', 'blur_0.5', 'jpeg_80', 'blur_0.3']

# Scale cascade (test mikrotekstur przy różnych skalach)
['scale_0.5', 'scale_0.75', 'scale_0.9', 'scale_0.75', 'scale_0.5']

# Mixed degradations
['noise_0.01', 'jpeg_70', 'blur_0.5', 'scale_0.75']
```

**Metoda:**
1. Generuj N=100 różnych pętli (random + ręczne)
2. Testuj każdą na małej próbce (n=200)
3. Oblicz AUC dla każdej pętli
4. Wybierz top-K (K=10-15) najlepszych
5. Re-testuj na pełnym datasecie

**Implementacja:**
```python
from sklearn.metrics import roc_auc_score

def optimize_loops(encoder, images, labels, n_candidates=100):
    candidate_loops = generate_random_loops(n_candidates)
    
    scores = []
    for loop in candidate_loops:
        hol_features = extract_holonomy_for_loop(encoder, images, loop)
        auc = roc_auc_score(labels, hol_features)
        scores.append((loop, auc))
    
    # Sort by AUC
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:15]  # Top 15
```

---

## 🔧 STRATEGIA 2: Feature Engineering

### 2.1 **Dodatkowe cechy z pętli**

Zamiast tylko `H(x) = ||e(T_n∘...∘T_1(x)) - e(x)||`, ekstraktuj:

**A) Trajektoria embedding w pętli:**
```
z_0 = e(x)
z_1 = e(T_1(x))
z_2 = e(T_2(T_1(x)))
...
z_n = e(T_n(...))

Cechy:
- Cumulative distance: sum(||z_i - z_{i-1}||)
- Max deviation: max_i(||z_i - z_0||)
- Path curvature: mierzy "zakręty" w trajektorii
- Monotonicity: czy odległość rośnie monotonicznie
```

**B) Momentum cechy:**
```python
def extract_trajectory_features(encoder, image, loop):
    embeddings = compute_trajectory(encoder, image, loop)
    
    features = []
    
    # 1. Holonomy (baseline)
    holonomy = np.linalg.norm(embeddings[-1] - embeddings[0])
    features.append(holonomy)
    
    # 2. Total path length
    path_length = sum(np.linalg.norm(embeddings[i] - embeddings[i-1]) 
                      for i in range(1, len(embeddings)))
    features.append(path_length)
    
    # 3. Max deviation from origin
    max_dev = max(np.linalg.norm(e - embeddings[0]) 
                  for e in embeddings)
    features.append(max_dev)
    
    # 4. Curvature (sum of angles)
    curvature = 0
    for i in range(1, len(embeddings)-1):
        v1 = embeddings[i] - embeddings[i-1]
        v2 = embeddings[i+1] - embeddings[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        curvature += angle
    features.append(curvature)
    
    # 5. Monotonicity score
    distances = [np.linalg.norm(e - embeddings[0]) for e in embeddings]
    monotonic = 1 if all(distances[i] <= distances[i+1] for i in range(len(distances)-1)) else 0
    features.append(monotonic)
    
    return np.array(features)
```

**Wymiary:**
- Baseline: 8 pętli × 1 cecha = 8D
- Enhanced: 8 pętli × 5 cech = **40D** ✅

---

### 2.2 **Cross-loop interactions**

**Idea:** Różne pętle mogą mieć komplementarne sygnały.

```python
# Dla każdej pary pętli (i, j):
correlation = np.corrcoef(holonomy_i, holonomy_j)[0, 1]
ratio = holonomy_i / (holonomy_j + epsilon)
difference = abs(holonomy_i - holonomy_j)

# Feature vector: [h_1, h_2, ..., h_8, corr_12, corr_13, ..., ratio_12, ...]
# Wymiary: 8 + C(8,2) + C(8,2) = 8 + 28 + 28 = 64D
```

---

## 🔧 STRATEGIA 3: Normalizacja i Skalowanie

### 3.1 **Per-image normalization**

Różne obrazy mają różne "baseline" holonomie. Normalizuj względem prostej degradacji:

```python
def normalize_holonomy(encoder, image, loop):
    # Holonomy dla testowanej pętli
    H_loop = compute_holonomy(encoder, image, loop)
    
    # Baseline: identity transformation
    H_baseline = compute_holonomy(encoder, image, ['identity', 'identity'])
    
    # Normalized
    H_norm = H_loop / (H_baseline + epsilon)
    
    return H_norm
```

### 3.2 **Feature scaling**

Testuj różne metody:
- StandardScaler (z-score)
- MinMaxScaler ([0, 1])
- RobustScaler (mediana + IQR)
- PowerTransformer (Yeo-Johnson)
- QuantileTransformer (uniform distribution)

```python
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Test różnych scalerów
scalers = {
    'standard': StandardScaler(),
    'power': PowerTransformer(),
    # ...
}

for name, scaler in scalers.items():
    features_scaled = scaler.fit_transform(features)
    auc = evaluate(features_scaled, labels)
    print(f"{name}: AUC={auc:.4f}")
```

---

## 🔧 STRATEGIA 4: Selekcja Cech (Feature Selection)

### 4.1 **Usunięcie redundantnych pętli**

Jeśli 2 pętle dają bardzo podobne wyniki, usuń jedną:

```python
from sklearn.feature_selection import SelectKBest, f_classif

# ANOVA F-test
selector = SelectKBest(f_classif, k=6)  # wybierz 6 najlepszych z 8
features_selected = selector.fit_transform(features, labels)

# Które pętle zostały?
selected_indices = selector.get_support(indices=True)
print(f"Selected loops: {selected_indices}")
```

### 4.2 **PCA/LDA compression**

Jeśli mamy 40D po feature engineering, zredukuj do najważniejszych wymiarów:

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# PCA (unsupervised)
pca = PCA(n_components=10)
features_pca = pca.fit_transform(features_40d)

# LDA (supervised - najlepsze dla klasyfikacji!)
lda = LinearDiscriminantAnalysis(n_components=1)  # Max = n_classes - 1
features_lda = lda.fit_transform(features_40d, labels)
```

---

## 🔧 STRATEGIA 5: Ensemble Loops

### 5.1 **Bagging loops**

Zamiast wybierać top-8 pętli, użyj wielu zestawów:

```python
# 5 zestawów po 8 pętli
loop_ensembles = [
    [loops_1_8],    # Top aggressive
    [loops_9_16],   # Top gentle
    [loops_17_24],  # Top mixed
    [loops_25_32],  # Top long
    [loops_33_40],  # Top short
]

# Dla każdego obrazu: 5 × 8 = 40D
# Lub średnia: mean([h1, h2, h3, h4, h5]) = 8D ale stabilniejsze
```

---

## 🔧 STRATEGIA 6: Adaptacyjne Pętle

### 6.1 **Image-specific loops**

Różne obrazy mogą potrzebować różnych pętli:

```python
def adaptive_loops(encoder, image):
    # Quick test: które pętle dają największą holonomię dla TEGO obrazu?
    holonomies = []
    for loop in candidate_loops:
        h = compute_holonomy(encoder, image, loop)
        holonomies.append((loop, h))
    
    # Wybierz top-3 dla tego obrazu
    holonomies.sort(key=lambda x: x[1], reverse=True)
    selected = holonomies[:3]
    
    # Feature: holonomie z top-3 pętli
    return np.array([h for _, h in selected])
```

---

## 🔧 STRATEGIA 7: Multi-Scale Holonomy

### 7.1 **Testuj w różnych rozdzielczościach**

```python
def multiscale_holonomy(encoder, image, loop):
    holonomies = []
    
    for size in [112, 224, 448]:
        img_resized = image.resize((size, size))
        h = compute_holonomy(encoder, img_resized, loop)
        holonomies.append(h)
    
    # Features: [H_112, H_224, H_448]
    # + derived: std, max-min, ratios
    return holonomies
```

---

## 🎯 PLAN DZIAŁANIA (Priorytet)

### **FAZA 1: Quick Wins** (1-2h implementacji)
1. ✅ **Feature Engineering** (Strategia 2.1)
   - Dodaj: path_length, max_dev, curvature
   - 8 pętli × 5 cech = 40D
   - Oczekiwany boost: +5-10% AUC

2. ✅ **Normalizacja** (Strategia 3.1)
   - Per-image normalization
   - Oczekiwany boost: +2-5% AUC

3. ✅ **Feature Scaling** (Strategia 3.2)
   - Testuj PowerTransformer
   - Oczekiwany boost: +1-3% AUC

### **FAZA 2: Loop Optimization** (2-4h)
4. ✅ **Systematyczne przeszukiwanie** (Strategia 1.1)
   - Generuj 100 kandydatów
   - Testuj na próbce
   - Wybierz top-15
   - Oczekiwany boost: +5-15% AUC

### **FAZA 3: Advanced** (4-8h)
5. 🔬 **Cross-loop interactions** (Strategia 2.2)
6. 🔬 **LDA compression** (Strategia 4.2)
7. 🔬 **Ensemble loops** (Strategia 5)

---

## 💡 OCZEKIWANE WYNIKI

**Baseline (obecny):**
- AUC standalone: ~0.65-0.72
- Wymiary: 8D

**Po optymalizacji (Faza 1-2):**
- AUC standalone: ~**0.75-0.82** ✅
- Wymiary: 40D (z możliwością redukcji do 10-15D przez LDA)

**Impact na RGB+Holonomy:**
- Baseline RGB: ~95% accuracy
- RGB + Holonomy (przed optym): ~96-97%
- RGB + Holonomy (po optym): ~**97-98%** 🚀

---

## 🛠️ IMPLEMENTACJA

Stworzę `optimize_loop_holonomy.py` który:
1. Implementuje wszystkie strategie
2. Testuje systematycznie
3. Wybiera najlepszą konfigurację
4. Zapisuje optymalny extractor

**Uruchomimy to ZARAZ po otrzymaniu wyników current analysis!**
