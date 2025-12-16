# Deepfake Detection Research Project

## 🎯 Cel Projektu

Kompleksowy framework badawczy do porównania różnych architektur detekcji deepfake, z implementacją najnowszych innowacji z literatury naukowej.

## 📁 Struktura Projektu

```
deepfake_research/
├── __init__.py
├── config.py                 # Centralna konfiguracja
│
├── models/                   # Wszystkie architektury
│   ├── __init__.py
│   ├── backbones.py         # EfficientNet, ViT, Xception
│   ├── frequency.py         # FFT, DCT, DWT branches
│   ├── attention.py         # Spatial, Channel, CBAM, Artifact
│   ├── hybrid.py            # Hybrid & Ultimate Detector
│   ├── xray.py              # Face X-ray implementation
│   ├── ensemble.py          # Ensemble methods
│   └── factory.py           # Model factory
│
├── data/                     # Data loading & augmentation
│   ├── __init__.py
│   ├── datasets.py          # Dataset classes
│   ├── augmentation.py      # Augmentation strategies
│   └── sbi.py               # Self-Blended Images
│
├── training/                 # Training utilities
│   ├── __init__.py
│   ├── trainer.py           # Main trainer
│   ├── losses.py            # Loss functions
│   └── optimizers.py        # Optimizers & schedulers
│
└── evaluation/               # Evaluation & benchmarking
    ├── __init__.py
    ├── metrics.py           # Evaluation metrics
    ├── benchmark.py         # Benchmarking framework
    └── visualization.py     # Plotting utilities

run_experiments.py           # Główny skrypt eksperymentów
quick_test.py                # Test poprawności instalacji
```

## 🚀 Szybki Start

### 1. Instalacja zależności

```bash
pip install torch torchvision
pip install scikit-learn numpy pillow tqdm
pip install matplotlib seaborn  # dla wizualizacji
pip install wandb              # opcjonalnie, dla logowania
pip install opencv-python      # dla SBI generator
```

### 2. Przygotowanie danych

Upewnij się, że masz dane w strukturze:
```
./data/
├── A_standardized_224/
│   ├── train/
│   │   ├── fake/
│   │   └── real/
│   ├── val/
│   │   ├── fake/
│   │   └── real/
│   └── test_A/
│       ├── fake/
│       └── real/
└── B_standardized_224/
    └── test_B/
        ├── fake/
        └── real/
```

### 3. Test instalacji

```bash
python quick_test.py
```

### 4. Uruchomienie eksperymentów

```bash
# Pełne porównanie wszystkich modeli (20 epok)
python run_experiments.py --experiment all --epochs 20

# Szybki test (tylko baseline, 5 epok)
python run_experiments.py --experiment baseline --epochs 5

# Tylko zaawansowane modele
python run_experiments.py --experiment advanced --epochs 15

# Ultimate detector z SBI augmentacją
python run_experiments.py --experiment ultimate --epochs 25 --use-sbi
```

## 🏗️ Dostępne Modele

| Model | Opis | Innowacja |
|-------|------|-----------|
| `baseline_efficientnet` | EfficientNet-B0 | Transfer learning baseline |
| `baseline_vit` | ViT-B/16 | Vision Transformer baseline |
| `freq_efficientnet` | EfficientNet + FFT/DCT | Frequency domain analysis |
| `attention_efficientnet` | EfficientNet + CBAM | Attention na artefakty |
| `hybrid` | Spatial + Frequency + Attention | Multi-stream fusion |
| `xray` | Face X-ray | Blending boundary detection |
| `ultimate` | Pełna architektura | Wszystkie komponenty |
| `ensemble` | Ensemble modeli | Kombinacja predykcji |

## 📊 Kluczowe Innowacje

### 1. Frequency Branch (FFT/DCT)
Deepfake zostawia artefakty w dziedzinie częstotliwości niewidoczne gołym okiem.
- **FFT** - Fast Fourier Transform z high-pass filtering
- **DCT** - Discrete Cosine Transform (jak w JPEG)
- **DWT** - Discrete Wavelet Transform

### 2. Self-Blended Images (SBI)
Syntetyczne "fake" obrazy z prawdziwych dla lepszej generalizacji.
- Nie wymaga prawdziwych deepfake'ów do treningu
- Uczy model wykrywać generyczne artefakty blendingu

### 3. Attention Mechanisms
Fokus na regiony z potencjalnymi artefaktami.
- **CBAM** - Channel + Spatial Attention
- **ArtifactAttention** - Specjalizowany dla artefaktów
- **BlendingBoundaryAttention** - Wykrywanie granic blendingu

### 4. Hybrid Architecture
Łączy wszystkie podejścia:
```
Input Image
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
 Spatial       Frequency      Attention
 Branch         Branch         Module
(EfficientNet)  (FFT/DCT)      (CBAM)
    │              │              │
    └──────────────┴──────────────┘
                   │
                   ▼
             Learned Fusion
             (Gated/Attention)
                   │
                   ▼
              Classifier
            (Real / Fake)
```

## 📈 Metryki Ewaluacji

- **Accuracy** - podstawowa dokładność
- **Precision** - precyzja
- **Recall** - czułość
- **F1 Score** - harmoniczna średnia P i R
- **AUC-ROC** - area under ROC curve
- **AUC-PR** - area under Precision-Recall curve
- **EER** - Equal Error Rate

### Cross-Dataset Evaluation
Model jest trenowany na dataset A, a testowany na A i B.
To pokazuje prawdziwą generalizację.

## 📝 Przykładowe Wyniki

Po uruchomieniu `run_experiments.py` otrzymasz:

1. **benchmark_results/full_benchmark.json** - szczegółowe metryki
2. **benchmark_results/BENCHMARK_REPORT.md** - raport w markdown
3. **benchmark_results/cross_dataset_heatmap.png** - wizualizacja
4. **benchmark_results/model_comparison.png** - porównanie modeli

## 🔬 Dla Pracy Naukowej

### Cytowanie metod:

1. **FreqNet** (AAAI 2024): "Frequency-Aware Deepfake Detection"
2. **SBI** (CVPR 2022): "Detecting Deepfakes with Self-Blended Images"
3. **Face X-ray** (CVPR 2020): "Face X-ray for More General Face Forgery Detection"
4. **CBAM** (ECCV 2018): "Convolutional Block Attention Module"

### Struktura eksperymentu dla publikacji:

1. **Ablation Study** - wyłączaj kolejne komponenty
2. **Cross-Dataset** - testuj na niezależnych datasetach
3. **Porównanie z SOTA** - użyj DeepfakeBench
4. **Wizualizacja** - GradCAM, attention maps

## 🛠️ Dodatkowe Skrypty

```bash
# Generowanie SBI datasetu
python -c "from deepfake_research.data.sbi import create_sbi_dataset; create_sbi_dataset('./data/real_images', './data/sbi_dataset')"

# Ewaluacja pojedynczego modelu
python -c "
from deepfake_research.models.factory import create_model
from deepfake_research.evaluation.metrics import MetricsComputer
import torch

model = create_model('ultimate')
model.load_state_dict(torch.load('path/to/checkpoint.pth')['model_state_dict'])
# ... evaluate
"
```

## 📧 Kontakt

Projekt badawczy dla detekcji deepfake.

---

*Utworzono: 2024-12-15*
