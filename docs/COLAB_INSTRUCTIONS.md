# 🚀 Jak Uruchomić na Google Colab z GitHub

## Krok 1: Wrzuć projekt na GitHub

### Opcja A: Przez Git (zalecane)
```bash
cd "e:\AI iNflu\Kenczuks"

# Inicjalizuj repo (jeśli nie ma)
git init

# Dodaj wszystkie pliki
git add .

# Commit
git commit -m "Deepfake Detection Research Project"

# Dodaj remote (zmień na swoje repo!)
git remote add origin https://github.com/TWOJ-USERNAME/Kenczuks.git

# Push
git push -u origin main
```

### Opcja B: Przez GitHub Desktop
1. Otwórz GitHub Desktop
2. File → Add Local Repository → Wybierz folder Kenczuks
3. Commit wszystkie zmiany
4. Push to GitHub

---

## Krok 2: Otwórz Colab

1. Idź do [Google Colab](https://colab.research.google.com/)
2. File → Upload Notebook
3. Wrzuć plik `Deepfake_Detection_Colab.ipynb` z tego folderu

**LUB** (jeśli masz już notebook na GitHub):

1. Idź do Colab
2. File → Open Notebook → GitHub
3. Wklej URL do swojego repo

---

## Krok 3: Włącz GPU

1. W Colab: **Runtime → Change runtime type**
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (lub lepsze jak V100, A100 jeśli masz Pro)
4. Save

---

## Krok 4: Uruchom komórki po kolei

1. **Sprawdź GPU** - upewnij się że masz GPU
2. **Sklonuj repo** - zmień URL na swoje repo!
3. **Zainstaluj zależności**
4. **Quick test** - sprawdź czy wszystko działa
5. **Przygotuj dane** - z Drive lub HuggingFace
6. **Uruchom eksperymenty**
7. **Zapisz wyniki na Drive**

---

## 📊 Szacowany Czas na T4 GPU

| Eksperyment | Opis | Czas |
|-------------|------|------|
| `--experiment baseline --epochs 5` | Szybki test | ~15 min |
| `--experiment baseline --epochs 20` | EfficientNet + ViT | ~1 godz |
| `--experiment advanced --epochs 20` | + Frequency + Attention | ~2 godz |
| `--experiment all --epochs 20` | Wszystkie 6 modeli | ~3-4 godz |
| `--experiment ultimate --epochs 25` | Ultimate model | ~1.5 godz |

---

## ⚡ Optymalizacje CUDA (automatyczne)

Projekt używa:
- ✅ **torch.compile()** - 20-40% speedup (PyTorch 2.0+)
- ✅ **cuDNN Benchmark** - 10-20% speedup
- ✅ **TensorFloat-32** - 3x szybszy matmul na Ampere GPUs
- ✅ **Mixed Precision (AMP)** - 2x szybciej, mniej VRAM
- ✅ **Flash Attention** - automatyczne dla ViT

---

## 🔧 Troubleshooting

### "CUDA out of memory"
Zmniejsz batch size:
```bash
python run_experiments.py --experiment all --epochs 20 --batch-size 16
```

### "Module not found"
Upewnij się że jesteś w folderze projektu:
```python
%cd /content/Kenczuks
```

### Colab się zresetował
- Zapisuj wyniki na Google Drive regularnie!
- Użyj komórki "Zapisz wyniki na Drive"

### Wolny download danych
Użyj mniejszego datasetu:
```bash
python efficientnet_b0_deepfake.py --prepare --max-per-class-a 1000
```

---

## 📁 Struktura Wyników na Drive

Po zapisaniu na Drive:
```
/content/drive/MyDrive/deepfake_results_YYYYMMDD_HHMMSS/
├── benchmark/
│   ├── full_benchmark.json      # Wszystkie metryki
│   └── BENCHMARK_REPORT.md      # Raport markdown
├── cross_dataset_heatmap.png    # Wizualizacja
├── model_comparison.png         # Porównanie modeli
└── [model_name]/
    ├── [model_name]_best.pth    # Wagi najlepszego modelu
    └── [model_name]_history.json # Historia treningu
```

---

## 🎯 Rekomendowany Workflow

1. **Pierwszy raz:** `--experiment baseline --epochs 5` (szybki test)
2. **Jeśli działa:** `--experiment all --epochs 10` (porównanie)
3. **Pełny benchmark:** `--experiment all --epochs 20`
4. **Najlepszy model:** `--experiment ultimate --epochs 30 --use-sbi`

---

*Powodzenia! 🚀*
