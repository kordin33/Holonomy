"""
stage1_baseline.py - Stage 1: Baseline Detector

Używa pretrained CLIP (bez żadnego fine-tuningu) + Vector DB
do detekcji deepfake przez k-NN similarity search.

To jest BASELINE - oczekuj ~65-75% accuracy.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from .encoders import CLIPEncoder, get_encoder
from .vector_db import DeepfakeVectorDB


@dataclass
class Stage1Config:
    """Konfiguracja Stage 1"""
    encoder_name: str = "clip"
    encoder_variant: str = "ViT-B/32"  # "ViT-L/14" dla lepszych wyników
    k_neighbors: int = 10
    db_backend: str = "numpy"  # "chromadb" lub "faiss" dla dużych datasetów
    device: str = "cuda"


@dataclass
class DetectionResult:
    """Wynik detekcji"""
    prediction: str  # "real" lub "fake"
    confidence: float
    method: str
    details: Dict[str, Any]


class Stage1BaselineDetector:
    """
    Stage 1: Baseline Deepfake Detector
    
    Podejście:
    1. Użyj pretrained CLIP do ekstrakcji embeddings
    2. Zbuduj bazę wektorową z known real/fake
    3. Dla query: znajdź k najbliższych sąsiadów
    4. Voting: więcej podobnych do real czy fake?
    
    Oczekiwana accuracy: 65-75%
    """
    
    def __init__(self, config: Optional[Stage1Config] = None):
        self.config = config or Stage1Config()
        
        # Initialize encoder
        print(f"\n{'='*50}")
        print("Stage 1: Baseline Detector")
        print(f"{'='*50}")
        
        self.encoder = get_encoder(
            encoder_name=self.config.encoder_name,
            model_variant=self.config.encoder_variant,
            device=self.config.device,
        )
        
        # Initialize vector database
        self.db = DeepfakeVectorDB(backend=self.config.db_backend)
        
        # Stats
        self.is_fitted = False
    
    def fit(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Zbuduj bazę wektorową z dataloadera.
        
        Args:
            dataloader: DataLoader returning (images, labels)
                        labels: 0=fake, 1=real
            show_progress: Show progress bar
            
        Returns:
            Statistics dict
        """
        print("\nBuilding vector database...")
        
        all_embeddings = []
        all_labels = []
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Encoding images")
        
        for images, labels in iterator:
            # Get embeddings
            embeddings, _ = self.encoder.encode_dataloader(
                [(images, labels)],
                show_progress=False,
            )
            
            # Convert labels to strings
            label_strs = ["real" if l == 1 else "fake" for l in labels.numpy()]
            
            all_embeddings.append(embeddings)
            all_labels.extend(label_strs)
        
        # Add to database
        embeddings_array = np.vstack(all_embeddings)
        self.db.add(embeddings_array, all_labels)
        
        self.is_fitted = True
        
        stats = self.db.get_statistics()
        print(f"\n✓ Database built:")
        print(f"  Total: {stats['total_count']} embeddings")
        print(f"  Real: {stats['real_count']}")
        print(f"  Fake: {stats['fake_count']}")
        print(f"  Embedding dim: {stats['embedding_dim']}")
        
        return stats
    
    def fit_from_folder(
        self,
        real_folder: str,
        fake_folder: str,
        max_images: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Zbuduj bazę z folderów z obrazami.
        
        Args:
            real_folder: Folder z prawdziwymi obrazami
            fake_folder: Folder z fake obrazami
            max_images: Max obrazów per klasa
            batch_size: Batch size do przetwarzania
        """
        from torchvision import transforms
        
        print("\nLoading images from folders...")
        
        # Get image paths
        real_paths = list(Path(real_folder).glob("*.*"))
        fake_paths = list(Path(fake_folder).glob("*.*"))
        
        # Filter valid images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        real_paths = [p for p in real_paths if p.suffix.lower() in valid_extensions]
        fake_paths = [p for p in fake_paths if p.suffix.lower() in valid_extensions]
        
        # Limit if needed
        if max_images:
            real_paths = real_paths[:max_images]
            fake_paths = fake_paths[:max_images]
        
        print(f"  Found {len(real_paths)} real images")
        print(f"  Found {len(fake_paths)} fake images")
        
        # Process in batches
        all_embeddings = []
        all_labels = []
        
        for label, paths in [("real", real_paths), ("fake", fake_paths)]:
            print(f"\nProcessing {label} images...")
            
            for i in tqdm(range(0, len(paths), batch_size), desc=label):
                batch_paths = paths[i:i+batch_size]
                
                # Load images
                images = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        continue
                
                if not images:
                    continue
                
                # Encode
                embeddings = self.encoder.encode_batch(images, show_progress=False)
                
                all_embeddings.append(embeddings)
                all_labels.extend([label] * len(embeddings))
        
        # Add to database
        embeddings_array = np.vstack(all_embeddings)
        self.db.add(embeddings_array, all_labels)
        
        self.is_fitted = True
        
        return self.db.get_statistics()
    
    def predict(
        self,
        image: Image.Image,
        method: str = "knn",  # "knn" lub "centroid"
    ) -> DetectionResult:
        """
        Klasyfikuj pojedynczy obraz.
        
        Args:
            image: PIL Image
            method: "knn" (k-NN voting) lub "centroid" (distance to class centroids)
            
        Returns:
            DetectionResult
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first!")
        
        # Get embedding
        embedding = self.encoder.encode_pil(image)
        
        # Classify
        if method == "knn":
            result = self.db.classify_knn(embedding, k=self.config.k_neighbors)
        else:
            result = self.db.classify_centroid(embedding)
        
        return DetectionResult(
            prediction=result["prediction"],
            confidence=result["confidence"],
            method=method,
            details=result,
        )
    
    def predict_batch(
        self,
        images: List[Image.Image],
        method: str = "knn",
    ) -> List[DetectionResult]:
        """Klasyfikuj batch obrazów"""
        return [self.predict(img, method) for img in tqdm(images, desc="Predicting")]
    
    def evaluate(
        self,
        dataloader: DataLoader,
        methods: List[str] = ["knn", "centroid"],
    ) -> Dict[str, Dict[str, float]]:
        """
        Ewaluacja na testowym dataloaderze.
        
        Args:
            dataloader: DataLoader returning (images, labels)
            methods: Metody do przetestowania
            
        Returns:
            Dict z metrykami dla każdej metody
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        print("\nEvaluating detector...")
        
        # Get all embeddings
        print("Encoding test images...")
        embeddings, labels = self.encoder.encode_dataloader(dataloader)
        
        # Convert labels
        label_strs = ["real" if l == 1 else "fake" for l in labels]
        label_binary = labels  # 1=real, 0=fake
        
        results = {}
        
        for method in methods:
            print(f"\nMethod: {method}")
            
            predictions = []
            confidences = []
            
            for emb in tqdm(embeddings, desc="Classifying"):
                if method == "knn":
                    result = self.db.classify_knn(emb, k=self.config.k_neighbors)
                else:
                    result = self.db.classify_centroid(emb)
                
                predictions.append(result["prediction"])
                confidences.append(result["confidence"])
            
            # Convert to binary
            pred_binary = np.array([1 if p == "real" else 0 for p in predictions])
            
            # Compute metrics
            acc = accuracy_score(label_binary, pred_binary)
            prec = precision_score(label_binary, pred_binary, zero_division=0)
            rec = recall_score(label_binary, pred_binary, zero_division=0)
            f1 = f1_score(label_binary, pred_binary, zero_division=0)
            
            try:
                auc = roc_auc_score(label_binary, confidences)
            except:
                auc = 0.5
            
            cm = confusion_matrix(label_binary, pred_binary)
            
            results[method] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc,
                "confusion_matrix": cm.tolist(),
            }
            
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  AUC:       {auc:.4f}")
        
        return results
    
    def evaluate_from_folder(
        self,
        real_folder: str,
        fake_folder: str,
        max_images: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Ewaluacja z folderów"""
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        
        # Load images
        real_paths = list(Path(real_folder).glob("*.*"))[:max_images]
        fake_paths = list(Path(fake_folder).glob("*.*"))[:max_images]
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        real_paths = [p for p in real_paths if p.suffix.lower() in valid_extensions]
        fake_paths = [p for p in fake_paths if p.suffix.lower() in valid_extensions]
        
        print(f"\nEvaluating on {len(real_paths)} real + {len(fake_paths)} fake images")
        
        # Predict
        results = {"knn": {"correct": 0, "total": 0}, "centroid": {"correct": 0, "total": 0}}
        all_preds = {"knn": [], "centroid": []}
        all_labels = []
        
        for label, paths in [("real", real_paths), ("fake", fake_paths)]:
            for path in tqdm(paths, desc=f"Testing {label}"):
                try:
                    img = Image.open(path).convert("RGB")
                    
                    for method in ["knn", "centroid"]:
                        result = self.predict(img, method=method)
                        is_correct = result.prediction == label
                        results[method]["total"] += 1
                        if is_correct:
                            results[method]["correct"] += 1
                        
                        all_preds[method].append(1 if result.prediction == "real" else 0)
                    
                    all_labels.append(1 if label == "real" else 0)
                except:
                    continue
        
        # Calculate final metrics
        final_results = {}
        for method in ["knn", "centroid"]:
            acc = results[method]["correct"] / results[method]["total"]
            f1 = f1_score(all_labels, all_preds[method])
            cm = confusion_matrix(all_labels, all_preds[method])
            
            final_results[method] = {
                "accuracy": acc,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
            }
            
            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        
        return final_results
    
    def save(self, path: str) -> None:
        """Zapisz detektor"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save database
        self.db.save(str(path / "vector_db"))
        
        # Save config
        import json
        config_dict = {
            "encoder_name": self.config.encoder_name,
            "encoder_variant": self.config.encoder_variant,
            "k_neighbors": self.config.k_neighbors,
            "db_backend": self.config.db_backend,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Detector saved to {path}")
    
    def load(self, path: str) -> None:
        """Wczytaj detektor"""
        path = Path(path)
        
        # Load config
        import json
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        
        self.config = Stage1Config(**config_dict)
        
        # Load database
        self.db.load(str(path / "vector_db"))
        
        self.is_fitted = True
        print(f"✓ Detector loaded from {path}")


def run_stage1_experiment(
    train_real_folder: str,
    train_fake_folder: str,
    test_real_folder: str,
    test_fake_folder: str,
    max_train: int = 2000,
    max_test: int = 500,
    encoder_variant: str = "ViT-B/32",
    k_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Uruchom kompletny eksperyment Stage 1.
    
    Args:
        train_real_folder: Folder z train real
        train_fake_folder: Folder z train fake
        test_real_folder: Folder z test real
        test_fake_folder: Folder z test fake
        max_train: Max obrazów per klasa do treningu
        max_test: Max obrazów per klasa do testu
        encoder_variant: CLIP variant
        k_neighbors: K dla k-NN
        
    Returns:
        Dict z wynikami
    """
    config = Stage1Config(
        encoder_variant=encoder_variant,
        k_neighbors=k_neighbors,
    )
    
    detector = Stage1BaselineDetector(config)
    
    # Build database
    print("\n" + "="*50)
    print("BUILDING DATABASE (Training)")
    print("="*50)
    
    train_stats = detector.fit_from_folder(
        real_folder=train_real_folder,
        fake_folder=train_fake_folder,
        max_images=max_train,
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION (Testing)")
    print("="*50)
    
    test_results = detector.evaluate_from_folder(
        real_folder=test_real_folder,
        fake_folder=test_fake_folder,
        max_images=max_test,
    )
    
    return {
        "train_stats": train_stats,
        "test_results": test_results,
        "config": {
            "encoder_variant": encoder_variant,
            "k_neighbors": k_neighbors,
            "max_train": max_train,
            "max_test": max_test,
        }
    }
