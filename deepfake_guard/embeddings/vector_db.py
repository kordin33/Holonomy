"""
vector_db.py - Vector Database for Embedding Storage & Retrieval
=================================================================

Multi-backend vector database for storing and querying image embeddings.
Supports fast similarity search for k-NN based deepfake classification.

BACKENDS:
---------
    1. NumPy (default): Pure Python, no dependencies, works everywhere
    2. ChromaDB: Easy to use, persistent storage, good for prototyping
    3. FAISS: Fastest for large datasets (100K+ embeddings)

USAGE:
------
    from deepfake_guard.embeddings.vector_db import DeepfakeVectorDB
    
    # Create database
    db = DeepfakeVectorDB(backend="numpy")
    
    # Add embeddings with labels
    db.add(embeddings, labels=["real", "fake", "fake", ...])
    
    # Query nearest neighbors
    results = db.query(query_embedding, k=10)
    
    # k-NN classification
    prediction = db.classify_knn(query_embedding, k=10)
    # Returns: {'prediction': 'real', 'confidence': 0.87, ...}
    
    # Centroid-based classification
    prediction = db.classify_centroid(query_embedding)

FEATURES:
---------
    - Cosine similarity search (L2 normalized embeddings)
    - Weighted k-NN voting (similarity-weighted)
    - Centroid-based classification
    - Database persistence (save/load)
    - Statistics and analysis

AUTHOR: Konrad Kenczuk
VERSION: 1.0.0
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import json


class VectorDatabase:
    """
    Wrapper dla bazy wektorowej.
    
    Wspiera:
    - ChromaDB (domyślny, łatwy w użyciu)
    - FAISS (szybszy dla dużych datasetów)
    - Numpy (fallback, wolny ale zawsze działa)
    """
    
    def __init__(
        self,
        backend: str = "numpy",  # "chromadb", "faiss", "numpy"
        collection_name: str = "deepfake_embeddings",
        persist_dir: Optional[str] = None,
    ):
        self.backend = backend
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        
        self._init_backend()
    
    def _init_backend(self):
        if self.backend == "chromadb":
            self._init_chromadb()
        elif self.backend == "faiss":
            self._init_faiss()
        else:
            self._init_numpy()
    
    def _init_chromadb(self):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            print("ChromaDB not installed. Using numpy backend.")
            self.backend = "numpy"
            self._init_numpy()
            return
        
        if self.persist_dir:
            self.client = chromadb.Client(Settings(
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
            ))
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        print(f"✓ ChromaDB collection: {self.collection_name}")
    
    def _init_faiss(self):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            print("FAISS not installed. Using numpy backend.")
            self.backend = "numpy"
            self._init_numpy()
            return
        
        self.index = None
        self.metadata = []
        print("✓ FAISS backend initialized")
    
    def _init_numpy(self):
        self.embeddings = []
        self.metadata = []
        self.ids = []
        print("✓ Numpy backend initialized")
    
    def add(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Dodaj embeddingi do bazy.
        
        Args:
            embeddings: [N, dim] array
            labels: Lista etykiet ("real" lub "fake")
            ids: Opcjonalne ID każdego embedingu
            metadata: Opcjonalne dodatkowe metadane
        """
        n = len(embeddings)
        
        if ids is None:
            existing_count = self.count()
            ids = [f"emb_{existing_count + i}" for i in range(n)]
        
        if metadata is None:
            metadata = [{"label": label} for label in labels]
        else:
            for i, label in enumerate(labels):
                metadata[i]["label"] = label
        
        if self.backend == "chromadb":
            self.collection.add(
                embeddings=embeddings.tolist(),
                metadatas=metadata,
                ids=ids,
            )
        
        elif self.backend == "faiss":
            if self.index is None:
                dim = embeddings.shape[1]
                self.index = self.faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / (norms + 1e-8)
            self.index.add(embeddings_norm.astype(np.float32))
            self.metadata.extend(metadata)
        
        else:  # numpy
            self.embeddings.extend(embeddings.tolist())
            self.metadata.extend(metadata)
            self.ids.extend(ids)
    
    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Znajdź k najbliższych sąsiadów.
        
        Args:
            query_embedding: [dim] lub [1, dim]
            k: Liczba sąsiadów
            
        Returns:
            Dict z keys: distances, metadatas, ids
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if self.backend == "chromadb":
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
            )
            return {
                "distances": results["distances"][0],
                "metadatas": results["metadatas"][0],
                "ids": results["ids"][0],
            }
        
        elif self.backend == "faiss":
            # Normalize query
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            distances, indices = self.index.search(query_norm.astype(np.float32), k)
            
            return {
                "distances": (1 - distances[0]).tolist(),  # Convert similarity to distance
                "metadatas": [self.metadata[i] for i in indices[0]],
                "ids": [f"emb_{i}" for i in indices[0]],
            }
        
        else:  # numpy
            embeddings_array = np.array(self.embeddings)
            
            # Cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            emb_norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_norm = embeddings_array / (emb_norms + 1e-8)
            
            similarities = np.dot(embeddings_norm, query_norm.T).flatten()
            
            # Get top k
            top_indices = np.argsort(similarities)[::-1][:k]
            
            return {
                "distances": (1 - similarities[top_indices]).tolist(),
                "metadatas": [self.metadata[i] for i in top_indices],
                "ids": [self.ids[i] for i in top_indices],
                "similarities": similarities[top_indices].tolist(),
            }
    
    def count(self) -> int:
        """Liczba embeddingów w bazie"""
        if self.backend == "chromadb":
            return self.collection.count()
        elif self.backend == "faiss":
            return self.index.ntotal if self.index else 0
        else:
            return len(self.embeddings)
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pobierz wszystkie embeddingi i labels"""
        if self.backend == "chromadb":
            results = self.collection.get(include=["embeddings", "metadatas"])
            embeddings = np.array(results["embeddings"])
            labels = np.array([m["label"] for m in results["metadatas"]])
            return embeddings, labels
        
        elif self.backend == "faiss":
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            labels = np.array([m["label"] for m in self.metadata])
            return embeddings, labels
        
        else:
            embeddings = np.array(self.embeddings)
            labels = np.array([m["label"] for m in self.metadata])
            return embeddings, labels
    
    def save(self, path: str) -> None:
        """Zapisz bazę do pliku"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.backend == "numpy":
            data = {
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "ids": self.ids,
            }
            np.save(str(path.with_suffix('.npy')), data, allow_pickle=True)
        
        print(f"✓ Database saved to {path}")
    
    def load(self, path: str) -> None:
        """Wczytaj bazę z pliku"""
        path = Path(path)
        
        if self.backend == "numpy":
            data = np.load(str(path.with_suffix('.npy')), allow_pickle=True).item()
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
            self.ids = data["ids"]
        
        print(f"✓ Database loaded from {path}")


class DeepfakeVectorDB(VectorDatabase):
    """
    Specjalized Vector DB dla deepfake detection.
    
    Dodatkowe metody do klasyfikacji i analizy.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.real_count = 0
        self.fake_count = 0
    
    def add(self, embeddings, labels, **kwargs):
        """Dodaj i zlicz real/fake"""
        super().add(embeddings, labels, **kwargs)
        
        for label in labels:
            if label == "real":
                self.real_count += 1
            else:
                self.fake_count += 1
    
    def classify_knn(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Klasyfikacja przez k-NN voting.
        
        Args:
            query_embedding: Embedding do klasyfikacji
            k: Liczba sąsiadów
            
        Returns:
            Dict z prediction i confidence
        """
        results = self.query(query_embedding, k=k)
        
        # Count votes
        labels = [m["label"] for m in results["metadatas"]]
        real_votes = sum(1 for l in labels if l == "real")
        fake_votes = k - real_votes
        
        # Weighted voting (bliżsi sąsiedzi = więcej głosów)
        if "similarities" in results:
            similarities = results["similarities"]
            real_weight = sum(s for s, l in zip(similarities, labels) if l == "real")
            fake_weight = sum(s for s, l in zip(similarities, labels) if l == "fake")
        else:
            real_weight = real_votes
            fake_weight = fake_votes
        
        total_weight = real_weight + fake_weight + 1e-8
        
        prediction = "real" if real_weight > fake_weight else "fake"
        confidence = max(real_weight, fake_weight) / total_weight
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "real_votes": real_votes,
            "fake_votes": fake_votes,
            "real_weight": real_weight,
            "fake_weight": fake_weight,
            "neighbors": results,
        }
    
    def classify_centroid(
        self,
        query_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Klasyfikacja przez odległość do centroidów klas.
        
        Returns:
            Dict z prediction i distances
        """
        embeddings, labels = self.get_all_embeddings()
        
        # Compute centroids
        real_mask = labels == "real"
        fake_mask = labels == "fake"
        
        real_centroid = embeddings[real_mask].mean(axis=0)
        fake_centroid = embeddings[fake_mask].mean(axis=0)
        
        # Query distances
        query_norm = query_embedding.flatten()
        query_norm = query_norm / (np.linalg.norm(query_norm) + 1e-8)
        
        real_sim = np.dot(real_centroid / np.linalg.norm(real_centroid), query_norm)
        fake_sim = np.dot(fake_centroid / np.linalg.norm(fake_centroid), query_norm)
        
        prediction = "real" if real_sim > fake_sim else "fake"
        
        return {
            "prediction": prediction,
            "real_similarity": float(real_sim),
            "fake_similarity": float(fake_sim),
            "confidence": abs(real_sim - fake_sim),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statystyki bazy"""
        embeddings, labels = self.get_all_embeddings()
        
        real_mask = labels == "real"
        fake_mask = labels == "fake"
        
        return {
            "total_count": len(embeddings),
            "real_count": real_mask.sum(),
            "fake_count": fake_mask.sum(),
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        }
