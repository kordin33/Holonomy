"""
visualization.py - Wizualizacja Embeddings

t-SNE, UMAP, clustering visualization, decision boundaries
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class EmbeddingVisualizer:
    """
    Wizualizacja embeddings dla deepfake detection.
    
    Features:
    - t-SNE / UMAP projection
    - Cluster visualization
    - Decision boundaries
    - Similarity heatmaps
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed. Visualization disabled.")
    
    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Visualization of Embeddings",
        save_path: Optional[str] = None,
        perplexity: int = 30,
        highlight_indices: Optional[List[int]] = None,
    ) -> None:
        """
        t-SNE visualization.
        
        Args:
            embeddings: [N, dim] array
            labels: [N] array of "real"/"fake" or 0/1
            title: Plot title
            save_path: Path to save figure
            perplexity: t-SNE perplexity
            highlight_indices: Indices to highlight (e.g., test samples)
        """
        if not HAS_MATPLOTLIB:
            return
        
        from sklearn.manifold import TSNE
        
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        self._plot_2d(
            embeddings_2d, labels, title, save_path, highlight_indices
        )
    
    def plot_umap(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "UMAP Visualization of Embeddings",
        save_path: Optional[str] = None,
        n_neighbors: int = 15,
        highlight_indices: Optional[List[int]] = None,
    ) -> None:
        """
        UMAP visualization.
        """
        if not HAS_MATPLOTLIB:
            return
        
        try:
            import umap
        except ImportError:
            print("UMAP not installed. Run: pip install umap-learn")
            return
        
        print("Computing UMAP...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        self._plot_2d(
            embeddings_2d, labels, title, save_path, highlight_indices
        )
    
    def _plot_2d(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        title: str,
        save_path: Optional[str],
        highlight_indices: Optional[List[int]] = None,
    ) -> None:
        """Internal 2D plotting"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert labels to binary if needed
        if isinstance(labels[0], str):
            label_binary = np.array([1 if l == "real" else 0 for l in labels])
        else:
            label_binary = labels
        
        # Colors
        colors = np.where(label_binary == 1, '#2ecc71', '#e74c3c')  # Green=real, Red=fake
        
        # Plot
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            alpha=0.6,
            s=20,
        )
        
        # Highlight specific points
        if highlight_indices:
            ax.scatter(
                embeddings_2d[highlight_indices, 0],
                embeddings_2d[highlight_indices, 1],
                c='blue',
                s=100,
                marker='*',
                edgecolors='white',
                linewidths=1,
                label='Query',
            )
        
        # Legend
        real_patch = mpatches.Patch(color='#2ecc71', label='Real')
        fake_patch = mpatches.Patch(color='#e74c3c', label='Fake')
        ax.legend(handles=[real_patch, fake_patch], loc='upper right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        # Style
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_similarity_heatmap(
        self,
        query_embeddings: np.ndarray,
        db_embeddings: np.ndarray,
        db_labels: np.ndarray,
        query_names: Optional[List[str]] = None,
        title: str = "Similarity Heatmap",
        save_path: Optional[str] = None,
        top_k: int = 20,
    ) -> None:
        """
        Heatmap of similarities between queries and database.
        """
        if not HAS_MATPLOTLIB:
            return
        
        import seaborn as sns
        
        # Normalize
        query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = query_norm @ db_norm.T
        
        # Get top-k most similar for each query
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:]
        
        # Create heatmap data
        heatmap_data = np.zeros((len(query_embeddings), top_k))
        for i in range(len(query_embeddings)):
            heatmap_data[i] = similarities[i, top_indices[i]]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(
            heatmap_data,
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        
        if query_names:
            ax.set_yticklabels(query_names, rotation=0)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Nearest Neighbors (sorted by similarity)')
        ax.set_ylabel('Query Images')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_cluster_analysis(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "Cluster Analysis",
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Analiza klasteryzacji - jak dobrze separują się klasy?
        """
        if not HAS_MATPLOTLIB:
            return {}
        
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        # Convert labels
        if isinstance(labels[0], str):
            label_binary = np.array([1 if l == "real" else 0 for l in labels])
        else:
            label_binary = labels
        
        # Compute class centroids
        real_mask = label_binary == 1
        fake_mask = label_binary == 0
        
        real_centroid = embeddings[real_mask].mean(axis=0)
        fake_centroid = embeddings[fake_mask].mean(axis=0)
        
        # Inter-class distance
        inter_class_dist = np.linalg.norm(real_centroid - fake_centroid)
        
        # Intra-class distances (average distance to centroid)
        real_intra = np.mean(np.linalg.norm(embeddings[real_mask] - real_centroid, axis=1))
        fake_intra = np.mean(np.linalg.norm(embeddings[fake_mask] - fake_centroid, axis=1))
        
        # Silhouette score
        silhouette = silhouette_score(embeddings, label_binary)
        
        # K-means clustering accuracy
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Match clusters to labels
        cluster_0_real = (cluster_labels[real_mask] == 0).sum()
        cluster_1_real = (cluster_labels[real_mask] == 1).sum()
        
        if cluster_0_real > cluster_1_real:
            kmeans_pred = 1 - cluster_labels
        else:
            kmeans_pred = cluster_labels
        
        kmeans_acc = (kmeans_pred == label_binary).mean()
        
        metrics = {
            "inter_class_distance": inter_class_dist,
            "real_intra_class_distance": real_intra,
            "fake_intra_class_distance": fake_intra,
            "silhouette_score": silhouette,
            "kmeans_accuracy": kmeans_acc,
            "separation_ratio": inter_class_dist / ((real_intra + fake_intra) / 2),
        }
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: t-SNE
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)
        
        colors = np.where(label_binary == 1, '#2ecc71', '#e74c3c')
        axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.6, s=20)
        axes[0].set_title('t-SNE Projection', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Right: Metrics
        metric_names = ['Silhouette', 'KMeans Acc', 'Separation\nRatio']
        metric_values = [silhouette, kmeans_acc, min(metrics['separation_ratio'], 2)]
        
        bars = axes[1].bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#9b59b6'])
        axes[1].set_ylim(0, max(metric_values) * 1.2)
        axes[1].set_title('Clustering Metrics', fontsize=12)
        
        for bar, val in zip(bars, [silhouette, kmeans_acc, metrics['separation_ratio']]):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return metrics
    
    def plot_knn_explanation(
        self,
        query_embedding: np.ndarray,
        db_embeddings: np.ndarray,
        db_labels: np.ndarray,
        prediction: str,
        k: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Wizualizacja wyjaśnienia decyzji k-NN.
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Compute similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
        similarities = db_norm @ query_norm
        
        # Get top-k
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        top_k_sim = similarities[top_k_idx]
        top_k_labels = db_labels[top_k_idx]
        
        # Count
        real_count = sum(1 for l in top_k_labels if l == "real" or l == 1)
        fake_count = k - real_count
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Similarity bars
        colors = ['#2ecc71' if l == "real" or l == 1 else '#e74c3c' for l in top_k_labels]
        bars = axes[0].barh(range(k), top_k_sim, color=colors)
        axes[0].set_yticks(range(k))
        axes[0].set_yticklabels([f'#{i+1}' for i in range(k)])
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_title(f'Top-{k} Nearest Neighbors')
        axes[0].invert_yaxis()
        
        # Right: Voting pie
        axes[1].pie(
            [real_count, fake_count],
            labels=['Real', 'Fake'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.0f%%',
            startangle=90,
            explode=(0.05, 0.05),
        )
        axes[1].set_title(f'Prediction: {prediction.upper()}')
        
        plt.suptitle('k-NN Classification Explanation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
