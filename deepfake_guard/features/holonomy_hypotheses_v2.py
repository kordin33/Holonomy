"""
holonomy_hypotheses_v2.py - POPRAWIONE hipotezy holonomii

NAPRAWY:
1. H4: Prawdziwy coupling wektorowy (cosine similarity między patchami)
2. H1/H5: Twardy błąd gdy PCA nie fittowany (bez fallback do slice)
3. H5: G ma rozmiar r×r gdzie r = pca.n_components_
4. H2: Stała struktura pętli, tylko parametry zmieniają się monotonicznie
5. H3: Seed per-image (deterministyczny z hash obrazu)
6. H6: u = (z0 - μ_train) w PCA/whitened space
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats
import io
import hashlib


# ============================================================================
# DEGRADATION OPERATORS
# ============================================================================

def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB').copy()
    buffer.close()
    return img

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def downscale_upscale(image: Image.Image, scale_factor: float) -> Image.Image:
    w, h = image.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    downscaled = image.resize((new_w, new_h), Image.LANCZOS)
    return downscaled.resize((w, h), Image.LANCZOS)

def identity(image: Image.Image) -> Image.Image:
    return image.copy()


def make_transform(jpeg_q: int, blur_sigma: float):
    """Tworzy funkcję transformacji z parametrami."""
    def transform(img):
        img = jpeg_compression(img, jpeg_q)
        img = gaussian_blur(img, blur_sigma)
        return img
    return transform


BASE_LOOPS = [
    ['jpeg_70', 'blur_0.5', 'scale_0.9'],
    ['blur_0.5', 'jpeg_80', 'scale_0.75'],
    ['jpeg_60', 'blur_0.7', 'scale_0.9'],
]

TRANSFORMATIONS = {
    'jpeg_90': lambda img: jpeg_compression(img, 90),
    'jpeg_80': lambda img: jpeg_compression(img, 80),
    'jpeg_70': lambda img: jpeg_compression(img, 70),
    'jpeg_60': lambda img: jpeg_compression(img, 60),
    'jpeg_50': lambda img: jpeg_compression(img, 50),
    'blur_0.3': lambda img: gaussian_blur(img, 0.3),
    'blur_0.5': lambda img: gaussian_blur(img, 0.5),
    'blur_0.7': lambda img: gaussian_blur(img, 0.7),
    'blur_1.0': lambda img: gaussian_blur(img, 1.0),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
    'identity': identity,
}


# ============================================================================
# H1: HOLONOMY SPECTRUM (FIXED)
# ============================================================================

class H1_HolonomySpectrum:
    """
    Widmo niedomknięcia w PCA space.
    NAPRAWIONE: Twardy błąd gdy PCA nie fittowany.
    """
    
    def __init__(self, pca_dim: int = 32, top_k: int = 8):
        self.pca_dim = pca_dim
        self.top_k = top_k
        self.pca: Optional[PCA] = None
    
    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA na zbiorze embeddingów."""
        n_components = min(self.pca_dim, embeddings.shape[1], embeddings.shape[0])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
    
    def compute_features(self, encoder, image: Image.Image, loop: List[str]) -> Dict[str, float]:
        """Oblicza cechy spektralne dla jednej pętli."""
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca on TRAIN embeddings.")
        
        # Get embeddings
        images = [image]
        current = image
        for t_name in loop:
            current = TRANSFORMATIONS[t_name](current)
            images.append(current)
        
        embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
        
        z_0 = embeddings[0]
        z_end = embeddings[-1]
        
        # Project to PCA space
        z_0_pca = self.pca.transform(z_0.reshape(1, -1))[0]
        z_end_pca = self.pca.transform(z_end.reshape(1, -1))[0]
        
        # Holonomy vector in PCA space
        h = z_end_pca - z_0_pca
        r = len(h)
        
        # Features
        energy = float(np.linalg.norm(h))
        
        # Spectrum: sorted |h_i|
        h_abs = np.abs(h)
        h_sorted = np.sort(h_abs)[::-1]
        spectrum_topk = h_sorted[:min(self.top_k, r)].tolist()
        # Pad if needed
        while len(spectrum_topk) < self.top_k:
            spectrum_topk.append(0.0)
        
        # Directional entropy
        h_sq = h ** 2
        h_sq_sum = h_sq.sum() + 1e-10
        p = h_sq / h_sq_sum
        p = p[p > 1e-10]
        entropy = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0
        
        # Anisotropy
        anisotropy = float(h_sq.max() / h_sq_sum)
        
        return {
            'energy': energy,
            'spectrum_topk': spectrum_topk,
            'entropy': entropy,
            'anisotropy': anisotropy
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        features = []
        for loop in BASE_LOOPS:
            stats = self.compute_features(encoder, image, loop)
            features.append(stats['energy'])
            features.extend(stats['spectrum_topk'][:4])
            features.append(stats['entropy'])
            features.append(stats['anisotropy'])
        return np.array(features, dtype=np.float32)


# ============================================================================
# H2: AREA/SCALE LAW (FIXED)
# ============================================================================

class H2_AreaScaleLaw:
    """
    Wykładnik potęgowy holonomii.
    NAPRAWIONE: Stała struktura pętli, tylko parametry zmieniają się monotonicznie.
    """
    
    def __init__(self):
        # Stała struktura: jpeg -> blur, tylko parametry rosną
        self.scales = [
            (90, 0.3, 0.1),   # (jpeg_q, blur_sigma, area)
            (80, 0.5, 0.3),
            (70, 0.7, 0.5),
            (60, 1.0, 0.7),
            (50, 1.3, 1.0),
        ]
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        holonomies = []
        areas = []
        
        for jpeg_q, blur_sigma, area in self.scales:
            # Stała struktura: jpeg -> blur
            img1 = jpeg_compression(image, jpeg_q)
            img2 = gaussian_blur(img1, blur_sigma)
            
            images = [image, img1, img2]
            embeddings = encoder.encode_batch(images, batch_size=3, show_progress=False)
            
            H = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
            holonomies.append(H)
            areas.append(area)
        
        holonomies = np.array(holonomies)
        areas = np.array(areas)
        
        # Fit log H ≈ α log a + c
        log_H = np.log(holonomies + 1e-8)
        log_a = np.log(areas + 1e-8)
        
        from scipy.stats import linregress
        result = linregress(log_a, log_H)
        
        alpha = result.slope
        intercept = result.intercept
        r_squared = result.rvalue ** 2
        residuals = log_H - (alpha * log_a + intercept)
        residual_std = float(residuals.std())
        
        return {
            'alpha': float(alpha),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'residual_std': residual_std,
            'mean_H': float(holonomies.mean())
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['alpha'],
            stats['intercept'],
            stats['r_squared'],
            stats['residual_std'],
            stats['mean_H']
        ], dtype=np.float32)


# ============================================================================
# H3: HOLONOMY CONSISTENCY INDEX (FIXED)
# ============================================================================

class H3_ConsistencyIndex:
    """
    Stabilność holonomii pod losowaniem pętli.
    NAPRAWIONE: Seed per-image (deterministyczny z hash obrazu).
    """
    
    def __init__(self, n_samples: int = 20):
        self.n_samples = n_samples
        self.alphabet = [
            'jpeg_90', 'jpeg_80', 'jpeg_70', 'jpeg_60',
            'blur_0.3', 'blur_0.5', 'blur_0.7',
            'scale_0.75', 'scale_0.9',
        ]
    
    def _get_image_seed(self, image: Image.Image) -> int:
        """Generuje deterministyczny seed z obrazu."""
        # Hash z pierwszych 100 pikseli
        arr = np.array(image)[:10, :10].flatten()
        return int(hashlib.md5(arr.tobytes()).hexdigest()[:8], 16)
    
    def _generate_random_loops(self, rng: np.random.Generator) -> List[List[str]]:
        loops = []
        for _ in range(self.n_samples):
            length = rng.integers(3, 6)
            loop = [self.alphabet[i] for i in rng.choice(len(self.alphabet), size=length)]
            loops.append(loop)
        return loops
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        # Seed per-image!
        seed = self._get_image_seed(image)
        rng = np.random.default_rng(seed)
        loops = self._generate_random_loops(rng)
        
        holonomies = []
        
        for loop in loops:
            images = [image]
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            H = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
            holonomies.append(H)
        
        holonomies = np.array(holonomies)
        
        median = float(np.median(holonomies))
        iqr = float(np.percentile(holonomies, 75) - np.percentile(holonomies, 25))
        p90 = float(np.percentile(holonomies, 90))
        mean = float(holonomies.mean())
        std = float(holonomies.std())
        cv = std / (mean + 1e-8)
        
        return {
            'median': median,
            'iqr': iqr,
            'p90': p90,
            'mean': mean,
            'cv': cv
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['median'],
            stats['iqr'],
            stats['p90'],
            stats['cv']
        ], dtype=np.float32)


# ============================================================================
# H4: PATCH-COUPLED HOLONOMY (FIXED - WEKTOROWY COUPLING!)
# ============================================================================

class H4_PatchCoupled:
    """
    Korelacje holonomii między obszarami obrazu.
    NAPRAWIONE: Prawdziwy coupling wektorowy - cosine similarity między wektorami holonomii patchów.
    """
    
    def __init__(self, grid: int = 3, pca_dim: int = 32):
        self.grid = grid  # 3x3 = 9 patchy
        self.pca_dim = pca_dim
        self.pca: Optional[PCA] = None
    
    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA."""
        n_components = min(self.pca_dim, embeddings.shape[1], embeddings.shape[0])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca on TRAIN embeddings.")
        
        loop = BASE_LOOPS[0]  # Użyj jednej pętli
        
        w, h = image.size
        ps = min(w, h) // self.grid
        
        holonomy_vectors = []
        
        for gy in range(self.grid):
            for gx in range(self.grid):
                # Extract patch
                patch = image.crop((gx*ps, gy*ps, (gx+1)*ps, (gy+1)*ps))
                patch = patch.resize((112, 112), Image.LANCZOS)  # Resize dla encodera
                
                # Apply loop
                images = [patch]
                current = patch
                for t_name in loop:
                    current = TRANSFORMATIONS[t_name](current)
                    images.append(current)
                
                embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
                
                # Holonomy vector
                h_vec = embeddings[-1] - embeddings[0]
                
                # Project to PCA space
                h_vec_pca = self.pca.transform(h_vec.reshape(1, -1))[0]
                
                # Normalize (kierunek, nie siła)
                h_vec_norm = h_vec_pca / (np.linalg.norm(h_vec_pca) + 1e-8)
                
                holonomy_vectors.append(h_vec_norm)
        
        H = np.stack(holonomy_vectors)  # (P, r)
        
        # Macierz podobieństw cosinusowych
        S = H @ H.T  # (P, P)
        
        # Średnia poza przekątną = coupling
        n_patches = S.shape[0]
        mask = ~np.eye(n_patches, dtype=bool)
        off_diagonal = S[mask]
        
        mean_coupling = float(off_diagonal.mean())
        std_coupling = float(off_diagonal.std())
        max_coupling = float(off_diagonal.max())
        min_coupling = float(off_diagonal.min())
        
        return {
            'mean_coupling': mean_coupling,
            'std_coupling': std_coupling,
            'max_coupling': max_coupling,
            'min_coupling': min_coupling
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['mean_coupling'],
            stats['std_coupling'],
            stats['max_coupling'],
            stats['min_coupling']
        ], dtype=np.float32)


# ============================================================================
# H5: NON-ABELIAN HOLONOMY TENSOR (FIXED)
# ============================================================================

class H5_HolonomyTensor:
    """
    Macierz Grama z commutatorów wektorowych.
    NAPRAWIONE: G ma rozmiar r×r gdzie r = pca.n_components_, twardy błąd bez PCA.
    """
    
    def __init__(self, pca_dim: int = 32):
        self.pca_dim = pca_dim
        self.pca: Optional[PCA] = None
        
        self.pairs = [
            ('jpeg_70', 'blur_0.5'),
            ('jpeg_80', 'scale_0.75'),
            ('blur_0.5', 'scale_0.9'),
            ('jpeg_60', 'blur_0.7'),
            ('blur_0.7', 'jpeg_70'),
        ]
    
    def fit_pca(self, embeddings: np.ndarray):
        n_components = min(self.pca_dim, embeddings.shape[1], embeddings.shape[0])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca on TRAIN embeddings.")
        
        r = self.pca.n_components_
        commutator_vectors = []
        
        for a_name, b_name in self.pairs:
            a = TRANSFORMATIONS[a_name]
            b = TRANSFORMATIONS[b_name]
            
            img_ab = a(b(image))
            img_ba = b(a(image))
            
            embeddings = encoder.encode_batch([img_ab, img_ba], batch_size=2, show_progress=False)
            
            z_ab = self.pca.transform(embeddings[0].reshape(1, -1))[0]
            z_ba = self.pca.transform(embeddings[1].reshape(1, -1))[0]
            
            c = z_ab - z_ba
            commutator_vectors.append(c)
        
        # Gram matrix G = Σ c_i c_i^T
        G = np.zeros((r, r))
        for c in commutator_vectors:
            G += np.outer(c, c)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(G)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        # Features
        trace = float(np.trace(G))
        top_eig = eigenvalues[:4].tolist()
        while len(top_eig) < 4:
            top_eig.append(0.0)
        
        # Spectral entropy
        eig_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
        eig_norm = eig_norm[eig_norm > 1e-10]
        spectral_entropy = float(-np.sum(eig_norm * np.log(eig_norm))) if len(eig_norm) > 0 else 0.0
        
        return {
            'trace': trace,
            'top_eigenvalues': top_eig,
            'spectral_entropy': spectral_entropy
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        stats = self.compute_features(encoder, image)
        features = [stats['trace']] + stats['top_eigenvalues'] + [stats['spectral_entropy']]
        return np.array(features, dtype=np.float32)


# ============================================================================
# H6: LOOP-TO-EMBEDDING ORTHOGONALITY (FIXED)
# ============================================================================

class H6_Orthogonality:
    """
    Kąt między wektorem holonomii h a wektorem "treści" u.
    NAPRAWIONE: u = (z0 - μ_train) w PCA/whitened space.
    """
    
    def __init__(self, pca_dim: int = 32):
        self.pca_dim = pca_dim
        self.pca: Optional[PCA] = None
        self.mean_embedding: Optional[np.ndarray] = None
    
    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA i zapisuje mean embeddingu."""
        n_components = min(self.pca_dim, embeddings.shape[1], embeddings.shape[0])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
        
        # Zapisz mean w PCA space
        self.mean_embedding = self.pca.transform(embeddings).mean(axis=0)
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        if self.pca is None or self.mean_embedding is None:
            raise RuntimeError("PCA not fitted. Call fit_pca on TRAIN embeddings.")
        
        angles = []
        
        for loop in BASE_LOOPS:
            images = [image]
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            
            # Project to PCA space
            z_0_pca = self.pca.transform(embeddings[0].reshape(1, -1))[0]
            z_end_pca = self.pca.transform(embeddings[-1].reshape(1, -1))[0]
            
            # u = (z0 - μ_train) - wektor treści względem średniej
            u = z_0_pca - self.mean_embedding
            
            # h = holonomy vector
            h = z_end_pca - z_0_pca
            
            # Angle
            dot = np.dot(h, u)
            norm_h = np.linalg.norm(h)
            norm_u = np.linalg.norm(u)
            
            cos_theta = dot / (norm_h * norm_u + 1e-8)
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = float(np.arccos(cos_theta))
            
            angles.append(theta)
        
        angles = np.array(angles)
        
        return {
            'mean_angle': float(angles.mean()),
            'std_angle': float(angles.std()),
            'min_angle': float(angles.min()),
            'max_angle': float(angles.max())
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['mean_angle'],
            stats['std_angle'],
            stats['min_angle'],
            stats['max_angle']
        ], dtype=np.float32)
