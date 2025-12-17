"""
holonomy_hypotheses.py - 6 nowych hipotez holonomii

H1: Holonomy Spectrum - widmo niedomknięcia w PCA space
H2: Area/Scale Law - wykładnik potęgowy holonomii
H3: Holonomy Consistency Index - stabilność pod losowaniem pętli
H4: Patch-Coupled Holonomy - korelacje między obszarami
H5: Non-Abelian Holonomy Tensor - macierz Grama z commutatorów
H6: Loop-to-Embedding Orthogonality - kąt h vs u
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats
import io


# ============================================================================
# DEGRADATION OPERATORS (reuse from baseline)
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

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(factor)

def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    img_array = np.array(image).astype(np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    return Image.fromarray((corrected * 255).astype(np.uint8))

def identity(image: Image.Image) -> Image.Image:
    return image.copy()


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
    'scale_0.5': lambda img: downscale_upscale(img, 0.5),
    'scale_0.75': lambda img: downscale_upscale(img, 0.75),
    'scale_0.9': lambda img: downscale_upscale(img, 0.9),
    'sharpen_1.5': lambda img: sharpen(img, 1.5),
    'sharpen_2.0': lambda img: sharpen(img, 2.0),
    'gamma_1.2': lambda img: adjust_gamma(img, 1.2),
    'gamma_1.5': lambda img: adjust_gamma(img, 1.5),
    'identity': identity,
}

BASE_LOOPS = [
    ['scale_0.9', 'blur_0.7', 'jpeg_70', 'scale_0.9'],
    ['blur_0.5', 'jpeg_70', 'blur_0.3', 'identity'],
    ['jpeg_80', 'blur_0.3', 'jpeg_60', 'blur_0.5'],
]


# ============================================================================
# H1: HOLONOMY SPECTRUM
# ============================================================================

class H1_HolonomySpectrum:
    """
    Widmo niedomknięcia w zredukowanej przestrzeni PCA.
    
    Cechy: energia, top-k spectrum, entropia kierunkowa, anisotropy
    """
    
    def __init__(self, pca_dim: int = 32, top_k: int = 8):
        self.pca_dim = pca_dim
        self.top_k = top_k
        self.pca = None
    
    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA na zbiorze embeddingów."""
        self.pca = PCA(n_components=min(self.pca_dim, embeddings.shape[1]))
        self.pca.fit(embeddings)
    
    def compute_features(self, encoder, image: Image.Image, loop: List[str]) -> Dict[str, float]:
        """Oblicza cechy spektralne dla jednej pętli."""
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
        if self.pca is not None:
            z_0_pca = self.pca.transform(z_0.reshape(1, -1))[0]
            z_end_pca = self.pca.transform(z_end.reshape(1, -1))[0]
        else:
            z_0_pca = z_0[:self.pca_dim]
            z_end_pca = z_end[:self.pca_dim]
        
        # Holonomy vector in PCA space
        h = z_end_pca - z_0_pca
        
        # Features
        energy = float(np.linalg.norm(h))
        
        # Spectrum: sorted |h_i|
        h_abs = np.abs(h)
        h_sorted = np.sort(h_abs)[::-1]
        spectrum_topk = h_sorted[:self.top_k].tolist()
        
        # Directional entropy
        h_sq = h ** 2
        h_sq_sum = h_sq.sum() + 1e-10
        p = h_sq / h_sq_sum
        p = p[p > 1e-10]  # avoid log(0)
        entropy = float(-np.sum(p * np.log(p)))
        
        # Anisotropy
        anisotropy = float(h_sq.max() / h_sq_sum)
        
        return {
            'energy': energy,
            'spectrum_topk': spectrum_topk,
            'entropy': entropy,
            'anisotropy': anisotropy
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy dla obrazu (wszystkie pętle)."""
        features = []
        
        for loop in BASE_LOOPS:
            stats = self.compute_features(encoder, image, loop)
            features.append(stats['energy'])
            features.extend(stats['spectrum_topk'][:4])  # top-4
            features.append(stats['entropy'])
            features.append(stats['anisotropy'])
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# H2: AREA/SCALE LAW
# ============================================================================

class H2_AreaScaleLaw:
    """
    Wykładnik potęgowy holonomii przy różnych intensywnościach.
    
    log H(a) ≈ α log a + c
    
    Cechy: α (exponent), c (intercept), R² (fit quality), residual_std
    """
    
    def __init__(self):
        # Pętle w różnych intensywnościach
        self.scales = [
            (['jpeg_90', 'blur_0.3'], 0.1),
            (['jpeg_80', 'blur_0.5'], 0.3),
            (['jpeg_70', 'blur_0.7'], 0.5),
            (['jpeg_60', 'blur_1.0'], 0.7),
            (['jpeg_50', 'scale_0.75', 'blur_1.0'], 1.0),
        ]
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """Oblicza prawo skali."""
        holonomies = []
        areas = []
        
        for loop, area in self.scales:
            # Compute holonomy
            images = [image]
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            H = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
            
            holonomies.append(H)
            areas.append(area)
        
        holonomies = np.array(holonomies)
        areas = np.array(areas)
        
        # Fit log H ≈ α log a + c
        log_H = np.log(holonomies + 1e-8)
        log_a = np.log(areas + 1e-8)
        
        # Linear regression
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
        """Ekstraktuje cechy."""
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['alpha'],
            stats['intercept'],
            stats['r_squared'],
            stats['residual_std'],
            stats['mean_H']
        ], dtype=np.float32)


# ============================================================================
# H3: HOLONOMY CONSISTENCY INDEX
# ============================================================================

class H3_ConsistencyIndex:
    """
    Stabilność holonomii pod losowaniem pętli.
    
    Cechy: median, IQR, p90, CV (coefficient of variation)
    """
    
    def __init__(self, n_samples: int = 20, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        
        # Alfabet operacji
        self.alphabet = [
            'jpeg_90', 'jpeg_80', 'jpeg_70', 'jpeg_60',
            'blur_0.3', 'blur_0.5', 'blur_0.7',
            'scale_0.75', 'scale_0.9',
            'sharpen_1.5', 'gamma_1.2'
        ]
    
    def _generate_random_loops(self, rng: np.random.Generator) -> List[List[str]]:
        """Generuje losowe pętle."""
        loops = []
        for _ in range(self.n_samples):
            length = rng.integers(3, 6)  # 3-5 kroków
            loop = [self.alphabet[i] for i in rng.choice(len(self.alphabet), size=length)]
            loops.append(loop)
        return loops
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """Oblicza statystyki stabilności."""
        rng = np.random.default_rng(self.seed)
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
        cv = std / (mean + 1e-8)  # Coefficient of variation
        
        return {
            'median': median,
            'iqr': iqr,
            'p90': p90,
            'mean': mean,
            'cv': cv
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy."""
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['median'],
            stats['iqr'],
            stats['p90'],
            stats['cv']
        ], dtype=np.float32)


# ============================================================================
# H4: PATCH-COUPLED HOLONOMY
# ============================================================================

class H4_PatchCoupled:
    """
    Korelacje holonomii między obszarami obrazu.
    
    Cechy: mean_corr, max_corr, spectral_entropy
    """
    
    def __init__(self, n_patches: int = 9, patch_size: int = 96, seed: int = 42):
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.seed = seed
    
    def _extract_patches(self, image: Image.Image, rng: np.random.Generator) -> List[Image.Image]:
        """Ekstraktuje losowe patche."""
        w, h = image.size
        patches = []
        
        for _ in range(self.n_patches):
            if w >= self.patch_size and h >= self.patch_size:
                x = rng.integers(0, w - self.patch_size + 1)
                y = rng.integers(0, h - self.patch_size + 1)
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
            else:
                patch = image.resize((self.patch_size, self.patch_size), Image.LANCZOS)
            patches.append(patch)
        
        return patches
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """Oblicza korelacje między patchami."""
        rng = np.random.default_rng(self.seed)
        patches = self._extract_patches(image, rng)
        loop = BASE_LOOPS[0]  # Użyj jednej pętli
        
        # Holonomy dla każdego patcha
        patch_holonomies = []
        
        for patch in patches:
            images = [patch]
            current = patch
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            H = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
            patch_holonomies.append(H)
        
        patch_holonomies = np.array(patch_holonomies)
        
        # Correlation matrix (dla skalów, użyjemy outer product)
        # Dla prostoty: korelacja między wartościami H
        if len(patch_holonomies) > 1 and patch_holonomies.std() > 1e-8:
            # Oblicz "coupling" jako variance ratio
            mean_H = patch_holonomies.mean()
            std_H = patch_holonomies.std()
            cv = std_H / (mean_H + 1e-8)
            
            # Pseudo-correlation: jak bardzo patche są "jednakowe"
            # Wysoki CV = duża różnorodność (real?)
            # Niski CV = podobne holonomie (fake?)
            
            # Spectral features
            sorted_H = np.sort(patch_holonomies)[::-1]
            spectral_entropy = float(-np.sum((sorted_H/sorted_H.sum() + 1e-10) * 
                                             np.log(sorted_H/sorted_H.sum() + 1e-10)))
        else:
            cv = 0.0
            spectral_entropy = 0.0
        
        return {
            'mean_H': float(patch_holonomies.mean()),
            'std_H': float(patch_holonomies.std()),
            'cv': float(cv),
            'spectral_entropy': spectral_entropy
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy."""
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['mean_H'],
            stats['std_H'],
            stats['cv'],
            stats['spectral_entropy']
        ], dtype=np.float32)


# ============================================================================
# H5: NON-ABELIAN HOLONOMY TENSOR
# ============================================================================

class H5_HolonomyTensor:
    """
    Macierz Grama z commutatorów wektorowych.
    
    G = Σ c_i c_i^T
    
    Cechy: trace(G), top eigenvalues, spectral_entropy
    """
    
    def __init__(self, pca_dim: int = 32):
        self.pca_dim = pca_dim
        self.pca = None
        
        # Pary operacji
        self.pairs = [
            ('jpeg_70', 'blur_0.5'),
            ('jpeg_80', 'scale_0.75'),
            ('blur_0.5', 'scale_0.9'),
            ('sharpen_1.5', 'jpeg_70'),
            ('gamma_1.2', 'blur_0.5'),
        ]
    
    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA."""
        self.pca = PCA(n_components=min(self.pca_dim, embeddings.shape[1]))
        self.pca.fit(embeddings)
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """Oblicza tensor krzywizny."""
        commutator_vectors = []
        
        for a_name, b_name in self.pairs:
            a = TRANSFORMATIONS[a_name]
            b = TRANSFORMATIONS[b_name]
            
            # a(b(x)) i b(a(x))
            img_ab = a(b(image))
            img_ba = b(a(image))
            
            embeddings = encoder.encode_batch([img_ab, img_ba], batch_size=2, show_progress=False)
            
            # Commutator vector
            if self.pca is not None:
                z_ab = self.pca.transform(embeddings[0].reshape(1, -1))[0]
                z_ba = self.pca.transform(embeddings[1].reshape(1, -1))[0]
            else:
                z_ab = embeddings[0][:self.pca_dim]
                z_ba = embeddings[1][:self.pca_dim]
            
            c = z_ab - z_ba
            commutator_vectors.append(c)
        
        commutator_vectors = np.array(commutator_vectors)
        
        # Gram matrix G = Σ c_i c_i^T
        G = np.zeros((self.pca_dim, self.pca_dim))
        for c in commutator_vectors:
            G += np.outer(c, c)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(G)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Features
        trace = float(np.trace(G))
        top_eig = eigenvalues[:4].tolist()
        
        # Spectral entropy
        eig_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
        eig_norm = eig_norm[eig_norm > 1e-10]
        spectral_entropy = float(-np.sum(eig_norm * np.log(eig_norm)))
        
        return {
            'trace': trace,
            'top_eigenvalues': top_eig,
            'spectral_entropy': spectral_entropy
        }
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        """Ekstraktuje cechy."""
        stats = self.compute_features(encoder, image)
        features = [stats['trace']] + stats['top_eigenvalues'] + [stats['spectral_entropy']]
        return np.array(features, dtype=np.float32)


# ============================================================================
# H6: LOOP-TO-EMBEDDING ORTHOGONALITY
# ============================================================================

class H6_Orthogonality:
    """
    Kąt między wektorem holonomii h a wektorem "treści" u.
    
    θ = arccos(<h, u> / (||h|| ||u|| + ε))
    """
    
    def compute_features(self, encoder, image: Image.Image) -> Dict[str, float]:
        """Oblicza kąty dla kilku pętli."""
        angles = []
        
        for loop in BASE_LOOPS:
            images = [image]
            current = image
            for t_name in loop:
                current = TRANSFORMATIONS[t_name](current)
                images.append(current)
            
            embeddings = encoder.encode_batch(images, batch_size=len(images), show_progress=False)
            
            z_0 = embeddings[0]
            z_end = embeddings[-1]
            
            # u = content vector (z_0)
            u = z_0
            
            # h = holonomy vector
            h = z_end - z_0
            
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
        """Ekstraktuje cechy."""
        stats = self.compute_features(encoder, image)
        return np.array([
            stats['mean_angle'],
            stats['std_angle'],
            stats['min_angle'],
            stats['max_angle']
        ], dtype=np.float32)


# ============================================================================
# COMBINED EXTRACTOR
# ============================================================================

class AllHypothesesExtractor:
    """Ekstrahuje cechy ze wszystkich 6 hipotez."""
    
    def __init__(self):
        self.h1 = H1_HolonomySpectrum()
        self.h2 = H2_AreaScaleLaw()
        self.h3 = H3_ConsistencyIndex()
        self.h4 = H4_PatchCoupled()
        self.h5 = H5_HolonomyTensor()
        self.h6 = H6_Orthogonality()
    
    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA for H1 and H5."""
        self.h1.fit_pca(embeddings)
        self.h5.fit_pca(embeddings)
    
    def extract_all(self, encoder, image: Image.Image) -> Dict[str, np.ndarray]:
        """Ekstraktuje wszystkie cechy."""
        return {
            'H1_spectrum': self.h1.extract_features(encoder, image),
            'H2_scale_law': self.h2.extract_features(encoder, image),
            'H3_consistency': self.h3.extract_features(encoder, image),
            'H4_patch_coupled': self.h4.extract_features(encoder, image),
            'H5_tensor': self.h5.extract_features(encoder, image),
            'H6_orthogonality': self.h6.extract_features(encoder, image),
        }
