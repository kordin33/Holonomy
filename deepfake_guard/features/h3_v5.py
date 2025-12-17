"""
h3_v5.py - H3 FRONTIER: Patch-Coherence + Roughness (HYPER-OPTIMIZED)

Optymalizacje:
1. Jeden mega-batch dla wszystkich 3 sekwencji + wszystkich źródeł
2. Precompute sekwencji przed encode
3. Wektoryzacja step vector i roughness

Cechy: 18D (3 sekwencje × 6 invariants)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple, Dict
import io


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).convert('RGB').copy()
    buffer.close()
    return img

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def downscale_upscale(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    return image.resize((int(w*scale), int(h*scale)), Image.LANCZOS).resize((w, h), Image.LANCZOS)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-10)
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)


# ============================================================================
# CONFIG
# ============================================================================

SEQ_PARAMS = {
    'J': [95, 90, 85, 80, 75],
    'B': [0.2, 0.35, 0.5, 0.65, 0.8],
    'S': [0.95, 0.9, 0.85, 0.8, 0.75],
}


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    return [image.crop((x, y, x+ps, y+ps)).resize((224, 224), Image.LANCZOS) for x, y in positions]


def prepare_all_sequences_for_source(image: Image.Image) -> Tuple[List[Image.Image], dict]:
    """
    Prepare all 3 sequences for one source.
    
    Returns:
        images: [J_seq(6), B_seq(6), S_seq(6)] = 18 images
        indices: {'J': (0, 6), 'B': (6, 12), 'S': (12, 18)}
    """
    imgs = []
    indices = {}
    
    for seq_name, params in SEQ_PARAMS.items():
        start = len(imgs)
        
        # Build sequence: [original, step0, step1, step2, step3, step4]
        seq_imgs = [image]
        curr = image
        for p in params:
            if seq_name == 'J':
                curr = jpeg_compression(curr, p)
            elif seq_name == 'B':
                curr = gaussian_blur(curr, p)
            elif seq_name == 'S':
                curr = downscale_upscale(curr, p)
            seq_imgs.append(curr)
        
        imgs.extend(seq_imgs)
        indices[seq_name] = (start, start + 6)
    
    return imgs, indices  # 18 images


def compute_step_vector_vectorized(embs: np.ndarray) -> np.ndarray:
    """Compute step distances from embeddings. embs: (6, D)"""
    dots = np.sum(embs[:-1] * embs[1:], axis=1)  # (5,)
    steps = np.sqrt(np.maximum(0, 2 - 2 * np.clip(dots, -1, 1)))
    return steps


def compute_roughness(v: np.ndarray) -> np.ndarray:
    if len(v) < 3:
        return np.zeros(3, dtype=np.float32)
    diffs = np.diff(v)
    allan = np.mean(diffs ** 2)
    S2_1 = np.mean((v[1:] - v[:-1]) ** 2)
    S2_2 = np.mean((v[2:] - v[:-2]) ** 2)
    rough_R = S2_2 / (S2_1 + 1e-10)
    jitter_std = np.std(diffs)
    return np.array([rough_R, allan, jitter_std], dtype=np.float32)


class H3_V5:
    """
    H3 Hyper-Optimized: 18D features.
    
    Per image: 6 sources × 18 images = 108 CLIP inferences.
    All in ONE encode_batch call.
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Build all sources: global + 5 patches
        patches = get_5_patches(image)
        sources = [image] + patches  # 6 sources
        
        # 2. Build ONE mega-batch for all sources, all sequences
        mega_batch = []
        source_indices = []
        
        for src in sources:
            offset = len(mega_batch)
            imgs, idx_map = prepare_all_sequences_for_source(src)
            mega_batch.extend(imgs)
            source_indices.append((offset, idx_map))
        
        # 3. ONE ENCODE CALL
        all_embs = encoder.encode_batch(mega_batch, batch_size=128, show_progress=False)
        all_embs = l2_normalize(np.asarray(all_embs, dtype=np.float32))
        
        # 4. Compute features for each sequence
        all_feats = []
        
        for seq_name in ['J', 'B', 'S']:
            step_vectors = []
            
            for offset, idx_map in source_indices:
                start, end = idx_map[seq_name]
                seq_embs = all_embs[offset + start : offset + end]
                steps = compute_step_vector_vectorized(seq_embs)
                step_vectors.append(steps)
            
            step_vectors = np.array(step_vectors)  # (6, 5)
            
            # Global roughness
            v_global = step_vectors[0]
            roughness = compute_roughness(v_global)  # 3D
            
            # Patch coherence
            patch_vectors = step_vectors[1:]  # (5, 5)
            corr = np.corrcoef(patch_vectors)
            triu = corr[np.triu_indices(5, k=1)]
            triu = triu[~np.isnan(triu)]
            sync_mean = np.mean(triu) if len(triu) > 0 else 0
            sync_min = np.min(triu) if len(triu) > 0 else 0
            
            # Worst patch energy
            deviations = np.linalg.norm(patch_vectors - v_global, axis=1)
            worst_patch = np.max(deviations)
            
            seq_feats = np.concatenate([roughness, [sync_mean, sync_min, worst_patch]])
            all_feats.append(seq_feats)
        
        return np.concatenate(all_feats).astype(np.float32)  # 18D
