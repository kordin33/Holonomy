"""
h2_v5.py - H2 FRONTIER: Curvature-Density (HYPER-OPTIMIZED)

Optymalizacje:
1. 101 → 61 obrazów per source (A_i i B_j nie duplikowane)
2. Precompute A_i, B_j, S_k - AB/BA generowane z cache
3. Wektoryzacja kappa (np.dot zamiast pętli)
4. Jeden mega-batch dla obu par + wszystkich źródeł
5. Współdzielenie z0 i J_i między parami

Cechy: 16D (2 pary × 8 invariants)
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple
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

JPEG_Q = [95, 90, 85, 80, 75]
BLUR_S = [0.2, 0.4, 0.6, 0.8, 1.0]
SCALE_F = [0.95, 0.9, 0.85, 0.8, 0.75]


def get_5_patches(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    ps = min(w, h) // 2
    positions = [
        (0, 0), (w - ps, 0),
        (0, h - ps), (w - ps, h - ps),
        ((w - ps) // 2, (h - ps) // 2),
    ]
    return [image.crop((x, y, x+ps, y+ps)).resize((224, 224), Image.LANCZOS) for x, y in positions]


def prepare_combo_batch_for_source(image: Image.Image) -> Tuple[List[Image.Image], dict]:
    """
    Prepare all images for BOTH pairs (J,B) and (J,S) from single source.
    
    Returns:
        images: [z0, J_0..J_4, B_0..B_4, S_0..S_4, JB_00..JB_44, BJ_00..BJ_44, JS_00..JS_44, SJ_00..SJ_44]
        indices: dict mapping to positions
    
    Total: 1 + 5 + 5 + 5 + 25 + 25 + 25 + 25 = 116 images (shared J_i for both pairs)
    """
    imgs = [image]  # z0
    
    # Precompute single degradations (cached for AB/BA generation)
    J_imgs = [jpeg_compression(image, q) for q in JPEG_Q]  # 5
    B_imgs = [gaussian_blur(image, s) for s in BLUR_S]     # 5
    S_imgs = [downscale_upscale(image, f) for f in SCALE_F]  # 5
    
    imgs.extend(J_imgs)  # indices 1-5
    imgs.extend(B_imgs)  # indices 6-10
    imgs.extend(S_imgs)  # indices 11-15
    
    # JB: AB = blur(J_i), BA = jpeg(B_j)
    JB_AB = [gaussian_blur(J_imgs[i], BLUR_S[j]) for i in range(5) for j in range(5)]  # 25
    JB_BA = [jpeg_compression(B_imgs[j], JPEG_Q[i]) for i in range(5) for j in range(5)]  # 25
    
    # JS: AS = scale(J_i), SA = jpeg(S_k)
    JS_AS = [downscale_upscale(J_imgs[i], SCALE_F[k]) for i in range(5) for k in range(5)]  # 25
    JS_SA = [jpeg_compression(S_imgs[k], JPEG_Q[i]) for i in range(5) for k in range(5)]  # 25
    
    imgs.extend(JB_AB)   # 16-40
    imgs.extend(JB_BA)   # 41-65
    imgs.extend(JS_AS)   # 66-90
    imgs.extend(JS_SA)   # 91-115
    
    indices = {
        'z0': 0,
        'J': (1, 6),      # J[0..4]
        'B': (6, 11),     # B[0..4]
        'S': (11, 16),    # S[0..4]
        'JB_AB': (16, 41),
        'JB_BA': (41, 66),
        'JS_AS': (66, 91),
        'JS_SA': (91, 116),
    }
    
    return imgs, indices


def compute_kappa_vectorized(z0: np.ndarray, zA: np.ndarray, zB: np.ndarray, 
                              zAB: np.ndarray, zBA: np.ndarray) -> np.ndarray:
    """
    Vectorized kappa computation from embeddings.
    
    Args:
        z0: (D,) base embedding
        zA: (5, D) first axis embeddings
        zB: (5, D) second axis embeddings  
        zAB: (25, D) composed A then B
        zBA: (25, D) composed B then A
    
    Returns:
        kappa: (5, 5) curvature matrix
    """
    # Chordal distance: d(a,b) = sqrt(2 - 2*dot(a,b))
    # sA[i] = d(z0, zA[i])
    dots_A = np.dot(zA, z0)  # (5,)
    sA = np.sqrt(np.maximum(0, 2 - 2 * np.clip(dots_A, -1, 1)))
    
    dots_B = np.dot(zB, z0)  # (5,)
    sB = np.sqrt(np.maximum(0, 2 - 2 * np.clip(dots_B, -1, 1)))
    
    # C[i,j] = d(zAB[i*5+j], zBA[i*5+j])
    dots_C = np.sum(zAB * zBA, axis=1)  # (25,)
    C_flat = np.sqrt(np.maximum(0, 2 - 2 * np.clip(dots_C, -1, 1)))
    C = C_flat.reshape(5, 5)
    
    # Area: A[i,j] = sA[i] * sB[j]
    A = np.outer(sA, sB)
    
    # kappa = C / (A + eps)
    kappa = C / (A + 1e-8)
    
    return kappa


def curvature_invariants(kappa: np.ndarray) -> np.ndarray:
    flat = kappa.flatten()
    k_mean = np.mean(flat)
    k_std = np.std(flat)
    k_p90 = np.percentile(flat, 90)
    k_max = np.max(flat)
    k_aniso = np.mean(kappa, axis=1).mean() - np.mean(kappa, axis=0).mean()
    try:
        s = np.linalg.svd(kappa, compute_uv=False)
        s = s[s > 1e-10]
        p = s / np.sum(s)
        k_rank_eff = np.exp(-np.sum(p * np.log(p + 1e-10)))
    except:
        k_rank_eff = 1.0
    return np.array([k_mean, k_std, k_p90, k_max, k_aniso, k_rank_eff], dtype=np.float32)


class H2_V5:
    """
    H2 Hyper-Optimized: 16D features.
    
    Per image: 6 sources × 116 images = 696 CLIP inferences (was ~2200).
    All in ONE encode_batch call.
    """
    
    def extract_features(self, encoder, image: Image.Image) -> np.ndarray:
        # 1. Build all sources: global + 5 patches
        patches = get_5_patches(image)
        sources = [image] + patches  # 6 sources
        
        # 2. Build ONE mega-batch for all sources
        mega_batch = []
        source_indices = []
        
        for src in sources:
            offset = len(mega_batch)
            imgs, idx_map = prepare_combo_batch_for_source(src)
            mega_batch.extend(imgs)
            source_indices.append((offset, idx_map))
        
        # 3. ONE ENCODE CALL with high batch_size
        all_embs = encoder.encode_batch(mega_batch, batch_size=256, show_progress=False)
        all_embs = l2_normalize(np.asarray(all_embs, dtype=np.float32))
        
        # 4. Compute features for each pair
        all_feats = []
        
        for pair_name, (a_key, b_key, ab_key, ba_key) in [
            ('JB', ('J', 'B', 'JB_AB', 'JB_BA')),
            ('JS', ('J', 'S', 'JS_AS', 'JS_SA')),
        ]:
            kappas = []
            
            for offset, idx_map in source_indices:
                z0 = all_embs[offset + idx_map['z0']]
                zA = all_embs[offset + idx_map[a_key][0] : offset + idx_map[a_key][1]]
                zB = all_embs[offset + idx_map[b_key][0] : offset + idx_map[b_key][1]]
                zAB = all_embs[offset + idx_map[ab_key][0] : offset + idx_map[ab_key][1]]
                zBA = all_embs[offset + idx_map[ba_key][0] : offset + idx_map[ba_key][1]]
                
                kappa = compute_kappa_vectorized(z0, zA, zB, zAB, zBA)
                kappas.append(kappa)
            
            # Global + patches stats
            global_inv = curvature_invariants(kappas[0])  # 6D
            patch_means = np.array([np.mean(k) for k in kappas[1:]])
            k_patch_range = np.max(patch_means) - np.min(patch_means)
            
            patch_kappas = np.array([k.flatten() for k in kappas[1:]])
            corr = np.corrcoef(patch_kappas)
            triu = corr[np.triu_indices(5, k=1)]
            triu = triu[~np.isnan(triu)]
            k_patch_sync = np.mean(triu) if len(triu) > 0 else 0.0
            
            pair_feats = np.concatenate([global_inv, [k_patch_range, k_patch_sync]])
            all_feats.append(pair_feats)
        
        return np.concatenate(all_feats).astype(np.float32)  # 16D
