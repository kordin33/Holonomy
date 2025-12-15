"""
quick_test.py - Szybki test czy wszystko działa

Uruchom to najpierw, żeby sprawdzić czy wszystkie moduły się importują
i czy modele tworzą się poprawnie.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        from deepfake_research.config import ExperimentConfig, EXPERIMENT_CONFIGS
        print("  ✓ config")
    except Exception as e:
        print(f"  ✗ config: {e}")
        return False
    
    try:
        from deepfake_research.models.backbones import BackboneFactory, get_backbone
        print("  ✓ backbones")
    except Exception as e:
        print(f"  ✗ backbones: {e}")
        return False
    
    try:
        from deepfake_research.models.frequency import FrequencyBranch, DCTBranch
        print("  ✓ frequency")
    except Exception as e:
        print(f"  ✗ frequency: {e}")
        return False
    
    try:
        from deepfake_research.models.attention import CBAM, ArtifactAttention
        print("  ✓ attention")
    except Exception as e:
        print(f"  ✗ attention: {e}")
        return False
    
    try:
        from deepfake_research.models.hybrid import HybridDeepfakeDetector, UltimateDeepfakeDetector
        print("  ✓ hybrid")
    except Exception as e:
        print(f"  ✗ hybrid: {e}")
        return False
    
    try:
        from deepfake_research.models.factory import create_model, list_models
        print("  ✓ factory")
    except Exception as e:
        print(f"  ✗ factory: {e}")
        return False
    
    try:
        from deepfake_research.data.augmentation import get_train_transforms, get_eval_transforms
        print("  ✓ augmentation")
    except Exception as e:
        print(f"  ✗ augmentation: {e}")
        return False
    
    try:
        from deepfake_research.data.sbi import SelfBlendedImageGenerator
        print("  ✓ sbi")
    except Exception as e:
        print(f"  ✗ sbi: {e}")
        return False
    
    try:
        from deepfake_research.training.trainer import Trainer
        print("  ✓ trainer")
    except Exception as e:
        print(f"  ✗ trainer: {e}")
        return False
    
    try:
        from deepfake_research.training.losses import DeepfakeLoss, FocalLoss
        print("  ✓ losses")
    except Exception as e:
        print(f"  ✗ losses: {e}")
        return False
    
    try:
        from deepfake_research.evaluation.metrics import compute_metrics
        print("  ✓ metrics")
    except Exception as e:
        print(f"  ✗ metrics: {e}")
        return False
    
    try:
        from deepfake_research.evaluation.benchmark import Benchmark
        print("  ✓ benchmark")
    except Exception as e:
        print(f"  ✗ benchmark: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_model_creation():
    """Test creating all models"""
    import torch
    from deepfake_research.models.factory import create_model, list_models
    
    print("\nTesting model creation...")
    print(f"Available models: {list(list_models().keys())}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)
    
    models_to_test = [
        "baseline_efficientnet",
        "baseline_vit",
        "freq_efficientnet",
        "attention_efficientnet",
        "hybrid",
        "ultimate",
    ]
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name).to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(x)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {model_name}: {num_params/1e6:.2f}M params, output shape {logits.shape}")
            
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")
    
    print("\nModel creation test completed!")


def test_frequency_modules():
    """Test frequency analysis modules"""
    import torch
    from deepfake_research.models.frequency import FrequencyBranch, DCTBranch, WaveletBranch
    
    print("\nTesting frequency modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 224, 224).to(device)
    
    # FFT Branch
    try:
        fft = FrequencyBranch(out_features=256).to(device)
        out = fft(x)
        assert out.shape == (2, 256)
        print(f"  ✓ FrequencyBranch: output {out.shape}")
    except Exception as e:
        print(f"  ✗ FrequencyBranch: {e}")
    
    # DCT Branch
    try:
        dct = DCTBranch(out_features=256).to(device)
        out = dct(x)
        assert out.shape == (2, 256)
        print(f"  ✓ DCTBranch: output {out.shape}")
    except Exception as e:
        print(f"  ✗ DCTBranch: {e}")
    
    # Wavelet Branch
    try:
        dwt = WaveletBranch(out_features=256).to(device)
        out = dwt(x)
        assert out.shape == (2, 256)
        print(f"  ✓ WaveletBranch: output {out.shape}")
    except Exception as e:
        print(f"  ✗ WaveletBranch: {e}")


def test_attention_modules():
    """Test attention modules"""
    import torch
    from deepfake_research.models.attention import (
        SpatialAttention, ChannelAttention, CBAM, ArtifactAttention
    )
    
    print("\nTesting attention modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 64, 56, 56).to(device)  # Feature map
    
    try:
        sa = SpatialAttention().to(device)
        out = sa(x)
        assert out.shape == x.shape
        print(f"  ✓ SpatialAttention: {out.shape}")
    except Exception as e:
        print(f"  ✗ SpatialAttention: {e}")
    
    try:
        ca = ChannelAttention(64).to(device)
        out = ca(x)
        assert out.shape == x.shape
        print(f"  ✓ ChannelAttention: {out.shape}")
    except Exception as e:
        print(f"  ✗ ChannelAttention: {e}")
    
    try:
        cbam = CBAM(64).to(device)
        out = cbam(x)
        assert out.shape == x.shape
        print(f"  ✓ CBAM: {out.shape}")
    except Exception as e:
        print(f"  ✗ CBAM: {e}")
    
    try:
        aa = ArtifactAttention(64).to(device)
        out = aa(x)
        assert out.shape == x.shape
        print(f"  ✓ ArtifactAttention: {out.shape}")
    except Exception as e:
        print(f"  ✗ ArtifactAttention: {e}")


def test_sbi():
    """Test Self-Blended Images generator"""
    from PIL import Image
    import numpy as np
    from deepfake_research.data.sbi import SelfBlendedImageGenerator
    
    print("\nTesting SBI generator...")
    
    try:
        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Generate SBI
        sbi_gen = SelfBlendedImageGenerator()
        sbi = sbi_gen.generate_sbi(img)
        
        assert sbi.size == img.size
        print(f"  ✓ SBI generator: input {img.size} → output {sbi.size}")
        
        # Test with mask
        sbi, mask = sbi_gen.generate_sbi_with_mask(img)
        assert mask.shape == (224, 224)
        print(f"  ✓ SBI with mask: mask shape {mask.shape}")
        
    except Exception as e:
        print(f"  ✗ SBI generator: {e}")


def main():
    print("="*60)
    print("DEEPFAKE DETECTION RESEARCH - QUICK TEST")
    print("="*60)
    
    # Run all tests
    if not test_imports():
        print("\nImport test failed. Fix errors before proceeding.")
        return
    
    test_model_creation()
    test_frequency_modules()
    test_attention_modules()
    test_sbi()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your data in ./data/A_standardized_224/ and ./data/B_standardized_224/")
    print("2. Run: python run_experiments.py --experiment all --epochs 20")
    print("3. Or for quick test: python run_experiments.py --experiment baseline --epochs 3")


if __name__ == "__main__":
    main()
