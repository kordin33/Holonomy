"""
cuda_utils.py - CUDA Optimizations for Maximum Performance

Optymalizacje:
- torch.compile() (PyTorch 2.0+) - 20-40% speedup
- cuDNN benchmark mode - 10-20% speedup
- Memory optimizations
- Flash Attention (jeśli dostępne)
"""

from __future__ import annotations
import os
import gc
from typing import Optional
import torch
import torch.nn as nn


def setup_cuda_optimizations(
    use_compile: bool = True,
    use_cudnn_benchmark: bool = True,
    use_tf32: bool = True,
    deterministic: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Konfiguruje optymalizacje CUDA dla maksymalnej wydajności.
    
    Args:
        use_compile: Użyj torch.compile() (PyTorch 2.0+)
        use_cudnn_benchmark: Włącz cuDNN benchmark mode
        use_tf32: Użyj TensorFloat-32 (Ampere+ GPUs)
        deterministic: Wymuś deterministyczne operacje (wolniejsze)
        verbose: Wyświetl informacje
        
    Returns:
        Dict z informacjami o konfiguracji
    """
    config = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cpu',
        'gpu_name': None,
        'gpu_memory': None,
        'optimizations': [],
    }
    
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️ CUDA niedostępna. Używam CPU.")
        return config
    
    # Device info
    config['device'] = 'cuda'
    config['gpu_name'] = torch.cuda.get_device_name(0)
    config['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if verbose:
        print(f"🎮 GPU: {config['gpu_name']}")
        print(f"💾 VRAM: {config['gpu_memory']:.1f} GB")
    
    # cuDNN Benchmark Mode
    if use_cudnn_benchmark and not deterministic:
        torch.backends.cudnn.benchmark = True
        config['optimizations'].append('cudnn_benchmark')
        if verbose:
            print("✅ cuDNN Benchmark Mode: ON (10-20% speedup)")
    
    # Deterministic mode (dla reprodukowalności, ale wolniejsze)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        config['optimizations'].append('deterministic')
        if verbose:
            print("⚠️ Deterministic Mode: ON (wolniejsze, ale reprodukowalne)")
    
    # TensorFloat-32 (Ampere GPUs - RTX 30xx, A100, etc.)
    if use_tf32:
        # TF32 daje ~3x speedup na matmul z minimalną stratą precyzji
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        config['optimizations'].append('tf32')
        if verbose:
            print("✅ TensorFloat-32: ON (Ampere+ GPUs, ~3x matmul speedup)")
    
    # Flash Attention (jeśli dostępne)
    try:
        # PyTorch 2.0+ ma wbudowane SDPA z Flash Attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            config['optimizations'].append('flash_attention_available')
            if verbose:
                print("✅ Flash Attention: Dostępne (automatyczne dla ViT)")
    except:
        pass
    
    # Memory optimizations
    torch.cuda.empty_cache()
    gc.collect()
    
    if verbose:
        print(f"🚀 Aktywne optymalizacje: {config['optimizations']}")
    
    return config


def compile_model(
    model: nn.Module,
    mode: str = "default",  # "default", "reduce-overhead", "max-autotune"
    verbose: bool = True,
) -> nn.Module:
    """
    Kompiluje model używając torch.compile() (PyTorch 2.0+).
    
    Daje 20-40% speedup bez zmian w kodzie!
    
    Args:
        model: Model do skompilowania
        mode: Tryb kompilacji
            - "default": Balans między czasem kompilacji a speedup
            - "reduce-overhead": Minimalizuje overhead, dobry dla małych batch
            - "max-autotune": Maksymalny speedup, długa kompilacja
        verbose: Wyświetl informacje
        
    Returns:
        Skompilowany model
    """
    # Sprawdź wersję PyTorch
    pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    
    if pytorch_version < (2, 0):
        if verbose:
            print(f"⚠️ torch.compile() wymaga PyTorch 2.0+. Masz: {torch.__version__}")
        return model
    
    try:
        if verbose:
            print(f"🔧 Kompilacja modelu (mode={mode})... ", end="")
        
        compiled_model = torch.compile(model, mode=mode)
        
        if verbose:
            print("✅ Sukces!")
            print("   ℹ️ Pierwsza iteracja będzie wolniejsza (kompilacja JIT)")
        
        return compiled_model
        
    except Exception as e:
        if verbose:
            print(f"⚠️ Kompilacja nieudana: {e}")
            print("   ℹ️ Używam nieskompilowanego modelu")
        return model


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple = (3, 224, 224),
    max_batch_size: int = 128,
    target_memory_usage: float = 0.8,
    device: str = "cuda",
) -> int:
    """
    Automatycznie znajduje optymalny batch size dla GPU.
    
    Args:
        model: Model do testowania
        input_shape: Shape pojedynczego inputu (C, H, W)
        max_batch_size: Maksymalny batch size do sprawdzenia
        target_memory_usage: Docelowe użycie pamięci (0-1)
        device: Device
        
    Returns:
        Optymalny batch size
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 16  # Domyślny dla CPU
    
    model = model.to(device)
    model.eval()
    
    # Wyczyść pamięć
    torch.cuda.empty_cache()
    gc.collect()
    
    optimal_batch = 1
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        if batch_size > max_batch_size:
            break
            
        try:
            # Test forward pass
            x = torch.randn(batch_size, *input_shape, device=device)
            
            with torch.no_grad():
                _ = model(x)
            
            # Sprawdź użycie pamięci
            memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            
            if memory_used < target_memory_usage:
                optimal_batch = batch_size
            else:
                break
                
            # Wyczyść
            del x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            break
    
    return optimal_batch


def print_cuda_memory_stats():
    """Wyświetla statystyki pamięci CUDA"""
    if not torch.cuda.is_available():
        print("CUDA niedostępna")
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"📊 GPU Memory:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved:  {reserved:.2f} GB")
    print(f"   Total:     {total:.2f} GB")
    print(f"   Free:      {total - reserved:.2f} GB")


def clear_cuda_cache():
    """Czyści cache CUDA"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 CUDA cache wyczyszczony")


class CUDAMemoryMonitor:
    """
    Context manager do monitorowania zużycia pamięci CUDA.
    
    Użycie:
        with CUDAMemoryMonitor("Training step"):
            model(x)
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_memory = 0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            diff = (end_memory - self.start_memory) / 1e6
            print(f"📊 {self.name}: {diff:+.1f} MB")


def optimize_dataloader_for_gpu(
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> dict:
    """
    Zwraca optymalne parametry dla DataLoader na GPU.
    
    Returns:
        Dict z parametrami dla DataLoader
    """
    if torch.cuda.is_available():
        return {
            'num_workers': num_workers,
            'pin_memory': True,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': True if num_workers > 0 else False,
        }
    else:
        return {
            'num_workers': 0,
            'pin_memory': False,
        }


# Gradient Checkpointing dla dużych modeli
def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Włącza gradient checkpointing dla modelu.
    
    Zmniejsza zużycie VRAM kosztem wolniejszego treningu (~20% wolniej).
    Przydatne gdy model nie mieści się w pamięci.
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        # Dla modeli HuggingFace
        model.gradient_checkpointing_enable()
        print("✅ Gradient Checkpointing: ON (mniej VRAM, wolniejszy trening)")
    elif hasattr(model, 'set_grad_checkpointing'):
        # Dla niektórych modeli timm
        model.set_grad_checkpointing(True)
        print("✅ Gradient Checkpointing: ON")
    else:
        print("⚠️ Model nie wspiera gradient checkpointing")
    
    return model
