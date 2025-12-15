"""
optimizers.py - Optimizer and Scheduler configurations
"""

from __future__ import annotations
from typing import Optional, Iterator
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> Optimizer:
    """
    Create optimizer.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adamw', 'adam', 'sgd', 'adafactor')
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    # Get parameters with different weight decay
    params = _get_parameter_groups(model, lr, weight_decay)
    
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
    
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
    
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True),
        )
    
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _get_parameter_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> list:
    """
    Create parameter groups with different settings.
    
    - No weight decay for biases and LayerNorm
    - Lower LR for pretrained backbone (optional)
    """
    # Parameters that should not have weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'bn']
    
    params_decay = []
    params_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd in name for nd in no_decay):
            params_no_decay.append(param)
        else:
            params_decay.append(param)
    
    return [
        {'params': params_decay, 'weight_decay': weight_decay},
        {'params': params_no_decay, 'weight_decay': 0.0},
    ]


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 20,
    warmup_epochs: int = 2,
    min_lr: float = 1e-6,
    **kwargs,
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Configured optimizer
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau', 'onecycle')
        epochs: Total training epochs
        warmup_epochs: Warmup epochs
        min_lr: Minimum learning rate
        
    Returns:
        Configured scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr,
        )
        
        if warmup_epochs > 0:
            return WarmupScheduler(
                optimizer,
                main_scheduler,
                warmup_epochs=warmup_epochs,
            )
        return main_scheduler
    
    elif scheduler_name == "step":
        step_size = kwargs.get('step_size', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    
    elif scheduler_name == "multistep":
        milestones = kwargs.get('milestones', [epochs // 2, 3 * epochs // 4])
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            min_lr=min_lr,
        )
    
    elif scheduler_name == "onecycle":
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / epochs if warmup_epochs > 0 else 0.3,
        )
    
    elif scheduler_name == "exponential":
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )
    
    elif scheduler_name == "none":
        return None
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class WarmupScheduler(_LRScheduler):
    """
    Warmup scheduler wrapper.
    
    Linear warmup followed by main scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        main_scheduler: _LRScheduler,
        warmup_epochs: int = 2,
        warmup_start_lr: float = 1e-7,
    ):
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.finished_warmup = False
        
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Use main scheduler
            return self.main_scheduler.get_last_lr()
    
    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epochs:
            self.main_scheduler.step()
        super().step(epoch)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts.
    
    LR cycles between max and min with increasing cycle lengths.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0
        
        super().__init__(optimizer)
    
    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.T_cur = epoch
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            self.cycle += 1
        
        super().step(epoch)
