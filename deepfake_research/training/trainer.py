"""
trainer.py - Main Training Loop
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_auc: float = 0.0
    val_f1: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_auc": self.val_auc,
            "val_f1": self.val_f1,
            "learning_rate": self.learning_rate,
            "epoch_time": self.epoch_time,
        }


class Trainer:
    """
    Main Trainer class for deepfake detection models.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling
    - Checkpoint saving
    - W&B logging
    - Auxiliary loss support
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        
        # Training params
        epochs: int = 20,
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        
        # Early stopping
        early_stopping: bool = True,
        patience: int = 5,
        
        # Checkpointing
        save_dir: Path = Path("./outputs"),
        save_best: bool = True,
        save_every: int = 0,  # 0 = only best
        
        # Logging
        use_wandb: bool = False,
        experiment_name: str = "experiment",
        
        # Auxiliary losses
        use_aux_loss: bool = False,
        aux_loss_weight: float = 0.3,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        
        self.epochs = epochs
        self.use_amp = use_amp and device == "cuda"
        self.gradient_clip = gradient_clip
        
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every = save_every
        
        self.use_wandb = use_wandb and wandb is not None
        self.experiment_name = experiment_name
        
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # State
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0
        self.history: List[TrainingMetrics] = []
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dict with training results
        """
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 50)
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            val_metrics = self._validate_epoch(epoch)
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                train_acc=train_metrics['acc'],
                val_loss=val_metrics['loss'],
                val_acc=val_metrics['acc'],
                val_auc=val_metrics.get('auc', 0.0),
                val_f1=val_metrics.get('f1', 0.0),
                learning_rate=current_lr,
                epoch_time=epoch_time,
            )
            self.history.append(metrics)
            
            # Print progress
            self._print_epoch_summary(metrics)
            
            # Log to W&B
            if self.use_wandb:
                wandb.log(metrics.to_dict())
            
            # Save checkpoint
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['acc']
                self.best_val_auc = val_metrics.get('auc', 0.0)
                self.epochs_no_improve = 0
                
                if self.save_best:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1
            
            if self.save_every > 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.early_stopping and self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save final history
        self._save_history()
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'final_epoch': len(self.history),
            'history': [m.to_dict() for m in self.history],
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]",
            leave=False,
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                
                # Handle dict output (hybrid models)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    
                    # Main loss
                    loss = self.criterion(logits, labels)
                    
                    # Auxiliary losses
                    if self.use_aux_loss:
                        if 'aux_spatial' in outputs:
                            aux_loss = self.criterion(outputs['aux_spatial'], labels)
                            loss = loss + self.aux_loss_weight * aux_loss
                        if 'aux_freq' in outputs:
                            aux_loss = self.criterion(outputs['aux_freq'], labels)
                            loss = loss + self.aux_loss_weight * aux_loss
                else:
                    logits = outputs
                    loss = self.criterion(logits, labels)
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': correct / total,
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'acc': correct / total,
        }
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Val]",
            leave=False,
        )
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of "real" class
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        return {
            'loss': total_loss / len(self.val_loader),
            'acc': acc,
            'f1': f1,
            'auc': auc,
        }
    
    def _print_epoch_summary(self, metrics: TrainingMetrics) -> None:
        """Print epoch summary"""
        print(
            f"Epoch {metrics.epoch:3d} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Train Acc: {metrics.train_acc:.4f} | "
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"Val Acc: {metrics.val_acc:.4f} | "
            f"Val AUC: {metrics.val_auc:.4f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Time: {metrics.epoch_time:.1f}s"
        )
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.save_dir / f"{self.experiment_name}_best.pth"
        else:
            path = self.save_dir / f"{self.experiment_name}_epoch{epoch}.pth"
        
        torch.save(checkpoint, path)
        print(f"  → Saved: {path.name}")
    
    def _save_history(self) -> None:
        """Save training history to JSON"""
        history_path = self.save_dir / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump([m.to_dict() for m in self.history], f, indent=2)
    
    def load_checkpoint(self, path: str | Path) -> None:
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {path}")
