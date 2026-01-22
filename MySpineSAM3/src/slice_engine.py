"""
2D Slice Training Engine
========================
Training loop for SAM-based 2D slice training with warmup + early stopping.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import monai

logger = logging.getLogger(__name__)


class SliceTrainer:
    """2D slice trainer for SAM models."""
    
    def __init__(self, model: nn.Module, config: Dict, train_loader: DataLoader, 
                 val_loader: DataLoader, device: Optional[torch.device] = None):
        self.model = model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = config
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_cfg = config["training"]
        self.num_epochs = train_cfg.get("num_epochs", 200)
        self.val_interval = train_cfg.get("val_interval", 2)
        
        sam_cfg = config.get("model", {}).get("sam", config.get("model", {}).get("segment_any_bone", {}))
        self.output_size = sam_cfg.get("output_size", 256)
        
        self.base_lr = sam_cfg.get("learning_rate", train_cfg.get("learning_rate", 5e-4))
        self.use_warmup = sam_cfg.get("warmup_enabled", True)
        self.warmup_period = sam_cfg.get("warmup_period", 200)
        
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.base_lr)
        
        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True, reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.best_dsc, self.last_update, self.patience = 0.0, 0, train_cfg.get("early_stopping", {}).get("patience", 20)
        self.checkpoint_dir = Path(config.get("checkpoints", {}).get("save_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _update_lr(self, iter_num, max_iters):
        if self.use_warmup and iter_num < self.warmup_period:
            lr = self.base_lr * ((iter_num + 1) / self.warmup_period)
        else:
            shift = iter_num - self.warmup_period if self.use_warmup else iter_num
            lr = self.base_lr * (1.0 - shift / max_iters) ** 0.9
        for g in self.optimizer.param_groups: g['lr'] = lr
        return lr
    
    def train_epoch(self, epoch, iter_num):
        self.model.train()
        total_loss = 0
        max_iters = self.num_epochs * len(self.train_loader)
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs = batch["image"].to(self.device)
            lbls = F.interpolate(batch["label"].float().to(self.device), (self.output_size, self.output_size), mode='nearest').long()
            
            self._update_lr(iter_num, max_iters)
            out = self.model(imgs)
            loss = self.dice_loss(out, lbls.float()) + self.ce_loss(out, lbls.squeeze(1))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            iter_num += 1
        
        return total_loss / len(self.train_loader), iter_num
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_dsc = 0, 0
        
        for batch in self.val_loader:
            imgs = batch["image"].to(self.device)
            lbls = F.interpolate(batch["label"].float().to(self.device), (self.output_size, self.output_size), mode='nearest').long()
            
            out = self.model(imgs)
            total_loss += (self.dice_loss(out, lbls.float()) + self.ce_loss(out, lbls.squeeze(1))).item()
            
            pred = out.argmax(1)
            for c in range(1, out.shape[1]):
                p, t = (pred == c).float(), (lbls.squeeze(1) == c).float()
                total_dsc += ((2 * (p * t).sum() + 1e-7) / (p.sum() + t.sum() + 1e-7)).item()
        
        return total_loss / len(self.val_loader), total_dsc / len(self.val_loader)
    
    def train(self):
        iter_num = 0
        for epoch in range(self.num_epochs):
            train_loss, iter_num = self.train_epoch(epoch, iter_num)
            
            if (epoch + 1) % self.val_interval == 0:
                val_loss, val_dsc = self.validate()
                logger.info(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, DSC: {val_dsc:.4f}")
                
                if val_dsc > self.best_dsc:
                    self.best_dsc, self.last_update = val_dsc, epoch
                    torch.save(self.model.state_dict(), self.checkpoint_dir / "checkpoint_best.pth")
                elif epoch - self.last_update > self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        torch.save(self.model.state_dict(), self.checkpoint_dir / "checkpoint_last.pth")
