"""
Training Engine for MySpineSAM3
===============================
3D volumetric training loop with DiceCELoss and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from monai.losses import DiceCELoss

logger = logging.getLogger(__name__)


class Trainer:
    """3D volumetric training loop."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        
        # Training config
        train_cfg = config["training"]
        self.num_epochs = train_cfg.get("num_epochs", 100)
        self.val_interval = train_cfg.get("val_interval", 2)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        
        # Loss
        self.criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
        
        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 15)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # Checkpoints
        self.checkpoint_dir = Path(config.get("checkpoints", {}).get("save_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return epoch_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        val_loss = 0.0
        dice_scores = []
        
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
            
            # Dice score
            preds = outputs.argmax(dim=1)
            for c in range(1, outputs.shape[1]):
                pred_c = (preds == c).float()
                label_c = (labels.squeeze(1) == c).float()
                dice = (2 * (pred_c * label_c).sum() + 1e-7) / (pred_c.sum() + label_c.sum() + 1e-7)
                dice_scores.append(dice.item())
        
        return val_loss / len(self.val_loader), np.mean(dice_scores)
    
    def train(self):
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            
            if (epoch + 1) % self.val_interval == 0:
                val_loss, val_dice = self.validate()
                logger.info(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Dice: {val_dice:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                    }, self.checkpoint_dir / "best_model.pth")
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
        
        torch.save(self.model.state_dict(), self.checkpoint_dir / "last_checkpoint.pth")
        logger.info("Training complete!")
