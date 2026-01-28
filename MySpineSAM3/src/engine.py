"""
Training Engine for MySpineSAM3
===============================
3D volumetric training loop with DiceCELoss and validation.
"""

import os
import csv
from datetime import datetime
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
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss, HausdorffDTLoss

from src.metrics import MetricCalculator, InferenceTimer

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
        self.hd95_interval = train_cfg.get("hd95_interval", 5)
        
        # TensorBoard
        tb_dir = train_cfg.get("tensorboard_dir", "./logs/tensorboard")
        self.writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logging to: {tb_dir}")
        
        # CSV Logging
        self.log_dir = Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Unique timestamped filename: YYYYMMDD_HHMMSS_training_metrics.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{timestamp}_training_metrics.csv"
        
        # Initialize CSV with header (overwrites existing)
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_DSC", "Val_IoU", "HD95", "ASD", "InferenceTime_ms"])
        logger.info(f"CSV metrics logging to: {self.csv_path}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        
        # Loss Configuration
        loss_cfg = train_cfg.get("loss", {})
        
        # 1. DiceCELoss (Volume Overlap)
        # 1. Dice Loss (Volume Overlap)
        self.criterion_dice_only = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
        self.lambda_dice = loss_cfg.get("lambda_dice", 1.0)
        
        # 1.b Cross Entropy Loss (Classification)
        # Note: DiceCELoss uses standard CrossEntropyLoss internally for the CE part.
        # It expects logits (no softmax).
        self.criterion_ce = nn.CrossEntropyLoss()
        self.lambda_ce = loss_cfg.get("lambda_ce", 1.0)
        
        # 2. HausdorffDTLoss (Boundary Accuracy)
        self.lambda_hd = loss_cfg.get("lambda_hausdorff", 0.0)
        if self.lambda_hd > 0:
            logger.info(f"Using HausdorffDTLoss with lambda={self.lambda_hd}")
            self.criterion_hd = HausdorffDTLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True, # Use softmax for multi-class
            )
        else:
            self.criterion_hd = None
        
        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 15)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # Checkpoints
        self.checkpoint_dir = Path(config.get("checkpoints", {}).get("save_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_ce = 0.0
        epoch_hd = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", dynamic_ncols=True, mininterval=1.0)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss Components
            loss_dice = self.criterion_dice_only(outputs, labels)
            loss_ce = self.criterion_ce(outputs, labels.squeeze(1).long())
            
            loss = (self.lambda_dice * loss_dice) + (self.lambda_ce * loss_ce)
            
            # HD95 Loss
            loss_hd_val = 0.0
            if self.criterion_hd is not None:
                # Need softmax probabilities for HD loss
                if isinstance(outputs, list): # For deep supervision
                    out_for_hd = torch.softmax(outputs[0], dim=1)
                else:
                    out_for_hd = torch.softmax(outputs, dim=1)
                    
                loss_hd = self.criterion_hd(out_for_hd, labels)
                loss += (self.lambda_hd * loss_hd)
                loss_hd_val = loss_hd.item()
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dice += loss_dice.item()
            epoch_ce += loss_ce.item()
            epoch_hd += loss_hd_val
            
            pbar.set_postfix({
                "L": f"{loss.item():.3f}",
                "D": f"{loss_dice.item():.3f}",
                "C": f"{loss_ce.item():.3f}"
            })
        
        steps = len(self.train_loader)
        return {
            "total": epoch_loss / steps,
            "dice": epoch_dice / steps,
            "ce": epoch_ce / steps,
            "hd": epoch_hd / steps
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        self.model.eval()
        val_loss = 0.0
        
        # Initialize calculator with 3D metrics + InferenceTime
        # Initialize calculator with 3D metrics + InferenceTime
        metrics = ["DSC", "IOU", "InferenceTime"]
        
        # Check if we should compute expensive metrics this epoch
        should_compute_hd95 = (epoch + 1) % self.hd95_interval == 0
        if should_compute_hd95:
            metrics.extend(["HD95", "ASD"])
            
        num_classes = self.config["model"]["out_channels"]
        calculator = MetricCalculator(metrics=metrics, 
                                      include_background=False, 
                                      num_classes=num_classes)
        
        for batch_idx, batch in enumerate(self.val_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Time inference
            with InferenceTimer() as timer:
                outputs = self.model(images)
            
            # Loss Components
            loss_dice = self.criterion_dice_only(outputs, labels)
            loss_ce = self.criterion_ce(outputs, labels.squeeze(1).long())
            
            if self.criterion_hd is not None:
                # Need softmax probabilities for HD loss
                if isinstance(outputs, list):
                    out_for_hd = torch.softmax(outputs[0], dim=1)
                else:
                    out_for_hd = torch.softmax(outputs, dim=1)
                
                loss_hd = self.criterion_hd(out_for_hd, labels)
                loss = (self.lambda_dice * loss_dice) + (self.lambda_ce * loss_ce) + (self.lambda_hd * loss_hd)
            else:
                loss = (self.lambda_dice * loss_dice) + (self.lambda_ce * loss_ce)
                
            val_loss += loss.item()
            
            # Post-processing for metrics
            # argmax to get class indices: (B, C, H, W, D) -> (B, H, W, D)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy().squeeze(1) # Remove channel dim if present
            
            # Compute metrics for each item in batch
            for i in range(len(preds)):
                calculator.compute(
                    pred=preds[i], 
                    target=targets[i], 
                    inference_time_ms=timer.elapsed_ms
                )
            
            # Visualize middle slice of the first item in the first batch
            if self.writer and batch_idx == 0:
                # Select first item
                img = images[0, 0].cpu().numpy()     # (H, W, D)
                lbl = targets[0]                     # (H, W, D)
                prd = preds[0]                       # (H, W, D)
                
                # Middle axial slice
                mid_slice = img.shape[2] // 2
                
                # Prepare slices for grid: (1, H, W)
                img_slice = img[:, :, mid_slice]
                lbl_slice = lbl[:, :, mid_slice]
                prd_slice = prd[:, :, mid_slice]
                
                # Normalize image for vis (0-1)
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-7)
                
                # Normalize labels for vis (scale by max class to make 0-1)
                max_cls = self.config["model"]["out_channels"]
                lbl_slice = lbl_slice / max_cls
                prd_slice = prd_slice / max_cls
                
                # Stack: (3, 1, H, W)
                vis_stack = torch.stack([
                    torch.from_numpy(img_slice).unsqueeze(0),
                    torch.from_numpy(lbl_slice).unsqueeze(0),
                    torch.from_numpy(prd_slice).unsqueeze(0),
                ])
                
                grid = make_grid(vis_stack, nrow=3, normalize=True)
                self.writer.add_image("Val/Slice_Mid_Axial", grid, global_step=epoch)
        
        avg_metrics = calculator.get_average()
        return val_loss / len(self.val_loader), avg_metrics
    
    def train(self):
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(epoch)
            train_loss = train_metrics["total"]
            
            self.scheduler.step()
            
            # TensorBoard training loss
            self.writer.add_scalar("Loss/train_total", train_loss, epoch)
            self.writer.add_scalar("Loss/train_dice_loss", train_metrics["dice"], epoch)
            self.writer.add_scalar("Loss/train_ce_loss", train_metrics["ce"], epoch)
            if self.criterion_hd is not None:
                self.writer.add_scalar("Loss/train_hd_loss", train_metrics["hd"], epoch)
            
            if (epoch + 1) % self.val_interval == 0:
                val_loss, avg_metrics = self.validate(epoch)
                
                # Extract specific metrics
                dsc = avg_metrics.get("DSC", 0.0)
                iou = avg_metrics.get("IOU", 0.0)
                
                # HD95/ASD might be missing if skipped
                hd95 = avg_metrics.get("HD95", None)
                asd = avg_metrics.get("ASD", None)
                inf_time = avg_metrics.get("InferenceTime", 0.0)
                
                log_msg = f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, DSC: {dsc:.4f}"
                if hd95 is not None:
                     log_msg += f", HD95: {hd95:.4f}"
                logger.info(log_msg)
                
                # TensorBoard validation metrics
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metric/DSC", dsc, epoch)
                self.writer.add_scalar("Metric/IOU", iou, epoch)
                if hd95 is not None:
                    self.writer.add_scalar("Metric/HD95", hd95, epoch)
                
                # CSV Logging
                # Handle NaN/None values for HD95/ASD
                if hd95 is not None:
                    hd95_str = f"{hd95:.4f}" if not np.isnan(hd95) else "NaN"
                else:
                    hd95_str = "NaN"
                    
                if asd is not None:
                    asd_str = f"{asd:.4f}" if not np.isnan(asd) else "NaN"
                else:
                    asd_str = "NaN"
                
                with open(self.csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1, 
                        f"{train_loss:.4f}", 
                        f"{val_loss:.4f}", 
                        f"{dsc:.4f}", 
                        f"{iou:.4f}", 
                        hd95_str, 
                        asd_str, 
                        f"{inf_time:.2f}"
                    ])
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_dice": dsc,
                        "metrics": avg_metrics
                    }, self.checkpoint_dir / "best_model.pth")
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
        
        torch.save(self.model.state_dict(), self.checkpoint_dir / "last_checkpoint.pth")
        self.writer.close()
        logger.info("Training complete!")
