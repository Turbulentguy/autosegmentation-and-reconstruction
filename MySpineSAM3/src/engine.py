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
        self.hd95_interval = train_cfg.get("hd95_interval", 10)  # Default 10 instead of 5
        
        # NEW: Mixed Precision Training
        self.use_amp = train_cfg.get("use_amp", False)
        if self.use_amp:
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
            logger.info("Mixed Precision Training (AMP) enabled - expect 2-3x speedup!")
        else:
            self.scaler = None
        
        # NEW: Gradient Accumulation
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
        if self.grad_accum_steps > 1:
            logger.info(f"Gradient Accumulation: {self.grad_accum_steps} steps (effective batch = {train_cfg.get('batch_size', 2) * self.grad_accum_steps})")
        
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
            # IMPROVED: Add more informative columns
            writer.writerow(["Epoch", "Train_Loss", "Train_Dice", "Train_CE", "Val_Loss", "Val_DSC", "Val_IoU", "HD95", "ASD", "InferenceTime_ms", "LR"])
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
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # MIXED PRECISION: Forward pass with autocast
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(images)
                    
                    # Loss Components
                    loss_dice = self.criterion_dice_only(outputs, labels)
                    loss_ce = self.criterion_ce(outputs, labels.squeeze(1).long())
                    loss = (self.lambda_dice * loss_dice) + (self.lambda_ce * loss_ce)
                    
                    # HD95 Loss (if enabled)
                    loss_hd_val = 0.0
                    if self.criterion_hd is not None:
                        if isinstance(outputs, list):
                            out_for_hd = torch.softmax(outputs[0], dim=1)
                        else:
                            out_for_hd = torch.softmax(outputs, dim=1)
                        loss_hd = self.criterion_hd(out_for_hd, labels)
                        loss += (self.lambda_hd * loss_hd)
                        loss_hd_val = loss_hd.item()
                    
                    # Gradient accumulation: Scale loss
                    loss = loss / self.grad_accum_steps
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
                
                # Optimizer step only every grad_accum_steps
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training (no AMP)
                outputs = self.model(images)
                
                loss_dice = self.criterion_dice_only(outputs, labels)
                loss_ce = self.criterion_ce(outputs, labels.squeeze(1).long())
                loss = (self.lambda_dice * loss_dice) + (self.lambda_ce * loss_ce)
                
                loss_hd_val = 0.0
                if self.criterion_hd is not None:
                    if isinstance(outputs, list):
                        out_for_hd = torch.softmax(outputs[0], dim=1)
                    else:
                        out_for_hd = torch.softmax(outputs, dim=1)
                    loss_hd = self.criterion_hd(out_for_hd, labels)
                    loss += (self.lambda_hd * loss_hd)
                    loss_hd_val = loss_hd.item()
                
                loss = loss / self.grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track metrics (multiply back by grad_accum_steps for true loss)
            epoch_loss += loss.item() * self.grad_accum_steps
            epoch_dice += loss_dice.item()
            epoch_ce += loss_ce.item()
            epoch_hd += loss_hd_val
            
            # IMPROVED: Clearer progress bar with explanation
            # L = Total Loss, D = Dice (0=perfect, 1=worst), CE = CrossEntropy (0=perfect)
            pbar.set_postfix({
                "Loss": f"{loss.item() * self.grad_accum_steps:.3f}",
                "Dice": f"{loss_dice.item():.3f}",
                "CE": f"{loss_ce.item():.2f}"
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
        logger.info("=" * 80)
        logger.info("LOSS INTERPRETATION GUIDE:")
        logger.info("  - Total Loss = Dice Loss + CrossEntropy Loss")
        logger.info("  - Dice Loss: 0.0 = perfect overlap, 1.0 = no overlap")
        logger.info("  - CrossEntropy: 0.0 = perfect classification, ~2.5 = random guessing")
        logger.info("  - Initial Total Loss: 2.5-4.0 is NORMAL!")
        logger.info("  - Target Total Loss: < 1.0 for good performance")
        logger.info("=" * 80)
        
        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(epoch)
            train_loss = train_metrics["total"]
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # TensorBoard training loss
            self.writer.add_scalar("Loss/train_total", train_loss, epoch)
            self.writer.add_scalar("Loss/train_dice_loss", train_metrics["dice"], epoch)
            self.writer.add_scalar("Loss/train_ce_loss", train_metrics["ce"], epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)
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
                
                # IMPROVED: More informative logging
                log_msg = f"Epoch {epoch+1}/{self.num_epochs} | "
                log_msg += f"Train: {train_loss:.4f} (Dice={train_metrics['dice']:.3f}, CE={train_metrics['ce']:.2f}) | "
                log_msg += f"Val: {val_loss:.4f} | DSC: {dsc*100:.1f}% | IoU: {iou*100:.1f}%"
                if hd95 is not None and hd95 < 999:  # Filter out invalid HD95
                    log_msg += f" | HD95: {hd95:.2f}mm"
                log_msg += f" | LR: {current_lr:.2e}"
                logger.info(log_msg)
                
                # TensorBoard validation metrics
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metric/DSC", dsc, epoch)
                self.writer.add_scalar("Metric/IOU", iou, epoch)
                if hd95 is not None:
                    self.writer.add_scalar("Metric/HD95", hd95, epoch)
                
                # CSV Logging with comprehensive metrics
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
                    # Updated format: Epoch, Train_Loss, Train_Dice, Train_CE, Val_Loss, Val_DSC, Val_IoU, HD95, ASD, InferenceTime_ms, LR
                    writer.writerow([
                        epoch + 1, 
                        f"{train_loss:.4f}", 
                        f"{train_metrics['dice']:.4f}",  # NEW: Train Dice
                        f"{train_metrics['ce']:.4f}",    # NEW: Train CE
                        f"{val_loss:.4f}", 
                        f"{dsc:.4f}", 
                        f"{iou:.4f}", 
                        hd95_str, 
                        asd_str, 
                        f"{inf_time:.2f}",
                        f"{current_lr:.2e}"  # NEW: Learning Rate
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
