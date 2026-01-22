#!/usr/bin/env python
"""
MySpineSAM3 Training Script
===========================
Main entry point. Supports both 3D volumetric and 2D slice-based training.

Usage:
    python train.py --config configs/spine_ct_config.yaml
    python train.py --config configs/spine_ct_config.yaml --model MobileSAM
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.model_loader import get_model
from src.engine import Trainer
from src.utils.data_augmentation import get_train_transforms, get_val_transforms


def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MySpineSAM3 Training")
    parser.add_argument("--config", default="configs/spine_ct_config.yaml")
    parser.add_argument("--model", choices=["UNET", "UNETR", "SwinUNETR", "SegmentAnyBone", "MobileSAM", "SAM2D", "SAM3"])
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = load_config(args.config)
    
    # Overrides
    if args.model: config["model"]["architecture"] = args.model
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    if args.epochs: config["training"]["num_epochs"] = args.epochs
    if args.lr: config["training"]["learning_rate"] = args.lr
    
    set_seed(args.seed or config["project"]["seed"])
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Device: {device}")
    
    logger.info(f"Model: {config['model']['architecture']}")
    model = get_model(config["model"])
    
    # Training mode
    training_mode = config.get("model", {}).get("training_mode", "3d")
    arch = config["model"].get("architecture", "").upper()
    
    if arch in ["MOBILESAM", "SAM2D", "SEGMENTANYBONE"] and training_mode == "3d":
        logger.info("SAM model detected, consider training_mode='2d'")
    
    if training_mode == "2d":
        # 2D slice training
        from src.slice_dataset import get_slice_dataloaders
        from src.slice_engine import SliceTrainer
        
        train_loader, val_loader, _ = get_slice_dataloaders(config)
        trainer = SliceTrainer(model, config, train_loader, val_loader, device)
    else:
        # 3D volumetric training
        train_transforms = get_train_transforms(config)
        val_transforms = get_val_transforms(config)
        
        data_source = config["data"].get("source", "local")
        if data_source == "ctspine1k":
            from src.ctspine1k_loader import get_ctspine1k_dataloaders
            train_loader, val_loader, _ = get_ctspine1k_dataloaders(
                config, train_transforms, val_transforms, config["data"].get("ctspine1k", {}).get("mode", "3d"))
        else:
            from src.dataset import get_dataloaders
            train_loader, val_loader, _ = get_dataloaders(config, train_transforms, val_transforms)
        
        trainer = Trainer(model, config, train_loader, val_loader, device)
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        trainer.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Resumed from {args.resume}")
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
