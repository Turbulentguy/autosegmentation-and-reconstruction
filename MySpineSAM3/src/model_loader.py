"""
Model Factory for MySpineSAM3
=============================
Implements a factory pattern to load different segmentation architectures
based on configuration. Supports UNET, UNETR, SwinUNETR, and SAM variants.

Usage:
    from src.model_loader import get_model
    model = get_model(config['model'])
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from monai.networks.nets import UNet, UNETR, SwinUNETR

logger = logging.getLogger(__name__)


class SegmentAnyBone(nn.Module):
    """Placeholder for legacy SegmentAnyBone (use SegmentAnyBoneModel instead)."""
    def __init__(self, in_channels=1, out_channels=2, img_size=(96,96,96), **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv3d(64, out_channels, 1),
        )
    def forward(self, x): return self.decoder(self.encoder(x))


class SegmentAnyModel3(nn.Module):
    """SAM3 placeholder using SwinUNETR backbone."""
    def __init__(self, in_channels=1, out_channels=2, img_size=(96,96,96), **kwargs):
        super().__init__()
        self.backbone = SwinUNETR(img_size=img_size, in_channels=in_channels, 
                                   out_channels=out_channels, feature_size=48)
    def forward(self, x): return self.backbone(x)


def get_model(model_config: Dict[str, Any]) -> nn.Module:
    """Factory function to create model based on configuration."""
    architecture = model_config.get("architecture", "UNET").upper()
    in_channels = model_config.get("in_channels", 1)
    out_channels = model_config.get("out_channels", 2)
    
    logger.info(f"Loading model architecture: {architecture}")
    
    if architecture == "UNET":
        unet_cfg = model_config.get("unet", {})
        model = UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=unet_cfg.get("features", [32,64,128,256,512]),
            strides=[2,2,2,2], dropout=unet_cfg.get("dropout", 0.1), num_res_units=2,
        )
    elif architecture == "UNETR":
        cfg = model_config.get("unetr", {})
        model = UNETR(
            in_channels=in_channels, out_channels=out_channels,
            img_size=tuple(cfg.get("img_size", [96,96,96])),
            feature_size=cfg.get("feature_size", 16),
            hidden_size=cfg.get("hidden_size", 768),
            mlp_dim=cfg.get("mlp_dim", 3072),
            num_heads=cfg.get("num_heads", 12), spatial_dims=3,
        )
    elif architecture == "SWINUNETR":
        cfg = model_config.get("swin_unetr", {})
        model = SwinUNETR(
            img_size=tuple(cfg.get("img_size", [96,96,96])),
            in_channels=in_channels, out_channels=out_channels,
            feature_size=cfg.get("feature_size", 48),
            depths=cfg.get("depths", [2,2,2,2]),
            num_heads=cfg.get("num_heads", [3,6,12,24]), use_checkpoint=True,
        )
    elif architecture == "SEGMENTANYBONE":
        from src.models.sam_adapter import SegmentAnyBoneModel
        sam_cfg = model_config.get("sam", model_config.get("segment_any_bone", {}))
        model = SegmentAnyBoneModel(
            num_classes=out_channels,
            image_size=sam_cfg.get("image_size", 1024),
            output_size=sam_cfg.get("output_size", 256),
            checkpoint_path=sam_cfg.get("checkpoint_path"),
            use_mobile_sam=sam_cfg.get("use_mobile_sam", True),
            adapter_dim=sam_cfg.get("adapter_dim", 64),
        )
    elif architecture in ["MOBILESAM", "SAM2D"]:
        from src.models.mobile_sam import create_mobile_sam
        model = create_mobile_sam(model_config)
    elif architecture in ["SEGMENTANYMODEL3", "SAM3"]:
        cfg = model_config.get("segment_any_model3", {})
        model = SegmentAnyModel3(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} total, {trainable:,} trainable")
    return model
