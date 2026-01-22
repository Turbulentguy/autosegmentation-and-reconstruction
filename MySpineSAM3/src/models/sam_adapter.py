"""
SAM Adapter Module for MySpineSAM3
==================================
Implements adapter layers for fine-tuning SAM on medical imaging.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    """Bottleneck adapter layer."""
    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x): return x + self.up(self.act(self.down(x)))


class SAMAdapter(nn.Module):
    """SAM wrapper with adapter layers."""
    def __init__(self, sam_model: nn.Module, adapter_dim: int = 64, 
                 freeze_encoder: bool = True, freeze_prompt_encoder: bool = True):
        super().__init__()
        self.sam = sam_model
        
        if freeze_encoder:
            for p in self.sam.image_encoder.parameters(): p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.sam.prompt_encoder.parameters(): p.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SAMAdapter: {trainable:,} trainable params")
    
    def forward(self, images, points=None, boxes=None, masks=None, multimask_output=True):
        img_emb = self.sam.image_encoder(images)
        sparse, dense = self.sam.prompt_encoder(points=points, boxes=boxes, masks=masks)
        masks_pred, iou = self.sam.mask_decoder(
            image_embeddings=img_emb, image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=multimask_output)
        return masks_pred, iou


class SegmentAnyBoneModel(nn.Module):
    """SegmentAnyBone-style model for bone segmentation."""
    
    def __init__(self, num_classes: int = 2, image_size: int = 1024, output_size: int = 256,
                 checkpoint_path: Optional[str] = None, use_mobile_sam: bool = True, adapter_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.output_size = output_size
        
        self.sam = self._load_sam(checkpoint_path, use_mobile_sam)
        self.output_conv = nn.Conv2d(3, num_classes, 1)
        logger.info(f"SegmentAnyBoneModel: {num_classes} classes")
    
    def _load_sam(self, checkpoint_path, use_mobile_sam):
        try:
            if use_mobile_sam:
                from mobile_sam import sam_model_registry
                sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
            else:
                from segment_anything import sam_model_registry
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            
            for p in sam.image_encoder.parameters(): p.requires_grad = False
            for p in sam.prompt_encoder.parameters(): p.requires_grad = False
            return sam
        except ImportError as e:
            logger.warning(f"SAM not installed: {e}, using placeholder")
            return self._placeholder()
    
    def _placeholder(self):
        class P(nn.Module):
            def __init__(s):
                super().__init__()
                s.image_encoder = nn.Conv2d(3, 256, 16, 16)
                s.prompt_encoder = type('PE', (), {'__call__': lambda *a,**k: (torch.zeros(1,0,256), torch.zeros(1,256,64,64)), 'get_dense_pe': lambda: torch.zeros(1,256,64,64)})()
                s.mask_decoder = type('MD', (), {'__call__': lambda *a,**k: (torch.zeros(1,3,256,256), torch.ones(1,3))})()
        return P()
    
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        if x.shape[2] != self.image_size:
            x = F.interpolate(x, (self.image_size, self.image_size), mode='bilinear')
        
        img_emb = self.sam.image_encoder(x)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        masks, _ = self.sam.mask_decoder(image_embeddings=img_emb, image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                          sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
        masks = self.output_conv(masks)
        return F.interpolate(masks, (self.output_size, self.output_size), mode='bilinear')


def create_sam_model(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> nn.Module:
    sam_cfg = config.get("sam", {})
    return SegmentAnyBoneModel(
        num_classes=config.get("out_channels", 2),
        image_size=sam_cfg.get("image_size", 1024),
        output_size=sam_cfg.get("output_size", 256),
        checkpoint_path=checkpoint_path or sam_cfg.get("checkpoint_path"),
        use_mobile_sam=sam_cfg.get("use_mobile_sam", True),
        adapter_dim=sam_cfg.get("adapter_dim", 64))
