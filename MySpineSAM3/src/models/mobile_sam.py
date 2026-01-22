"""
Mobile SAM Wrapper for MySpineSAM3
==================================
Lightweight Mobile SAM for efficient medical image segmentation.
"""

from typing import Optional, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MobileSAM(nn.Module):
    """Mobile SAM wrapper."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, num_classes: int = 2,
                 image_size: int = 1024, output_size: int = 256, freeze_encoder: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.output_size = output_size
        
        self.sam = self._load(checkpoint_path)
        if freeze_encoder and self.sam: self._freeze()
        self.output_conv = nn.Conv2d(3, num_classes, 1)
    
    def _load(self, path):
        try:
            from mobile_sam import sam_model_registry
            return sam_model_registry["vit_t"](checkpoint=path)
        except ImportError:
            logger.warning("mobile_sam not installed")
            return None
    
    def _freeze(self):
        for p in self.sam.image_encoder.parameters(): p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters(): p.requires_grad = False
    
    def forward(self, x):
        if self.sam is None: return torch.zeros(x.shape[0], self.num_classes, self.output_size, self.output_size, device=x.device)
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        if x.shape[2] != self.image_size: x = F.interpolate(x, (self.image_size, self.image_size), mode='bilinear')
        
        img_emb = self.sam.image_encoder(x)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        masks, _ = self.sam.mask_decoder(image_embeddings=img_emb, image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                          sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
        return F.interpolate(self.output_conv(masks), (self.output_size, self.output_size), mode='bilinear')


class MobileSAMWithAdapter(nn.Module):
    """Mobile SAM with adapter layers."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, num_classes: int = 2,
                 image_size: int = 1024, output_size: int = 256, adapter_dim: int = 64):
        super().__init__()
        self.base = MobileSAM(checkpoint_path, num_classes, image_size, output_size, True)
        self.adapter = nn.Sequential(nn.Conv2d(num_classes, adapter_dim, 1), nn.GELU(), nn.Conv2d(adapter_dim, num_classes, 1))
        nn.init.zeros_(self.adapter[-1].weight)
    
    def forward(self, x):
        out = self.base(x)
        return out + self.adapter(out)


def create_mobile_sam(config: Dict[str, Any]) -> nn.Module:
    sam_cfg = config.get("sam", config.get("mobile_sam", {}))
    if sam_cfg.get("use_adapter", True):
        return MobileSAMWithAdapter(
            sam_cfg.get("checkpoint_path"), config.get("out_channels", 2),
            sam_cfg.get("image_size", 1024), sam_cfg.get("output_size", 256), sam_cfg.get("adapter_dim", 64))
    return MobileSAM(sam_cfg.get("checkpoint_path"), config.get("out_channels", 2),
                     sam_cfg.get("image_size", 1024), sam_cfg.get("output_size", 256), sam_cfg.get("freeze_encoder", True))
