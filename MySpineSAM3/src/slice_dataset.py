"""
2D Slice Dataset for SAM Training
=================================
Extracts 2D slices from 3D volumes for SAM-based training.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List
import logging

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CTSpine1KSliceDataset(Dataset):
    """2D slice dataset from CTSpine1K."""
    
    def __init__(self, hf_dataset, transform=None, hu_min=-100, hu_max=1000, 
                 target_size=1024, output_size=256, binary_mask=True):
        self.hf = hf_dataset
        self.transform = transform
        self.hu_min, self.hu_max = hu_min, hu_max
        self.target_size, self.output_size = target_size, output_size
        self.binary_mask = binary_mask
    
    def __len__(self): return len(self.hf)
    
    def __getitem__(self, idx):
        s = self.hf[idx]
        img = np.clip(np.array(s["image"], dtype=np.float32), self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)
        lbl = (np.array(s["segmentation"]) > 0).astype(np.int64) if self.binary_mask else np.array(s["segmentation"])
        
        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl).unsqueeze(0)
        
        img = F.interpolate(img.unsqueeze(0), (self.target_size, self.target_size), mode='bilinear').squeeze(0)
        lbl = F.interpolate(lbl.float().unsqueeze(0), (self.output_size, self.output_size), mode='nearest').squeeze(0).long()
        
        return {"image": img, "label": lbl, "patient_id": s.get("patient_id", f"p{idx}")}


def get_slice_dataloaders(config, train_transform=None, val_transform=None):
    from src.ctspine1k_loader import load_ctspine1k_dataset
    
    data_cfg = config["data"]
    sam_cfg = config.get("model", {}).get("sam", {})
    
    hf = load_ctspine1k_dataset("2d", data_cfg.get("ctspine1k", {}).get("cache_dir"))
    
    kwargs = dict(hu_min=data_cfg.get("hu_min", -100), hu_max=data_cfg.get("hu_max", 1000),
                  target_size=sam_cfg.get("image_size", 1024), output_size=sam_cfg.get("output_size", 256),
                  binary_mask=data_cfg.get("ctspine1k", {}).get("binary_mask", True))
    
    batch = config["training"].get("batch_size", 16)
    train_ds = CTSpine1KSliceDataset(hf["train"], **kwargs)
    val_ds = CTSpine1KSliceDataset(hf["validation"], **kwargs)
    test_ds = CTSpine1KSliceDataset(hf["test"], **kwargs)
    
    return (DataLoader(train_ds, batch, shuffle=True, num_workers=4, drop_last=True),
            DataLoader(val_ds, batch, shuffle=False, num_workers=2),
            DataLoader(test_ds, batch, shuffle=False, num_workers=2))
