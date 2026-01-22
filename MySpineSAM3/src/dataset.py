"""
Dataset Module for MySpineSAM3
==============================
3D NIfTI dataset with nibabel I/O and MONAI transforms.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List
import logging

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SpineDataset(Dataset):
    """3D Spine CT dataset from NIfTI files."""
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        hu_min: float = -100,
        hu_max: float = 1000,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.hu_min = hu_min
        self.hu_max = hu_max
        
        self.image_files = sorted(self.image_dir.glob("*.nii*"))
        logger.info(f"SpineDataset: {len(self.image_files)} volumes from {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        lbl_path = self.label_dir / img_path.name
        
        image = nib.load(str(img_path)).get_fdata().astype(np.float32)
        label = nib.load(str(lbl_path)).get_fdata().astype(np.int64) if lbl_path.exists() else np.zeros_like(image)
        
        # HU windowing
        image = np.clip(image, self.hu_min, self.hu_max)
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)
        
        # Add channel dim
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        
        data = {"image": image, "label": label}
        if self.transform:
            data = self.transform(data)
        
        if isinstance(data["image"], np.ndarray):
            data["image"] = torch.from_numpy(data["image"].copy())
            data["label"] = torch.from_numpy(data["label"].copy())
        
        return data


def get_dataloaders(
    config: Dict[str, Any],
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    root = Path(config["data"]["root_dir"])
    batch_size = config["training"]["batch_size"]
    
    train_dataset = SpineDataset(root/"train"/"images", root/"train"/"labels", train_transform)
    val_dataset = SpineDataset(root/"val"/"images", root/"val"/"labels", val_transform)
    test_dataset = SpineDataset(root/"test"/"images", root/"test"/"labels", val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
