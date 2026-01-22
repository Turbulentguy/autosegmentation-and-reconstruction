"""
CTSpine1K Dataset Loader
========================
HuggingFace CTSpine1K integration.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CTSpine1KDataset(Dataset):
    """PyTorch wrapper for CTSpine1K HuggingFace dataset."""
    
    def __init__(self, hf_dataset, transform=None, hu_min=-100, hu_max=1000, binary_mask=True):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.hu_min, self.hu_max = hu_min, hu_max
        self.binary_mask = binary_mask
    
    def __len__(self): return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = np.clip(np.array(sample["image"], dtype=np.float32), self.hu_min, self.hu_max)
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)
        label = (np.array(sample["segmentation"]) > 0).astype(np.int64) if self.binary_mask else np.array(sample["segmentation"])
        
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        data = {"image": image, "label": label, "patient_id": sample.get("patient_id", f"p{idx}")}
        
        if self.transform: data = self.transform(data)
        if isinstance(data["image"], np.ndarray):
            data["image"] = torch.from_numpy(data["image"].copy())
            data["label"] = torch.from_numpy(data["label"].copy())
        return data


def load_ctspine1k_dataset(mode="3d", cache_dir=None, writer_batch_size=5):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    logger.info(f"Loading CTSpine1K (mode={mode})...")
    return load_dataset("alexanderdann/CTSpine1K", name=mode, trust_remote_code=True, 
                        writer_batch_size=writer_batch_size, cache_dir=cache_dir)


def get_ctspine1k_dataloaders(config, train_transform=None, val_transform=None, mode="3d"):
    data_cfg = config["data"]
    hf = load_ctspine1k_dataset(mode, data_cfg.get("ctspine1k", {}).get("cache_dir"))
    
    kwargs = dict(hu_min=data_cfg.get("hu_min", -100), hu_max=data_cfg.get("hu_max", 1000),
                  binary_mask=data_cfg.get("ctspine1k", {}).get("binary_mask", True))
    
    train_ds = CTSpine1KDataset(hf["train"], train_transform, **kwargs)
    val_ds = CTSpine1KDataset(hf["validation"], val_transform, **kwargs)
    test_ds = CTSpine1KDataset(hf["test"], val_transform, **kwargs)
    
    batch = config["training"]["batch_size"]
    return (DataLoader(train_ds, batch, shuffle=True, num_workers=4, drop_last=True),
            DataLoader(val_ds, 1, shuffle=False, num_workers=2),
            DataLoader(test_ds, 1, shuffle=False, num_workers=2))
