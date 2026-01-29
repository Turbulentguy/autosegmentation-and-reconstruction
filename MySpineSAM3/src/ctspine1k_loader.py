"""
CTSpine1K Dataset Loader
========================
HuggingFace CTSpine1K integration with support for local loading.
Matches LocalNiftiDataset behavior for normalization and cropping.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CTSpine1KDataset(Dataset):
    """PyTorch wrapper for CTSpine1K HuggingFace dataset."""
    
    def __init__(self, hf_dataset, transform=None, hu_min=-100, hu_max=1000, binary_mask=True,
                spatial_size=(96, 96, 96), is_train=True):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.hu_min, self.hu_max = hu_min, hu_max
        self.binary_mask = binary_mask
        self.spatial_size = spatial_size
        self.is_train = is_train
    
    def __len__(self): return len(self.hf_dataset)
    
    def _random_crop(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random spatial crop to spatial_size."""
        _, h, w, d = image.shape
        th, tw, td = self.spatial_size
        
        # If volume is smaller than target, pad it
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        pad_d = max(0, td - d)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            label = np.pad(label, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            h, w, d = image.shape[1:]
        
        # Random crop
        if self.is_train:
            x = random.randint(0, max(0, h - th))
            y = random.randint(0, max(0, w - tw))
            z = random.randint(0, max(0, d - td))
        else:
            # Center crop for val/test
            x = max(0, (h - th) // 2)
            y = max(0, (w - tw) // 2)
            z = max(0, (d - td) // 2)
        
        image = image[:, x:x+th, y:y+tw, z:z+td]
        label = label[:, x:x+th, y:y+tw, z:z+td]
        
        return image, label

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = np.clip(np.array(sample["image"], dtype=np.float32), self.hu_min, self.hu_max)
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)
        label = (np.array(sample["segmentation"]) > 0).astype(np.int64) if self.binary_mask else np.array(sample["segmentation"])
        
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        
        # Random crop
        image, label = self._random_crop(image, label)

        data = {"image": image, "label": label, "patient_id": sample.get("patient_id", f"p{idx}")}
        
        if self.transform: data = self.transform(data)
        if isinstance(data["image"], np.ndarray):
            data["image"] = torch.from_numpy(data["image"].copy())
            data["label"] = torch.from_numpy(data["label"].copy())
        return data


def load_ctspine1k_dataset(mode="3d", cache_dir=None, writer_batch_size=5):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    logger.info(f"Loading CTSpine1K from: {cache_dir}")
    
    # Try loading from local disk first (if it's a saved Arrow dataset)
    if cache_dir and Path(cache_dir).exists():
        try:
            # Check if it looks like a saved dataset (has state.json or dataset_info.json)
            if (Path(cache_dir) / "dataset_info.json").exists() or (Path(cache_dir) / "train").exists():
                logger.info("Detected local Arrow dataset, loading via load_from_disk...")
                return load_from_disk(cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load_from_disk: {e}, falling back to load_dataset")

    # Fallback to loading via script but using cache_dir as data_dir or cache
    # If the user provided a PATH to the dataset files, we might need a loading script.
    # But if they provided a path to the HG cache, we use cache_dir.
    
    # IMPORTANT: If the path is a bare folder of NIfTIs, this won't work.
    # But user says "datasets--alexanderdann--CTSpine1K", which implies HG cache structure.
    return load_dataset("alexanderdann/CTSpine1K", name=mode, cache_dir=cache_dir)


def get_ctspine1k_dataloaders(config, train_transform=None, val_transform=None, mode="3d"):
    data_cfg = config["data"]
    cache_dir = data_cfg.get("ctspine1k", {}).get("cache_dir")
    
    hf = load_ctspine1k_dataset(mode, cache_dir)
    
    spatial_size = tuple(data_cfg.get("spatial_size", [96, 96, 96]))
    kwargs = dict(hu_min=data_cfg.get("hu_min", -100), hu_max=data_cfg.get("hu_max", 1000),
                  binary_mask=data_cfg.get("ctspine1k", {}).get("binary_mask", True),
                  spatial_size=spatial_size)
    
    train_ds = CTSpine1KDataset(hf["train"], train_transform, is_train=True, **kwargs)
    val_ds = CTSpine1KDataset(hf["validation"], val_transform, is_train=False, **kwargs)
    test_ds = CTSpine1KDataset(hf["test"], val_transform, is_train=False, **kwargs)
    
    batch = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    pin_memory = config["training"].get("pin_memory", True)
    persistent_workers = (num_workers > 0)

    return (DataLoader(train_ds, batch, shuffle=True, num_workers=num_workers, drop_last=True,
                       pin_memory=pin_memory, persistent_workers=persistent_workers),
            DataLoader(val_ds, 1, shuffle=False, num_workers=2,
                       pin_memory=pin_memory, persistent_workers=False),
            DataLoader(test_ds, 1, shuffle=False, num_workers=2,
                       pin_memory=pin_memory, persistent_workers=False))
