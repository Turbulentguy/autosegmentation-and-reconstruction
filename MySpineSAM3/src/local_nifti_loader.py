"""
Local NIfTI Dataset Loader for CTSpine1K
=========================================
Loads local NIfTI files using official train/test splits from data_split.txt.
Supports random shuffling within each fixed split for reproducibility.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict
import logging
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def parse_data_split(split_file: Path) -> Dict[str, List[str]]:
    """Parse data_split.txt to get official train/test splits.
    
    Format:
        trainset:
        file1.nii.gz
        file2.nii.gz
        ...
        
        test_public:
        file3.nii.gz
        ...
        
        test_private:
        file4.nii.gz
        ...
    """
    splits = {"trainset": [], "test_public": [], "test_private": []}
    current_split = None
    
    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return splits
    
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_split = line[:-1]
                if current_split not in splits:
                    splits[current_split] = []
            elif current_split:
                # Remove .nii.gz extension to get base name
                base_name = line.replace(".nii.gz", "")
                splits[current_split].append(base_name)
    
    logger.info(f"Parsed splits: trainset={len(splits['trainset'])}, test_public={len(splits['test_public'])}, test_private={len(splits['test_private'])}")
    return splits


class LocalNiftiDataset(Dataset):
    """Load NIfTI images and segmentation masks from local directories.
    
    Uses official CTSpine1K train/test splits with random shuffling within each split.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        hu_min: float = -100,
        hu_max: float = 1000,
        binary_mask: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.hu_min, self.hu_max = hu_min, hu_max
        self.binary_mask = binary_mask
        self.seed = seed
        self.spatial_size = spatial_size
        self.is_train = (split == "train")
        
        # Find pairs and apply official splits
        all_pairs = self._find_all_pairs()
        self.pairs = self._apply_split(all_pairs, split, train_ratio, val_ratio)
        
        logger.info(f"LocalNiftiDataset: {len(self.pairs)} {split} pairs, spatial_size={spatial_size}, binary_mask={binary_mask}")
    
    def _find_all_pairs(self) -> Dict[str, Tuple[Path, Path]]:
        """Find all matching image and segmentation pairs recursively."""
        logger.info(f"Searching for NIfTI files recursively in: {self.data_dir}")
        
        # Find all .nii.gz files
        all_files = list(self.data_dir.rglob("*.nii.gz"))
        
        # Separate into volumes and segmentations
        seg_files = [f for f in all_files if "_seg" in f.name]
        vol_files = {f.name.replace(".nii.gz", ""): f for f in all_files if "_seg" not in f.name}
        
        pairs = {}
        for seg_path in seg_files:
            base_name = seg_path.name.replace("_seg.nii.gz", "")
            if base_name in vol_files:
                pairs[base_name] = (vol_files[base_name], seg_path)
        
        logger.info(f"Found {len(pairs)} matched image-mask pairs from {len(all_files)} total files")
        
        if len(pairs) == 0:
            logger.warning("No pairs found! Check if file names match (e.g. 'case1.nii.gz' and 'case1_seg.nii.gz')")
            
        return pairs
    
    def _apply_split(
        self, 
        all_pairs: Dict[str, Tuple[Path, Path]], 
        split: str, 
        train_ratio: float,
        val_ratio: float,
    ) -> List[Tuple[Path, Path]]:
        """Apply official CTSpine1K splits.
        
        Uses the official data_split.txt which defines:
        - trainset: 610 samples → training
        - test_public: 197 samples → validation
        - test_private: 198 samples → testing
        Total: 1,005 samples
        """
        # Try to load official splits from local data/metadata first
        split_file = Path(__file__).parent.parent / "data" / "metadata" / "data_split.txt"
        if not split_file.exists():
            # Try HuggingFace cache location
            split_file = self.data_dir / "metadata" / "data_split.txt"
        if not split_file.exists():
            for parent in [self.data_dir.parent, self.data_dir]:
                alt_path = parent / "data" / "metadata" / "data_split.txt"
                if alt_path.exists():
                    split_file = alt_path
                    break
        
        official_splits = parse_data_split(split_file)
        
        # Collect pairs for each official split
        def collect_pairs(split_name):
            pairs = []
            missing = 0
            for name in official_splits.get(split_name, []):
                if name in all_pairs:
                    pairs.append(all_pairs[name])
                else:
                    missing += 1
            if missing > 0:
                logger.warning(f"Missing {missing} files from {split_name}")
            return pairs
        
        train_pairs = collect_pairs("trainset")      # 610 samples
        val_pairs = collect_pairs("test_public")     # 197 samples  
        test_pairs = collect_pairs("test_private")   # 198 samples
        
        # Fallback if no official splits
        if not train_pairs and not val_pairs and not test_pairs:
            logger.warning("No official splits found, using ratio-based split")
            all_list = list(all_pairs.values())
            rng = random.Random(self.seed)
            rng.shuffle(all_list)
            n = len(all_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_pairs = all_list[:n_train]
            val_pairs = all_list[n_train:n_train + n_val]
            test_pairs = all_list[n_train + n_val:]
        
        # Shuffle with seed for reproducibility
        rng = random.Random(self.seed)
        rng.shuffle(train_pairs)
        rng.shuffle(val_pairs)
        rng.shuffle(test_pairs)
        
        if split == "train":
            pairs = train_pairs
        elif split == "validation" or split == "val":
            pairs = val_pairs
        else:  # test
            pairs = test_pairs
        
        logger.info(f"Split '{split}': {len(pairs)} samples (train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)})")
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def _random_crop(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random spatial crop to spatial_size with foreground-biased sampling.
        
        For training: 50% of crops center on a foreground (vertebra) voxel.
        This ensures the model sees vertebrae during training instead of mostly background.
        
        Args:
            image: (C, H, W, D) array
            label: (C, H, W, D) array
        
        Returns:
            Cropped image and label of shape (C, *spatial_size)
        """
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
        
        if self.is_train:
            # 50% foreground-biased sampling for training
            use_foreground_center = random.random() < 0.5
            
            if use_foreground_center:
                # Find foreground voxels (any vertebra class > 0)
                fg_coords = np.argwhere(label[0] > 0)
                
                if len(fg_coords) > 0:
                    # Pick a random foreground voxel as center
                    center_idx = random.randint(0, len(fg_coords) - 1)
                    cy, cx, cz = fg_coords[center_idx]
                    
                    # Calculate crop start to center on this voxel
                    x = max(0, min(h - th, cy - th // 2))
                    y = max(0, min(w - tw, cx - tw // 2))
                    z = max(0, min(d - td, cz - td // 2))
                else:
                    # No foreground found, fall back to random
                    x = random.randint(0, max(0, h - th))
                    y = random.randint(0, max(0, w - tw))
                    z = random.randint(0, max(0, d - td))
            else:
                # Random crop (could be background or foreground)
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
        vol_path, seg_path = self.pairs[idx]
        
        # Load NIfTI
        image = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        label = nib.load(str(seg_path)).get_fdata().astype(np.int64)
        
        # Normalize HU values
        image = np.clip(image, self.hu_min, self.hu_max)
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)
        
        # Binary mask if needed
        if self.binary_mask:
            label = (label > 0).astype(np.int64)
        
        # Add channel dimension: (H, W, D) -> (1, H, W, D)
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        
        # Random crop to spatial_size (e.g., 96x96x96)
        image, label = self._random_crop(image, label)
        
        data = {
            "image": image,
            "label": label,
            "patient_id": vol_path.stem.replace(".nii", "")
        }
        
        if self.transform:
            data = self.transform(data)
        
        if isinstance(data["image"], np.ndarray):
            data["image"] = torch.from_numpy(data["image"].copy())
            data["label"] = torch.from_numpy(data["label"].copy())
        
        return data


def get_local_dataloaders(config, train_transform=None, val_transform=None):
    """Create dataloaders from local NIfTI files with official splits."""
    data_cfg = config["data"]
    data_dir = data_cfg.get("local_data_dir", data_cfg.get("root_dir", "./data/ctspine1k_raw"))
    
    kwargs = dict(
        hu_min=data_cfg.get("hu_min", -100),
        hu_max=data_cfg.get("hu_max", 1000),
        # CRITICAL: Default False for multi-class segmentation!
        # Check both possible config paths for binary_mask
        binary_mask=data_cfg.get("ctspine1k", {}).get("binary_mask", 
                    data_cfg.get("binary_mask", False)),
        train_ratio=data_cfg.get("split", {}).get("train", 0.8),
        val_ratio=data_cfg.get("split", {}).get("val", 0.15),
        seed=config.get("project", {}).get("seed", 42),
        spatial_size=tuple(data_cfg.get("spatial_size", [96, 96, 96])),
    )
    
    train_ds = LocalNiftiDataset(data_dir, "train", train_transform, **kwargs)
    val_ds = LocalNiftiDataset(data_dir, "validation", val_transform, **kwargs)
    test_ds = LocalNiftiDataset(data_dir, "test", val_transform, **kwargs)
    
    batch = config["training"]["batch_size"]
    return (
        DataLoader(train_ds, batch, shuffle=True, num_workers=4, drop_last=True),
        DataLoader(val_ds, 1, shuffle=False, num_workers=2),
        DataLoader(test_ds, 1, shuffle=False, num_workers=2),
    )
