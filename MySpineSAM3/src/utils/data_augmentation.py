"""
Data Augmentation with MONAI Transforms
=======================================
Required: RandGaussianNoise is mandatory for all training.
"""

from typing import Dict, Any
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandSpatialCropd, RandFlipd,
    RandRotate90d, RandGaussianNoised, RandZoomd, RandShiftIntensityd, ToTensord,
)


def get_train_transforms(config: Dict[str, Any]) -> Compose:
    """Get training transforms with mandatory RandGaussianNoise."""
    data_cfg = config.get("data", {})
    aug_cfg = config.get("training", {}).get("augmentation", {})
    
    spatial_size = data_cfg.get("spatial_size", [96, 96, 96])
    hu_min, hu_max = data_cfg.get("hu_min", -100), data_cfg.get("hu_max", 1000)
    
    transforms = [
        ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=spatial_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=aug_cfg.get("flip_prob", 0.5), spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=aug_cfg.get("rotation_prob", 0.3), max_k=3),
        # MANDATORY: RandGaussianNoise
        RandGaussianNoised(keys=["image"], prob=aug_cfg.get("noise_prob", 0.5), std=aug_cfg.get("noise_std", 0.1)),
    ]
    
    if aug_cfg.get("use_zoom", True):
        transforms.append(RandZoomd(keys=["image", "label"], prob=aug_cfg.get("zoom_prob", 0.3),
                                     min_zoom=aug_cfg.get("zoom_range", [0.9, 1.1])[0],
                                     max_zoom=aug_cfg.get("zoom_range", [0.9, 1.1])[1]))
    
    if aug_cfg.get("use_intensity_shift", True):
        transforms.append(RandShiftIntensityd(keys=["image"], prob=aug_cfg.get("intensity_prob", 0.3),
                                               offsets=aug_cfg.get("intensity_shift_range", [-0.1, 0.1])[1]))
    
    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)


def get_val_transforms(config: Dict[str, Any]) -> Compose:
    """Get validation transforms (no augmentation)."""
    data_cfg = config.get("data", {})
    hu_min, hu_max = data_cfg.get("hu_min", -100), data_cfg.get("hu_max", 1000)
    
    return Compose([
        ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0, b_max=1, clip=True),
        ToTensord(keys=["image", "label"]),
    ])
