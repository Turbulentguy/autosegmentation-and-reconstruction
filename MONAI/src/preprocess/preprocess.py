import yaml
from monai.transforms import (
    LoadImaged, 
    Compose, 
    EnsureChannelFirstd, 
    CropForegroundd, 
    RandCropByPosNegLabeld, 
    Orientationd, 
    Spacingd, 
    ScaleIntensityRangePercentilesd,
    Lambdad
)

from src.utils.label_remap import label_remap

with open("../configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)

def train_transforms():
    train_transforms = Compose([
        LoadImaged(keys = ["image","mask"]),
        ScaleIntensityRangePercentilesd(keys = ["image"],
                                        lower = 5,
                                        upper = 99,
                                        b_min = 0,
                                        b_max = 1,
                                        clip = True
        ),
        EnsureChannelFirstd(keys = ["image", "mask"]),
        Orientationd(keys = ["image", "mask"], 
                axcodes = "RAS"),
        Spacingd(keys = ["image", "mask"],
                pixdim = (1.0, 1.0, 1.0),
                mode = ("trilinear", "nearest")),
        Lambdad(keys = ["mask"], 
                func = label_remap),
        CropForegroundd(keys = ["image", "mask"], 
                        source_key = "image", 
                        margin = 10),
        RandCropByPosNegLabeld(keys = ["image", "mask"], 
                            label_key = "mask", 
                            spatial_size = (160, 160, 160), 
                            pos = 3, 
                            neg = 1,
                            num_samples = 4),
    ])
    return train_transforms

def val_transforms():
    val_transforms = Compose([
        LoadImaged(keys = ["image","mask"]),
        ScaleIntensityRangePercentilesd(keys = ["image"],
                                        lower = 5,
                                        upper = 99,
                                        b_min = 0,
                                        b_max = 1,
                                        clip = True
        ),
        EnsureChannelFirstd(keys = ["image", "mask"]),
        Orientationd(keys = ["image", "mask"], 
                axcodes = "RAS"),
        Spacingd(keys = ["image", "mask"],
                pixdim = (1.0, 1.0, 1.0),
                mode = ("trilinear", "nearest")),
        Lambdad(keys = ["mask"], 
                func = label_remap), 
        CropForegroundd(keys = ["image", "mask"], 
                        source_key = "image", 
                        margin = 10)
    ])
    return val_transforms

def test_transforms():
    test_transforms = Compose([
        LoadImaged(keys = ["image","mask"]),
        ScaleIntensityRangePercentilesd(keys = ["image"],
                                        lower = 5,
                                        upper = 99,
                                        b_min = 0,
                                        b_max = 1,
                                        clip = True
        ), 
        EnsureChannelFirstd(keys = ["image", "mask"]),
        Orientationd(keys = ["image", "mask"], 
                axcodes = "RAS"),
        Spacingd(keys = ["image", "mask"],
                pixdim = (1.0, 1.0, 1.0),
                mode = ("trilinear", "nearest")),
        Lambdad(keys = ["mask"], 
                func = label_remap),
        CropForegroundd(keys = ["image", "mask"], 
                        source_key = "image", 
                        margin = 10),
    ])
    return test_transforms