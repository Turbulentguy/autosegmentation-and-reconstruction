from glob import glob
import os
from src.utils.configs import config

train_images_dir = config["data"]["train_images_dir"]
train_masks_dir = config["data"]["train_masks_dir"]
val_images_dir = config["data"]["val_images_dir"]
val_masks_dir = config["data"]["val_masks_dir"]
test_images_dir = config["data"]["test_images_dir"]

def data_pairs():
    
    train_images = sorted(glob(os.path.join(train_images_dir, "*_0000.nii.gz")))
    train_masks = sorted(glob(os.path.join(train_masks_dir, "*_0000_seg.nii.gz")))
    val_images = sorted(glob(os.path.join(val_images_dir, "*_0000.nii.gz")))
    val_masks = sorted(glob(os.path.join(val_masks_dir, "*_0000_seg.nii.gz")))
    test_images = sorted(glob(os.path.join(test_images_dir, "*_0000.nii.gz")))

    train_pairs = [
        {"image": image, "mask": mask}
        for image, mask in zip(train_images, train_masks)
    ]

    val_pairs = [
        {"image": image, "mask": mask}
        for image, mask in zip(val_images, val_masks)
    ]

    test_pairs = [
        {"image": image}
        for image in test_images
    ]

    print("Number of pairs:")
    print(f" Train: {len(train_pairs)}")
    print(f" Validation: {len(val_pairs)}")    
    print(f" Test: {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs