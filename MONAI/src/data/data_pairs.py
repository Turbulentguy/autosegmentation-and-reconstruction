from glob import glob
import os

def match_pairs(images, masks):
    mask_dict = {
        os.path.basename(m).replace("_0000_seg.nii.gz", ""): m
        for m in masks
    }

    pairs = []
    for img in images:
        key = os.path.basename(img).replace("_0000.nii.gz", "")

        if key in mask_dict:
            pairs.append({
                "image": img,
                "mask": mask_dict[key]
            })
        else:
            print(f"[WARNING] No mask for {img}")

    return pairs

def data_pairs(config):

    train_images_dir = config["data"]["train_images_dir"]
    train_masks_dir = config["data"]["train_masks_dir"]
    val_images_dir = config["data"]["val_images_dir"]
    val_masks_dir = config["data"]["val_masks_dir"]
    test_images_dir = config["data"]["test_images_dir"]
    
    train_images = sorted(glob(os.path.join(train_images_dir, "*_0000.nii.gz")))
    train_masks = sorted(glob(os.path.join(train_masks_dir, "*_0000_seg.nii.gz")))
    val_images = sorted(glob(os.path.join(val_images_dir, "*_0000.nii.gz")))
    val_masks = sorted(glob(os.path.join(val_masks_dir, "*_0000_seg.nii.gz")))
    test_images = sorted(glob(os.path.join(test_images_dir, "*_0000.nii.gz")))

    train_pairs = match_pairs(train_images, train_masks)
    val_pairs = match_pairs(val_images, val_masks)
    test_pairs = [
        {"image": image}
        for image in test_images
    ]

    print("Number of pairs:")
    print(f" Train: {len(train_pairs)}")
    print(f" Validation: {len(val_pairs)}")    
    print(f" Test: {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs