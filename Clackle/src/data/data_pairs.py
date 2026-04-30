from glob import glob
import os

def get_case_id_from_image_path(image_path):
    name = os.path.basename(image_path)
    if name.endswith("_0000.nii.gz"):
        return name[:-len("_0000.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[:-len(".nii.gz")]
    return name

def get_case_id_from_mask_path(mask_path):
    name = os.path.basename(mask_path)
    if name.endswith("_0000_seg.nii.gz"):
        return name[:-len("_0000_seg.nii.gz")]
    if name.endswith("_seg.nii.gz"):
        return name[:-len("_seg.nii.gz")]
    return name

def glob_images(images_dir):
    candidates = glob(os.path.join(images_dir, "*_0000.nii.gz")) + glob(os.path.join(images_dir, "*.nii.gz"))
    return sorted(set(candidates))

def glob_masks(masks_dir):
    candidates = glob(os.path.join(masks_dir, "*_0000_seg.nii.gz")) + glob(os.path.join(masks_dir, "*_seg.nii.gz"))
    return sorted(set(candidates))

def match_pairs(images, masks):
    mask_dict = {
        get_case_id_from_mask_path(m): m
        for m in masks
    }

    pairs = []
    for img in images:
        key = get_case_id_from_image_path(img)

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
    
    train_images = glob_images(train_images_dir)
    train_masks = glob_masks(train_masks_dir)
    val_images = glob_images(val_images_dir)
    val_masks = glob_masks(val_masks_dir)
    test_images = glob_images(test_images_dir)

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