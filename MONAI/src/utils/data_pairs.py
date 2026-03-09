from glob import glob
import os
import yaml

def data_pairs():
    with open("configs/configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_images = glob(os.path.join(config["data"]["train_images_dir"], "*_0000.nii.gz"))
    train_masks = glob(os.path.join(config["data"]["train_masks_dir"], "*_0000_seg.nii.gz"))
    val_images = glob(os.path.join(config["data"]["val_images_dir"], "*_0000.nii.gz"))
    val_masks = glob(os.path.join(config["data"]["val_masks_dir"], "*_0000_seg.nii.gz"))
    test_images = glob(os.path.join(config["data"]["test_images_dir"], "*_0000.nii.gz"))

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

    config["data"]["train_pairs"] = len(train_pairs)
    config["data"]["val_pairs"] = len(val_pairs)
    config["data"]["test_pairs"] = len(test_pairs)

    with open("configs/config.yaml", "w") as f:
        yaml.dump(config, f)

    print("Number of pairs:")
    print(f" Train: {len(train_pairs)}")
    print(f" Validation: {len(val_pairs)}")    
    print(f" Test: {len(test_pairs)}")