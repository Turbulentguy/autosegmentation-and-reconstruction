import os
from tqdm import tqdm
from src.utils.configs import config

images_dir = config["data"]["images_dir"]
masks_dir = config["data"]["masks_dir"]
meta_path = config["data"]["meta_path"]

def data_split():

    train_images_dir = config["data"]["train_images_dir"]
    train_masks_dir = config["data"]["train_masks_dir"]
    val_images_dir = config["data"]["val_images_dir"]
    val_masks_dir = config["data"]["val_masks_dir"]
    test_images_dir = config["data"]["test_images_dir"]

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    
    images_file_index = {}
    masks_file_index = {}

    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.endswith(".nii.gz"):
                images_file_index[f] = os.path.join(root, f)

    print(f"Indexed {len(images_file_index)} files.")

    for root, _, files in os.walk(masks_dir):
        for f in files:
            if f.endswith("_seg.nii.gz"):
                masks_file_index[f] = os.path.join(root, f)

    print(f"Indexed {len(masks_file_index)} files.")

    train_cases = []
    val_cases = []
    test_cases = []

    mode = None

    with open(meta_path, "r") as files:
        for line in tqdm(files):
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith("trainset:"):
                mode = "training"
                continue
            
            elif line.startswith("test_public:"):
                mode = "validation"
                continue

            elif line.startswith("test_private:"):
                mode = "test"
                continue
                
            filename = line
            case_id = filename.replace(".nii.gz", "")

            if mode == "training":
                train_cases.append(case_id)
            elif mode == "validation":
                val_cases.append(case_id)
            elif mode == "test":
                test_cases.append(case_id)

            source = images_file_index[filename]
            train_destination = os.path.join(train_images_dir, case_id + "_0000.nii.gz")
            train_mask_destination = os.path.join(train_masks_dir, case_id + "_0000_seg.nii.gz")
            validation_destination = os.path.join(val_images_dir, case_id + "_0000.nii.gz")
            validation_mask_destination = os.path.join(val_masks_dir, case_id + "_0000_seg.nii.gz")
            test_destination = os.path.join(test_images_dir, case_id + "_0000.nii.gz")

            if mode == "training":
                if not os.path.exists(train_destination):
                    os.symlink(source, train_destination)
                if not os.path.exists(train_mask_destination):
                    os.symlink(masks_file_index[filename.replace(".nii.gz", "_seg.nii.gz")], train_mask_destination)

            elif mode == "validation":
                if not os.path.exists(validation_destination):
                    os.symlink(source, validation_destination)
                if not os.path.exists(validation_mask_destination):
                    os.symlink(masks_file_index[filename.replace(".nii.gz", "_seg.nii.gz")], validation_mask_destination)

            elif mode == "test":
                if not os.path.exists(test_destination):
                    os.symlink(source, test_destination)
            
    split = [{
        "train": train_cases,
        "validation": val_cases,
        "test": test_cases
    }]

    print("Train cases:", len(train_cases))
    print("Val cases:", len(val_cases))
    print("Test cases:", len(test_cases))
    print("Total cases:", len(train_cases) + len(val_cases) + len(test_cases))
    print("Done.")
