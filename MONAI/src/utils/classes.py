import nibabel as nib
from glob import glob
from tqdm import tqdm
import numpy as np
import yaml

with open("configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)

files = glob(config["data"]["train_masks_dir"] + "/*_0000_seg.nii.gz")
vals = set()

for file in tqdm(files):
    images = nib.load(file)
    vals.update(np.unique(images.get_fdata()))

print(sorted(vals))
print(f"Number of unique values: {len(vals)}")

config["data"]["num_classes"] = len(vals)

with open("config/config.yaml", "w") as f:
    yaml.dump(config, f)
