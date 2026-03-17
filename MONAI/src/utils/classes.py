import nibabel as nib
from glob import glob
from tqdm import tqdm
import numpy as np
import yaml
from src.utils.configs import config

files = glob(config["data"]["train_masks_dir"] + "/*_0000_seg.nii.gz")
vals = set()

for file in tqdm(files):
    images = nib.load(file)
    vals.update(np.unique(images.get_fdata()))

print(sorted(vals))
print(f"Number of unique values: {len(vals)}")

config["data"]["num_classes"] = len(vals)

with open("configs/configs.yaml", "w") as f:
    yaml.dump(config, f)
