import os
from monai.transforms import SaveImaged
from src.utils.configs import set_runtime_config

def get_saver(config_path, best_model_path = None):
    config = set_runtime_config(config_path)

    if best_model_path:
        run_dir = os.path.dirname(os.path.dirname(best_model_path))
        output_dir = os.path.join(run_dir, "predictions")
    else:
        output_dir = os.path.join(config["outputs"]["outputs_path"], "predictions")

    os.makedirs(output_dir, exist_ok=True)

    saver = SaveImaged(keys = "pred",
                       output_dir = output_dir,
                       output_postfix = "seg",
                       output_ext = ".nii.gz",
                       resample = True,
                       mode = "nearest",
                       separate_folder = True,
                       print_log = True)
    
    return saver