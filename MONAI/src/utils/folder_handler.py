import os
from src.utils.configs import config

def folder_handler(model_name):
    base_dir = os.path.join(config["outputs"]["outputs_path"], model_name)
    os.makedirs(base_dir, exist_ok=True)

    runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    run_ids = [int(d.split("_")[1]) for d in runs] if runs else [0]
    run_id = max(run_ids) + 1

    run_dir = os.path.join(base_dir, f"run_{run_id:03d}")

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok = True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok = True)
    os.makedirs(os.path.join(run_dir, "tensorboard"), exist_ok = True)
    os.makedirs(os.path.join(run_dir, "csv"), exist_ok = True)
    
    return run_dir