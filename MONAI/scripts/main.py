import argparse
import tempfile
from src.training.train import train
from src.utils.configs import build_effective_config, save_config

def main():
    parser = argparse.ArgumentParser(description = "Autosegmentation and reconstruction")

    parser.add_argument("--config",
                        type = str,
                        default = "configs/configs.yaml")

    parser.add_argument("--model",
                        type = str, 
                        default = "unet",
                        help = "Model name")

    parser.add_argument("--meta_path",
                        type = str,
                        default = None,
                        help = "Override data.meta_path from config")

    parser.add_argument("--images_dir",
                        type = str,
                        default = None,
                        help = "Override data.images_dir from config")

    parser.add_argument("--masks_dir",
                        type = str,
                        default = None,
                        help = "Override data.masks_dir from config")
    
    args = parser.parse_args()

    effective_config = build_effective_config(
        config_path = args.config,
        meta_path = args.meta_path,
        images_dir = args.images_dir,
        masks_dir = args.masks_dir,
    )

    with tempfile.NamedTemporaryFile(mode = "w", suffix = ".yaml", delete = False) as temp_config:
        save_config(effective_config, temp_config.name)
        effective_config_path = temp_config.name

    train(effective_config_path, args.model)

if __name__ == "__main__":
    main()