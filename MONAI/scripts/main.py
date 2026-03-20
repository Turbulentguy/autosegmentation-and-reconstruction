import argparse
import tempfile
from src.training.train import train
from src.test.test import test
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
                        help = "Path to metadata")

    parser.add_argument("--images_dir",
                        type = str,
                        default = None,
                        help = "Path to image directory")

    parser.add_argument("--masks_dir",
                        type = str,
                        default = None,
                        help = "Path to mask directory")
    
    parser.add_argument("--resume_training_path",
                        type = str,
                        default = None,
                        help = "Path to checkpoint to resume training from")
    
    parser.add_argument("--outputs_path",
                        type = str,
                        default = None,
                        help = "Path to outputs directory")
    
    parser.add_argument("--best_model_path",
                        type = str,
                        default = None,
                        help = "Path to best model checkpoint for testing")
    
    parser.add_argument("--only_test", 
                        action="store_true")
    
    parser.add_argument("--random_seed",
                        type = int,
                        default = None,
                        help = "Random seed")
    
    parser.add_argument("--batch_size",
                        type = int,
                        default = None,
                        help = "Batch size")
    
    parser.add_argument("--num_workers",
                        type = int,
                        default = None,
                        help = "Number of workers")

    parser.add_argument("--num_epochs",
                        type = int,
                        default = None,
                        help = "Number of epochs")

    parser.add_argument("--learning_rate",
                        type = float,
                        default = None,
                        help = "Learning rate")

    parser.add_argument("--weight_decay",
                        type = float,
                        default = None,
                        help = "Weight decay")
    


    args = parser.parse_args()

    effective_config = build_effective_config(
        config_path = args.config,
        meta_path = args.meta_path,
        images_dir = args.images_dir,
        masks_dir = args.masks_dir,
        resume_training_path = args.resume_training_path,
        outputs_path = args.outputs_path,
        best_model_path = args.best_model_path,
        batch_size = args.batch_size,
        random_seed = args.random_seed,
        num_workers = args.num_workers,
        num_epochs = args.num_epochs,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
    )

    with tempfile.NamedTemporaryFile(mode = "w", suffix = ".yaml", delete = False) as temp_config:
        save_config(effective_config, temp_config.name)
        effective_config_path = temp_config.name

    if not args.only_test:
        trained_model_path = train(effective_config_path, args.model, args.resume_training_path)
    else:
        trained_model_path = None

    if args.only_test and args.best_model_path is None:
        raise ValueError("You must provide --best_model_path when using --only_test")

    best_test_model_path = args.best_model_path or trained_model_path

    test(effective_config_path, args.model, best_test_model_path)

if __name__ == "__main__":
    main()