import argparse
from src.training.train import train

def main():
    parser = argparse.ArgumentParser(description = "Autosegmentation and reconstruction")

    parser.add_argument("--config",
                        type = str,
                        default = "configs/configs.yaml")

    parser.add_argument("--model",
                        type = str, 
                        default = "unet",
                        help = "Model name")
    
    args = parser.parse_args()
    train(args.config, args.model)

if __name__ == "__main__":
    main()