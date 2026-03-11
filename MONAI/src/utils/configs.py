import yaml
from monai.utils import set_determinism

with open("configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_seed():
    seed = config["training"]["random_seed"]
    set_determinism(seed = seed)
    return seed


def get_batch_size():
    return config["training"]["batch_size"]


def get_num_workers():
    return config["training"]["num_workers"]


def get_num_epochs():
    return config["training"]["num_epochs"]


def get_learning_rate():
    return config["training"]["learning_rate"]