import yaml

with open("configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_seeds():
    return config["configs"]["random_seed"]

def get_batch_size():
    return config["configs"]["batch_size"]

def get_num_workers():
    return config["configs"]["num_workers"]

def get_num_epochs():
    return config["configs"]["num_epochs"]

def get_learning_rate():
    return config["configs"]["learning_rate"]