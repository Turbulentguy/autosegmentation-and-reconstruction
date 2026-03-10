import yaml
from monai.networks.nets import UNETR

with open("configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_unetr():
    params = {
            **config["models"]["global"],
            **config["models"]["unetr"]
    }

    model = UNETR(**params)
    return model