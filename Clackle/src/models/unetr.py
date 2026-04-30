from monai.networks.nets import UNETR
from src.utils.configs import config

def get_unetr():
    params = {
            **config["models"]["global"],
            **config["models"]["unetr"]
    }

    model = UNETR(**params)
    return model