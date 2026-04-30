from monai.networks.nets import SwinUNETR
from src.utils.configs import config

def get_swinunetr():
    params = {
        **config["models"]["swinunetr"],
        **{
            "in_channels": config["models"]["global"]["in_channels"],
            "out_channels": config["models"]["global"]["out_channels"],
            "spatial_dims": config["models"]["global"]["spatial_dims"],
        }
    }

    model = SwinUNETR(**params)
    return model