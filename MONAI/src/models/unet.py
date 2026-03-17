from monai.networks.nets import UNet
from src.utils.configs import config

def get_unet():
    params = {
            **config["models"]["unet"],
            **{
                "in_channels": config["models"]["global"]["in_channels"],
                "out_channels": config["models"]["global"]["out_channels"],
                "spatial_dims": config["models"]["global"]["spatial_dims"],
            }
    }

    model = UNet(**params)
    return model