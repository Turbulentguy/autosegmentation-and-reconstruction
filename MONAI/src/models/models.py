def build_model(model_name):
    model_name = model_name.lower()
    if model_name == "unet":
        from .unet import get_unet
        return get_unet()
    elif model_name == "unetr":
        from .unetr import get_unetr
        return get_unetr()
    elif model_name == "swinunetr":
        from .swinunetr import get_swinunetr
        return get_swinunetr()
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Please choose from 'unet', 'unetr', or 'swinunetr'.")
