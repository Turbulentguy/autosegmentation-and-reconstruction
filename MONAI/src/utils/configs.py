import yaml
from monai.utils import set_determinism

DEFAULT_CONFIG_PATH = "configs/configs.yaml"


def load_config(config_path = DEFAULT_CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()


def apply_data_path_overrides(config_data, meta_path = None, images_dir = None, masks_dir = None):
    data = config_data.setdefault("data", {})

    if meta_path is not None:
        data["meta_path"] = meta_path
    if images_dir is not None:
        data["images_dir"] = images_dir
    if masks_dir is not None:
        data["masks_dir"] = masks_dir

    return config_data


def build_effective_config(
    config_path = DEFAULT_CONFIG_PATH,
    meta_path = None,
    images_dir = None,
    masks_dir = None
):
    config_data = load_config(config_path)
    return apply_data_path_overrides(
        config_data,
        meta_path = meta_path,
        images_dir = images_dir,
        masks_dir = masks_dir,
    )


def save_config(config_data, output_path):
    with open(output_path, "w") as f:
        yaml.safe_dump(config_data, f, sort_keys = False)


def set_runtime_config(
    config_path = DEFAULT_CONFIG_PATH,
    meta_path = None,
    images_dir = None,
    masks_dir = None
):
    global config
    config = build_effective_config(
        config_path = config_path,
        meta_path = meta_path,
        images_dir = images_dir,
        masks_dir = masks_dir,
    )
    return config


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