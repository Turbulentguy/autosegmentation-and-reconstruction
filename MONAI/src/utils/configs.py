import yaml

DEFAULT_CONFIG_PATH = "configs/configs.yaml"

def load_config(config_path = DEFAULT_CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

def apply_data_path_overrides(config_data, 
                              meta_path = None, 
                              images_dir = None, 
                              masks_dir = None,
                              resume_training_path = None,
                              outputs_path = None
):
    data = config_data.setdefault("data", {})

    if meta_path is not None:
        data["meta_path"] = meta_path
    if images_dir is not None:
        data["images_dir"] = images_dir
    if masks_dir is not None:
        data["masks_dir"] = masks_dir
    if resume_training_path is not None:
        config_data["outputs"]["resume_training_path"] = resume_training_path
    if outputs_path is not None:
        config_data["outputs"]["outputs_path"] = outputs_path

    return config_data

def apply_training_overrides(config_data, 
                             random_seed = None,
                             batch_size = None,
                             num_workers = None,
                             num_epochs = None,
                             learning_rate = None,
                             weight_decay = None
):
    training = config_data.setdefault("training", {})

    if random_seed is not None:
        training["random_seed"] = random_seed
    if batch_size is not None:
        training["batch_size"] = batch_size
    if num_workers is not None:
        training["num_workers"] = num_workers
    if num_epochs is not None:
        training["num_epochs"] = num_epochs
    if learning_rate is not None:
        training["learning_rate"] = learning_rate
    if weight_decay is not None:
        training["weight_decay"] = weight_decay

    return config_data

def build_effective_config(
    config_path = DEFAULT_CONFIG_PATH,
    meta_path = None,
    images_dir = None,
    masks_dir = None,
    resume_training_path = None,
    outputs_path = None,
    random_seed = None,
    batch_size = None,
    num_workers = None,
    num_epochs = None,
    learning_rate = None,
    weight_decay = None
):

    config_data = load_config(config_path)

    config_data = apply_data_path_overrides(
        config_data,
        meta_path = meta_path,
        images_dir = images_dir,
        masks_dir = masks_dir,
        resume_training_path = resume_training_path,
        outputs_path = outputs_path
    )

    config_data = apply_training_overrides(
        config_data,
        random_seed = random_seed,
        batch_size = batch_size,
        num_workers = num_workers,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
        weight_decay = weight_decay
    )

    return config_data

def save_config(config_data, output_path):
    with open(output_path, "w") as f:
        yaml.safe_dump(config_data, f, sort_keys = False)

def set_runtime_config(
    config_path = DEFAULT_CONFIG_PATH,
    meta_path = None,
    images_dir = None,
    masks_dir = None,
    resume_training_path = None,
    outputs_path = None,
    random_seed = None,
    batch_size = None,
    num_workers = None,
    num_epochs = None,
    learning_rate = None,
    weight_decay = None
):
    global config
    config = build_effective_config(
        config_path = config_path,
        meta_path = meta_path,
        images_dir = images_dir,
        masks_dir = masks_dir,
        resume_training_path = resume_training_path,
        outputs_path = outputs_path,
        random_seed = random_seed,
        batch_size = batch_size,
        num_workers = num_workers,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
        weight_decay = weight_decay
    )
    return config