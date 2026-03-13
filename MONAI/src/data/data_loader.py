import yaml
from monai.data import Dataset, DataLoader
from src.preprocess.preprocess import (
    train_transforms,
    val_transforms,
    test_transforms
)

with open("configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["training"]["batch_size"]
num_workers = config["training"]["num_workers"]

def get_train_loader(train_pairs):
    train_ds = Dataset(
        data = train_pairs,
        transform = train_transforms()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = num_workers > 0
    )

    return train_loader

def get_val_loader(val_pairs):
    val_ds = Dataset(
        data = val_pairs,
        transform = val_transforms()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = num_workers > 0
    )

    return val_loader

def get_test_loader(test_pairs):
    test_ds = Dataset(
        data = test_pairs,
        transform = test_transforms()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = num_workers > 0
    )

    return test_loader