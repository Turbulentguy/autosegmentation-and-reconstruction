import numpy as np
import torch
from src.utils.configs import config

def build_lookup(config):
    lumbar_classes = config["data"]["lumbar_classes"]
    remapped_lumbar_classes = config["data"]["remapped_lumbar_classes"]

    max_label = max(lumbar_classes)
    lookup = np.zeros(max_label + 1, dtype = np.int16)

    for origin, remapped in zip(lumbar_classes, remapped_lumbar_classes):
        lookup[origin] = remapped

    return lookup


def label_remap(mask):
    lookup = build_lookup(config)

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.int32)
    remapped = lookup[mask]

    return remapped