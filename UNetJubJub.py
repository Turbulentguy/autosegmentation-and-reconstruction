import glob
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import warnings
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  
from tqdm.auto import tqdm
import ignite
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from monai.config import print_config
from monai.data import (
    ArrayDataset,
    create_test_image_3d,
    decollate_batch,
    DataLoader,
    Dataset,
    CacheDataset,  
    PersistentDataset,  
    PydicomReader,
    pad_list_data_collate,
)
from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    EnsureType,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
    Lambda,
    LoadImaged,
    ScaleIntensityd,
    EnsureChannelFirstd,
    Resized,
    EnsureTyped,
    Lambdad,
    RandSpatialCropd,
    RandFlipd,
    RandCropByPosNegLabeld,  
    RandRotate90d,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    CenterSpatialCropd,
)
from monai.utils import first
from datasets import load_dataset
import datasets
from huggingface_hub import snapshot_download
import random
import nibabel as nib
from tqdm import tqdm
from collections import Counter

def stem(path):
  name = os.path.basename(path)
  if name.endswith("_seg.nii.gz"):
    return name[:-len("_seg.nii.gz")]
  elif name.endswith(".nii.gz"):
    return name[:-len(".nii.gz")]


def build_pair(images, segments):
  # Remove files name format
  img_dicts = {stem(p): p for p in images}
  seg_dicts = {stem(p): p for p in segments}
  
  common_keys = sorted(set(img_dicts) & set(seg_dicts))
  
  pairs = [
    {"image": img_dicts[k], "label": seg_dicts[k]}
    for k in common_keys
  ]
  return pairs

def remap_labels(x):
    y = np.zeros_like(x)
    for old, new in label_maps.items():
        y[x == old] = new
    return y
    
def validation(model, loader):
  model.eval()
  
  val_loss = 0.0
  dice_metric.reset()
  iou_metric.reset()
  
  with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", leave=False):
            images = batch["image"].cuda()
            labels = batch["label"].cuda()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            preds = [post_pred(o) for o in decollate_batch(outputs)]
            gts   = [post_label(l) for l in decollate_batch(labels)]

            dice_metric(y_pred=preds, y=gts)
            iou_metric(y_pred=preds, y=gts)

  mean_loss = val_loss / len(loader)
  mean_dice = dice_metric.aggregate().item()
  mean_iou  = iou_metric.aggregate().item()

  return mean_loss, mean_dice, mean_iou
  
  

meta = pd.read_excel("metadata_colonog.xlsx")
ids = meta["Patient Id"].astype(str).tolist()

images_dir = "/lustrefs/disk/project/lt200431-ddmmss/wat/datasets--alexanderdann--CTSpine1K/snapshots/9b454add169b94f2c322ad6f08b66823975e8dbd/raw_data/volumes/COLONOG/"
segments_dir = "/lustrefs/disk/project/lt200431-ddmmss/wat/datasets--alexanderdann--CTSpine1K/snapshots/9b454add169b94f2c322ad6f08b66823975e8dbd/raw_data/labels/COLONOG/"
directory = "test"
images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
segments = sorted(glob.glob(os.path.join(segments_dir, "*_seg.nii.gz")))

print(f"Number of images: {len(images)}")
print(f"Number of masks: {len(segments)}")
num_classes = 19

"""
classes = set()

for i in range(len(segments)):
  load_segment = nib.load(segments[i])
  segments_info = load_segment.get_fdata()
  classes.update(np.unique(segments_info))

classes = sorted(classes)
print(f"All classes: {classes}")
print(f"Number of classes: {len(classes)}")
"""

inconsecutive_labels = [0, 1, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
label_maps = {v: i for i, v in enumerate(inconsecutive_labels)}

pairs = build_pair(images, segments)
print(f"Number of pairs: {len(pairs)}")
print("Sample pair:")
print(f"  Image: {os.path.basename(pairs[0]['image'])}")
print(f"  Label: {os.path.basename(pairs[0]['label'])}")

filtered_pairs = [
    p for p in pairs
    if stem(p["image"]) in ids
]

print(f"Number of filtered pairs: {len(filtered_pairs)}")

random.seed(42)
random.shuffle(filtered_pairs)

train_size = 460
val_size   = 58
test_size  = 57

train_pairs = filtered_pairs[:train_size]
val_pairs   = filtered_pairs[train_size : train_size + val_size]
test_pairs  = filtered_pairs[train_size + val_size : train_size + val_size + test_size]

print(f"\nData Split:")
print(f"  Training:   {len(train_pairs)} pairs")
print(f"  Validation: {len(val_pairs)} pairs")
print(f"  Testing:    {len(test_pairs)} pairs")
print(f"  Total:      {len(train_pairs) + len(val_pairs) + len(test_pairs)} pairs")

train_preprocess = Compose([
  LoadImaged(keys = ["image","label"]),
  EnsureChannelFirstd(keys = ["image", "label"]),
  Orientationd(keys = ["image", "label"], 
              axcodes = "RAS"),
  Spacingd(keys = ["image", "label"],
          pixdim = (1.0, 1.0, 1.0),
          mode = ("bilinear", "nearest")),
  ScaleIntensityRanged(keys = ["image"],
                      a_min = -1000, a_max = 1000,
                      b_min = 0.0, b_max = 1.0,
                      clip = True),
  SpatialPadd(keys = ["image", "label"],
            spatial_size = (128, 128, 128)),
  RandCropByPosNegLabeld(keys = ["image", "label"],
                        label_key = "label",
                        spatial_size = (128, 128, 128),
                        pos = 2,
                        neg = 1,
                        num_samples = 1),
  RandFlipd(keys = ["image", "label"],
           prob = 0.5, 
           spatial_axis=0),
  RandFlipd(keys = ["image", "label"], 
          prob = 0.5, 
          spatial_axis=1),
  RandFlipd(keys = ["image", "label"], 
          prob = 0.5, 
          spatial_axis=2),
  RandRotate90d(keys = ["image", "label"], 
          prob = 0.5, 
          max_k=3),
  RandShiftIntensityd(keys = ["image"], 
          offsets = 0.1, 
          prob=0.5),
  EnsureTyped(keys = ["image", "label"], 
            data_type="tensor", 
            track_meta=False),
  Lambdad(keys = ["label"], 
          func = remap_labels),
])

val_test_preprocess = Compose([
  LoadImaged(keys = ["image","label"]),
  EnsureChannelFirstd(keys = ["image", "label"]),
  Orientationd(keys = ["image", "label"], 
              axcodes = "RAS"),
  Spacingd(keys = ["image", "label"],
          pixdim = (1.0, 1.0, 1.0),
          mode = ("bilinear", "nearest")),
  ScaleIntensityRanged(keys = ["image"],
                      a_min = -1000, a_max = 1000,
                      b_min = 0.0, b_max = 1.0,
                      clip = True),
  SpatialPadd(
              keys = ["image", "label"],
              spatial_size = None,
              divisible = (16, 16, 16)
  ),
  CenterSpatialCropd(keys = ["image", "label"],
                    roi_size=(128, 128, 128)
  ),
  EnsureTyped(keys = ["image", "label"], 
            data_type="tensor", 
            track_meta=False),
  Lambdad(keys = ["label"], 
          func = remap_labels),
])

train_ds = Dataset(
    data=train_pairs,
    transform=train_preprocess
)

val_ds = Dataset(
    data=val_pairs,
    transform=val_test_preprocess
)

test_ds = Dataset(
    data=test_pairs,
    transform=val_test_preprocess
)

train_loader = DataLoader(
  train_ds,
  batch_size = 4,
  shuffle = True,
  num_workers = 4,
  pin_memory = True
)

val_loader = DataLoader(
  val_ds,
  batch_size = 4,
  shuffle = True,
  num_workers = 4,
  pin_memory = True,
  collate_fn = pad_list_data_collate
)

test_loader = DataLoader(
  test_ds,
  batch_size = 4,
  shuffle = True,
  num_workers = 4,
  pin_memory = True,
  collate_fn = pad_list_data_collate
)

batch = next(iter(train_loader))

image = batch["image"]   
label = batch["label"]

print("Image shape:", image.shape)
print("Label shape:", label.shape)
print("Label unique:", torch.unique(label))

image = image[0, 0]  
label = label[0, 0]

x = image.shape[0] // 2
y = image.shape[1] // 2
z = image.shape[2] // 2

views = {
    "axial":     (image[:, :, z], label[:, :, z]),
    "coronal":  (image[:, y, :], label[:, y, :]),
    "sagittal": (image[x, :, :], label[x, :, :]),
}

plt.figure(figsize=(12, 4))

for i, (name, (img, lbl)) in enumerate(views.items()):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap="gray")
    plt.imshow(lbl, alpha=0.4)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.savefig("sanity_check_3views.png", dpi=150)
plt.close()

"""
classes_count = Counter()
for batch in tqdm(train_loader, desc="Counting class voxels"):
    label = batch["label"]         
    label = label.long()
    
    for c in torch.unique(label):
      c = int(c.item())
      classes_count[c] += torch.sum(label == c).item()
      
print("\nClass voxel distribution:")
total = sum(classes_count.values())

for c in sorted(classes_count):
    count = classes_count[c]
    ratio = count / total * 100
    print(f"Class {c:2d}: {count:>12,d} voxels ({ratio:6.2f}%)")
"""

model = UNet(spatial_dims = 3,
            in_channels = 1,
            out_channels = num_classes,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2, 
).cuda()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-4,
    weight_decay = 1e-5
)

scaler = GradScaler()

loss_fn = DiceCELoss(include_background = False,
    to_onehot_y = True,
    softmax = True,
    lambda_dice = 1.0,
    lambda_ce = 0.5
)

dice_metric = DiceMetric(include_background = False,
                        reduction = "mean"
)

iou_metric = MeanIoU(
    include_background=False,
    reduction="mean"
)

post_pred = AsDiscrete(argmax = True, to_onehot = num_classes)
post_label = AsDiscrete(to_onehot = num_classes)

best_val_dice = 0.0
max_epochs = 99

for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0.0
    batch_times = []

    dice_metric.reset()
    iou_metric.reset()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        torch.cuda.synchronize()
        t0 = time.time()

        images = batch["image"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        batch_times.append(time.time() - t0)

        train_loss += loss.item()

        preds = [post_pred(o) for o in decollate_batch(outputs.detach())]
        gts   = [post_label(l) for l in decollate_batch(labels)]

        dice_metric(y_pred=preds, y=gts)
        iou_metric(y_pred=preds, y=gts)

    # ---- train summary ----
    train_loss /= len(train_loader)
    train_dice = dice_metric.aggregate().item()
    train_iou  = iou_metric.aggregate().item()

    avg_bt = np.mean(batch_times)
    p95_bt = np.percentile(batch_times, 95)

    val_loss, val_dice, val_iou = validation(model, val_loader)

    tag = ""
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_model.pt")
        tag = " <-- BEST"

    print(
      f"Epoch: {epoch:03d} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} | "
      f"Val Loss: {val_loss:.4f} | "
      f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
      f"Inference time: {avg_bt:.3f} s)"
      f"{tag}"
    )

