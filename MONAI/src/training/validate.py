from time import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from monai.transforms import AsDiscrete
import yaml

with open("configs/configs.yaml", "r") as f:
    config = yaml.safe_load(f)

num_classes = config["data"]["num_classes_lumbar"]
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
post_mask = AsDiscrete(to_onehot=num_classes)

def validate(model, loader, criterion, dice, iou, device):
    model.eval()

    dice.reset()
    iou.reset()
    val_loss = 0.0
    batch_time = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time()

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with autocast(enabled = torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()

            preds = post_pred(outputs)
            masks_onehot = post_mask(masks)

            dice(preds, masks_onehot)
            iou(preds, masks_onehot)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time.append(time() - t0)

    val_loss /= len(loader)
    val_dice_class = dice.aggregate()
    val_iou_class = iou.aggregate()

    val_mean_dice = val_dice_class.mean().item()
    val_mean_iou = val_iou_class.mean().item()
    dice.reset()
    iou.reset()

    return val_loss, val_dice_class, val_iou_class, val_mean_dice, val_mean_iou, batch_time
