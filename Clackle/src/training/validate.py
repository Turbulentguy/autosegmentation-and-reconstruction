from time import time
import torch
from tqdm import tqdm
from torch.amp import autocast
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

def validate(model, loader, criterion, dice, iou, device, config):
    model.eval()

    num_classes = config["data"]["num_classes_lumbar"]
    post_pred = AsDiscrete(argmax = True, to_onehot = num_classes)
    post_mask = AsDiscrete(to_onehot = num_classes)

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

            with autocast("cuda", enabled = torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()

            outputs_list = decollate_batch(outputs)
            masks_list = decollate_batch(masks)

            preds = [post_pred(out) for out in outputs_list]
            targets = [post_mask(m) for m in masks_list]

            dice(preds, targets)
            iou(preds, targets)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time.append(time() - t0)

    val_loss /= len(loader)
    val_dice_score = dice.aggregate()
    val_iou_score = iou.aggregate()

    val_mean_dice = torch.nanmean(val_dice_score).item()
    val_mean_iou = torch.nanmean(val_iou_score).item()
    dice.reset()
    iou.reset()

    return val_loss, val_dice_score, val_iou_score, val_mean_dice, val_mean_iou, batch_time
