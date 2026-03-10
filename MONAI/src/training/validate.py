from time import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

def validate(model, loader, criterion, dice, iou, time, device):
    model.eval()

    dice.reset()
    iou.reset()
    val_loss = 0.0
    time = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            torch.cuda.synchronize()
            t0 = time()

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()

            pred = outputs.argmax(dim = 1)

            dice(pred, masks)
            iou(pred, masks)

            torch.cuda.synchronize()
            time.append(time() - t0)
    
    val_loss /= len(loader)
    val_dice = dice.aggregate().item()
    val_iou = iou.aggregate().item()

    dice.reset()
    iou.reset()

    return val_loss, val_dice, val_iou, time
