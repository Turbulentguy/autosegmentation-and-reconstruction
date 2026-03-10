from time import time
import torch
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

from src.data.data_loader import get_train_loader, get_val_loader
from src.data.data_pairs import data_pairs
from src.models.models import build_model
from src.metrics.metrics import get_dice, get_iou
from src.training.validate import validate
from src.loss.loss import get_loss

def train():
    with open("configs/configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config["models"]["name"]

    train_pairs, val_pairs, _ = data_pairs()
    train_loader = get_train_loader(train_pairs)
    val_loader = get_val_loader(val_pairs)

    model = build_model(model_name).to(device)
    criterion = get_loss()
    train_dice = get_dice()
    train_iou = get_iou()
    val_dice_metric = get_dice()
    val_iou_metric = get_iou()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config["training"]["learning_rate"],
        weight_decay = config["training"]["weight_decay"]
    )

    scaler = GradScaler()
    max_epochs = config["training"]["num_epochs"]
    best_val_dice = 0.0
    
    print(f"================Training: {model_name}================")
    for epoch in range(1, max_epochs + 1):

        model.train()
        train_loss = 0.0
        batch_times = []
        val_batch_times = []

        train_dice.reset()
        train_iou.reset()

        for batch in tqdm(train_loader, desc = f"Epoch {epoch}/{max_epochs}"):

            torch.cuda.synchronize()
            t0 = time()

            images = batch["image"].to(device, non_blocking = True)
            masks = batch["mask"].to(device, non_blocking = True)

            optimizer.zero_grad(set_to_none = True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            preds = torch.argmax(outputs, dim = 1)

            train_dice(preds, masks)
            train_iou(preds, masks)

            torch.cuda.synchronize()
            batch_times.append(time() - t0)

        train_loss /= len(train_loader)

        train_dice_score = train_dice.aggregate().item()
        train_iou_score = train_iou.aggregate().item()

        train_dice.reset()
        train_iou.reset()

        val_loss, val_dice, val_iou, val_batch_times = validate(model, 
                                            val_loader,
                                            criterion,
                                            val_dice_metric,
                                            val_iou_metric,
                                            val_batch_times,
                                            device)

        tag = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"outputs/checkpoint/{model_name}_best_model.pt")
            tag = " (best)"

        print(f"Epoch {epoch} / {max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Dice: {train_dice_score:.4f} | Train IoU: {train_iou_score:.4f} | "
              f"Inference time: {sum(batch_times):.3f} s | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
              f"Val Batch Time: {sum(val_batch_times):.3f} s {tag}")


