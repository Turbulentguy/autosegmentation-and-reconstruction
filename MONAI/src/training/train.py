import os
import yaml
import torch
from tqdm import tqdm
from time import time
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

from src.data.data_loader import get_train_loader, get_val_loader
from src.data.data_pairs import data_pairs
from src.models.models import build_model
from src.metrics.metrics import get_dice, get_iou
from src.training.validate import validate
from src.loss.loss import get_loss

def train(config_path, model_name = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["random_seed"]
    set_determinism(seed=seed)

    os.makedirs("outputs/checkpoint", exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_name or config["models"]["name"]
    num_classes = config["data"]["num_classes_lumbar"]
    class_names = config["data"]["lumbar_class_names"]

    train_pairs, val_pairs, _ = data_pairs(config)
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
    
    post_pred = AsDiscrete(argmax = True, to_onehot = num_classes)
    post_mask = AsDiscrete(to_onehot = num_classes)
    
    print(f"================Training: {model_name}================")
    for epoch in range(1, max_epochs + 1):

        model.train()
        train_loss = 0.0
        batch_times = []

        train_dice.reset()
        train_iou.reset()

        for batch in tqdm(train_loader, desc = f"Epoch {epoch}/{max_epochs}"):
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time()

            images = batch["image"].to(device, non_blocking = True)
            masks = batch["mask"].to(device, non_blocking = True)

            optimizer.zero_grad(set_to_none = True)

            with autocast(enabled = torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            preds = post_pred(outputs)
            masks = post_mask(masks)

            train_dice(preds, masks)
            train_iou(preds, masks)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_times.append(time() - t0)

        train_loss /= len(train_loader)

        train_dice_score = train_dice.aggregate()
        train_iou_score = train_iou.aggregate()

        mean_dice = train_dice_score.mean().item()
        mean_iou = train_iou_score.mean().item()

        print("Training Per-class metrics:")
        for name, dice_score, iou_score in zip(class_names, train_dice_score, train_iou_score):
            print(f"Class = {name}: Dice = {dice_score.item():.4f}, IoU = {iou_score.item():.4f}")

        train_dice.reset()
        train_iou.reset()

        val_loss, val_dice_class, val_iou_class, val_mean_dice, val_mean_iou, val_batch_times = validate(model, 
                                            val_loader,
                                            criterion,
                                            val_dice_metric,
                                            val_iou_metric,
                                            device)

        print("Validation Per-class metrics:")
        for name, dice_score, iou_score in zip(class_names, val_dice_class, val_iou_class):
            print(f"Class = {name}: Dice = {dice_score.item():.4f}, IoU = {iou_score.item():.4f}")

        tag = ""
        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            torch.save(model.state_dict(), f"outputs/checkpoint/{model_name}_best_model.pt")
            tag = " (best)"

        print(f"Epoch {epoch} / {max_epochs} | "
              f"Train Mean Loss: {train_loss:.4f} | "
              f"Train Mean Dice: {mean_dice:.4f} | Train Mean IoU: {mean_iou:.4f} | "
              f"Inference time: {sum(batch_times) / len(batch_times):.3f} s | "
              f"Val Mean Loss: {val_loss:.4f} | "
              f"Val Mean Dice: {val_mean_dice:.4f} | Val Mean IoU: {val_mean_iou:.4f} | "
              f"Val Batch Time: {sum(val_batch_times) / len(val_batch_times):.3f} s {tag}")


