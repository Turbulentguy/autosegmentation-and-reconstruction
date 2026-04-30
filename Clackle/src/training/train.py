import os
import torch
from tqdm import tqdm
from time import time
from torch.amp import GradScaler, autocast
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.utils import set_determinism

from src.utils.configs import set_runtime_config
from src.data.data_loader import get_train_loader, get_val_loader
from src.data.data_pairs import data_pairs
from src.models.models import build_model
from src.metrics.metrics import get_dice, get_iou
from src.training.validate import validate
from src.loss.loss import get_loss
from src.utils.folder_handler import folder_handler
from src.utils.logging import (
    create_tensorboard,
    create_csv,
    log_tensorboard,
    log_csv
)

def train(config_path, model_name = None, resume_path = None):
    config = set_runtime_config(config_path = config_path)

    seed = config["training"]["random_seed"]
    set_determinism(seed = seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_name or config["models"]["name"]
    num_classes = config["data"]["num_classes_lumbar"]
    class_names = config["data"]["lumbar_class_names"]

    if resume_path is not None:
        run_dir = os.path.dirname(os.path.dirname(resume_path))
    else:
        run_dir = folder_handler(model_name)

    writer = create_tensorboard(run_dir)
    csv_path = create_csv(run_dir)

    train_pairs, val_pairs, _ = data_pairs(config)
    train_loader = get_train_loader(train_pairs, config)
    val_loader = get_val_loader(val_pairs, config)

    model = build_model(model_name).to(device)
    criterion = get_loss()
    train_dice = get_dice()
    train_iou = get_iou()
    val_dice_metric = get_dice()
    val_iou_metric = get_iou()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = float(config["training"]["learning_rate"]),
        weight_decay = float(config["training"]["weight_decay"])
    )

    scaler = GradScaler(enabled = torch.cuda.is_available())
    max_epochs = config["training"]["num_epochs"]
    best_model_path = None
    best_val_dice = 0.0
    start_epoch = 1

    if resume_path is not None:
        if os.path.isfile(resume_path):
            print(f"Resuming training from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location = device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            print(f"Resumed from epoch {checkpoint['epoch']}, best_val_dice: {best_val_dice:.4f}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    
    post_pred = AsDiscrete(argmax = True, to_onehot = num_classes)
    post_mask = AsDiscrete(to_onehot = num_classes)
    
    print(f"================Training: {model_name}================")
    for epoch in range(start_epoch, max_epochs + 1):

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

            with autocast("cuda", enabled = torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            outputs_list = decollate_batch(outputs)
            masks_list = decollate_batch(masks)

            preds = [post_pred(out) for out in outputs_list]
            targets = [post_mask(m) for m in masks_list]
            
            train_dice(preds, targets)
            train_iou(preds, targets)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_times.append(time() - t0)

        train_loss /= len(train_loader)

        train_dice_score = train_dice.aggregate()
        train_iou_score = train_iou.aggregate()

        mean_dice = torch.nanmean(train_dice_score).item()
        mean_iou = torch.nanmean(train_iou_score).item()

        train_dice_per_class = torch.nanmean(train_dice_score, dim = 0)
        train_iou_per_class = torch.nanmean(train_iou_score, dim = 0)

        print("Training Per-class metrics:")
        for name, dice_score, iou_score in zip(class_names[1:], train_dice_per_class, train_iou_per_class):
            print(f"Class = {name}: Dice = {dice_score.item():.4f}, IoU = {iou_score.item():.4f}")

        train_dice.reset()
        train_iou.reset()

        val_loss, val_dice_class, val_iou_class, val_mean_dice, val_mean_iou, val_batch_times = validate(model, 
                                            val_loader,
                                            criterion,
                                            val_dice_metric,
                                            val_iou_metric,
                                            device,
                                            config
        )
        
        val_dice_per_class = torch.nanmean(val_dice_class, dim = 0)
        val_iou_per_class = torch.nanmean(val_iou_class, dim = 0)

        print("Validation Per-class metrics:")
        for name, dice_score, iou_score in zip(class_names[1:], val_dice_per_class, val_iou_per_class):
            print(f"Class = {name}: Validation Dice = {dice_score.item():.4f}, Validation IoU = {iou_score.item():.4f}")

        tag = ""
        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            best_model_path = os.path.join(run_dir, "checkpoints", f"{model_name}_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_dice": best_val_dice,
            }, best_model_path)
            tag = "<--- (best)"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
        }, os.path.join(run_dir, "checkpoints", f"{model_name}_last.pt"))

        print(f"Epoch {epoch} / {max_epochs} | "
              f"Train Mean Loss: {train_loss:.4f} | "
              f"Train Mean Dice: {mean_dice:.4f} | Train Mean IoU: {mean_iou:.4f} | "
              f"Train Batch Time: {sum(batch_times) / len(batch_times):.3f} s | "
              f"Val Mean Loss: {val_loss:.4f} | "
              f"Val Mean Dice: {val_mean_dice:.4f} | Val Mean IoU: {val_mean_iou:.4f} | "
              f"Val Batch Time: {sum(val_batch_times) / len(val_batch_times):.3f} s {tag}")

        train_time = sum(batch_times) / len(batch_times)
        val_time = sum(val_batch_times) / len(val_batch_times)
        current_lr = optimizer.param_groups[0]["lr"]

        log_tensorboard(
            writer, epoch,
            train_loss, mean_dice, mean_iou, train_time,
            val_loss, val_mean_dice, val_mean_iou, val_time,
            current_lr
        )

        log_csv(
            csv_path, epoch,
            train_loss, mean_dice, mean_iou, train_time,
            val_loss, val_mean_dice, val_mean_iou, val_time,
            current_lr
        )

    writer.close()
    return best_model_path or os.path.join(run_dir, "checkpoints", f"{model_name}_last.pt")


