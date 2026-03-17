import os
import csv
from torch.utils.tensorboard import SummaryWriter

def create_tensorboard(run_dir):
    tensor_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensor_dir)

    return writer

def create_csv(run_dir):
    csv_dir = os.path.join(run_dir, "csv")

    csv_path = os.path.join(csv_dir, "metrics.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", 
                            "Training Loss", 
                            "Training Dice",
                            "Training IoU",
                            "Training Time",
                            "Validation Loss", 
                            "Validation Dice", 
                            "Validation IoU",
                            "Validation Time",
                            "Learning Rate"])
    
    return csv_path

def log_tensorboard(writer, epoch, train_loss, train_dice, train_iou, train_time, val_loss, val_dice, val_iou, val_time, lr):

    writer.add_scalar("Train Loss", train_loss, epoch)
    writer.add_scalar("Train Dice", train_dice, epoch)
    writer.add_scalar("Train IoU", train_iou, epoch)
    writer.add_scalar("Training time", train_time, epoch)

    writer.add_scalar("Validation Loss", val_loss, epoch)
    writer.add_scalar("Validation Dice", val_dice, epoch)
    writer.add_scalar("Validation IoU", val_iou, epoch)
    writer.add_scalar("Validation time", val_time, epoch) 

    writer.add_scalar("Learning Rate", lr, epoch)

def log_csv(csv_path, epoch, train_loss, train_dice, train_iou, train_time, val_loss, val_dice, val_iou, val_time, lr):

    with open(csv_path, "a", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            train_dice,
            train_iou,
            train_time,
            val_loss,
            val_dice,
            val_iou,
            val_time,
            lr
        ])    
    