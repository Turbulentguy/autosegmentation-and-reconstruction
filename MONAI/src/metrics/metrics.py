from monai.metrics import (
    DiceMetric,
    MeanIoU
)

def get_dice():
    dice = DiceMetric(
        include_background = False,
        reduction = "none",
        ignore_empty = True
    )
    return dice

def get_iou():
    iou = MeanIoU(
        include_background = False,
        reduction = "none",
        ignore_empty = True
    )
    return iou