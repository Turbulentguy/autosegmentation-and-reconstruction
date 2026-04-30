from monai.losses import DiceCELoss

def get_loss():
    loss = DiceCELoss(
        include_background = False,
        to_onehot_y = True,
        softmax = True,
    )

    return loss