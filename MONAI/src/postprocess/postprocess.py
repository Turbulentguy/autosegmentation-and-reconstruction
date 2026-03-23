from monai.transforms import (
    Compose, KeepLargestConnectedComponentd, RemoveSmallObjectsd, FillHolesd
)

def build_post_process(config):
    num_classes = config["data"]["num_classes_lumbar"]
    foreground = list(range(1, num_classes))
    post_process = Compose([
        KeepLargestConnectedComponentd(
            keys = "pred",
            applied_labels = foreground,
            is_onehot = True,
            independent = True
        ),
        RemoveSmallObjectsd(
            keys = "pred",
            min_size = 5000,
            independent_channels = True
        ),
        FillHolesd(
            keys = "pred",
            applied_labels = foreground,
        )
    ])
    return post_process

