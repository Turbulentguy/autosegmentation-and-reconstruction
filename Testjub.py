import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, EnsureTyped, SaveImaged
import os
from monai.inferers import sliding_window_inference


file_path = "/lustrefs/disk/home/sngamvil/test_ids.txt"
with open(file_path, "r") as f:
  test_ids = [line.strip() for line in f if line.strip()]
  
images_dir = "/lustrefs/disk/project/lt200431-ddmmss/wat/datasets--alexanderdann--CTSpine1K/snapshots/9b454add169b94f2c322ad6f08b66823975e8dbd/raw_data/volumes/COLONOG/"

num_classes = 19
model = UNet(spatial_dims = 3,
            in_channels = 1,
            out_channels = num_classes,
            channels = (16, 32, 64, 128, 256),
            strides = (2, 2, 2, 2),
            num_res_units = 2, 
).cuda()

model_path = "/lustrefs/disk/home/sngamvil/sam3/best_4gpu_model.pt"
state_dict = torch.load(model_path, map_location="cuda:0")
model.load_state_dict(state_dict)
model.eval()

transform = Compose([
    LoadImaged(keys = ["image"]),
    EnsureChannelFirstd(keys = ["image"]),
    Orientationd(keys = ["image"], axcodes = "RAS"),
    Spacingd(keys = ["image"],
        pixdim = (1.0, 1.0, 1.0),
        mode = "bilinear"
    ),
    ScaleIntensityRanged(keys = ["image"],
        a_min = -1000, a_max = 1000,
        b_min = 0.0, b_max = 1.0,
        clip = True
    ),
    SpatialPadd(keys = ["image"],
        spatial_size = None,
        divisible = (16, 16, 16)
    ),
    EnsureTyped(keys = ["image"],
                data_type = "tensor",
                track_meta = True
    ),
])

saver = SaveImaged(keys="pred",
    output_dir = "./predictions",
    output_postfix = "seg",
    output_ext = ".nii.gz",
    resample = False
)

for ids in test_ids:
    image_path = os.path.join(images_dir, ids + ".nii.gz")

    data = {"image": image_path}
    data = transform(data)
    image = data["image"].unsqueeze(0).cuda()

    with torch.no_grad():
        logits = sliding_window_inference(inputs = image,
                                          roi_size = (128, 128, 128),  
                                          sw_batch_size = 1,
                                          predictor = model,
                                          overlap = 0.5,
                                          mode = "gaussian"
        )
        pred = torch.argmax(logits, dim = 1)

    data["pred"] = pred
    saver(data)

    print(f"Saved segmentation for {ids}")
  
  