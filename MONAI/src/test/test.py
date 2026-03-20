import torch
from tqdm import tqdm
from src.utils.configs import set_runtime_config
from src.models.models import build_model
from src.data.data_pairs import data_pairs
from src.data.data_loader import get_test_loader
from src.test.saver import get_saver
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from src.postprocess.postprocess import build_post_process

def test(config_path, model_name = None, best_model_path = None):

    config = set_runtime_config(config_path = config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_name or config["models"]["name"]
    model_path = best_model_path or config["outputs"]["best_model_path"]

    model = build_model(model_name).to(device)
    
    checkpoint = torch.load(model_path, map_location = device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    _, _, test_pairs = data_pairs(config)
    test_loader = get_test_loader(test_pairs, config)
    
    num_classes = config["data"]["num_classes_lumbar"]
    post_pred = AsDiscrete(argmax = True, to_onehot = num_classes)
    post_label = AsDiscrete(argmax = True)
    roi_size = tuple(config["models"]["global"].get("img_size", [96, 96, 96]))
    sw_batch_size = 1
    overlap = 0.25

    saver = get_saver(config_path, model_path)
    post_process = build_post_process(config)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc = "Testing"):

            images = batch["image"].to(device)
            outputs = sliding_window_inference(
                images,
                roi_size = roi_size,
                sw_batch_size = sw_batch_size,
                predictor = model,
                overlap = overlap,
            )

            outputs_list = decollate_batch(outputs)
            batch_list = decollate_batch(batch)

            for output_tensor, output_dict in zip(outputs_list, batch_list):

                pred = post_pred(output_tensor)

                output_dict["pred"] = pred
                output_dict = post_process(output_dict)
                output_dict["pred"] = post_label(output_dict["pred"])

                image_meta = output_dict.get("image_meta_dict")
                if image_meta is None and "image" in output_dict and hasattr(output_dict["image"], "meta"):
                    image_meta = dict(output_dict["image"].meta)
                if image_meta is None:
                    raise KeyError("Missing image metadata for prediction saving.")

                output_dict["pred_meta_dict"] = image_meta

                saver(output_dict)

    print("Done testing & saving predictions.")




