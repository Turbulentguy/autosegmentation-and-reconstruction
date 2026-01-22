#!/usr/bin/env python
"""Evaluation script - compute metrics on test set."""

import argparse
import logging
from pathlib import Path
import json

import numpy as np
import nibabel as nib
import torch
import yaml

from src.model_loader import get_model
from src.metrics import MetricCalculator, InferenceTimer


def main():
    parser = argparse.ArgumentParser(description="Evaluate spine segmentation")
    parser.add_argument("--data", required=True, help="Test data directory")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--config", default="configs/spine_ct_config.yaml")
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model = get_model(config["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(args.device).eval()
    
    data_dir = Path(args.data)
    images = sorted((data_dir / "images").glob("*.nii*"))
    
    calc = MetricCalculator(config["evaluation"]["metrics"], num_classes=config["model"]["out_channels"])
    
    for img_path in images:
        lbl_path = data_dir / "labels" / img_path.name
        if not lbl_path.exists():
            continue
        
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int64)
        
        # Preprocess
        hu_min, hu_max = config["data"]["hu_min"], config["data"]["hu_max"]
        img = np.clip(img, hu_min, hu_max)
        img = (img - hu_min) / (hu_max - hu_min)
        
        with InferenceTimer() as timer:
            with torch.no_grad():
                x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(args.device)
                pred = model(x).argmax(dim=1).squeeze().cpu().numpy()
        
        calc.compute(pred, lbl, inference_time_ms=timer.elapsed_ms)
        logger.info(f"Processed {img_path.name}")
    
    results = {"mean": calc.get_average(), "std": calc.get_std()}
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results: {results['mean']}")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
