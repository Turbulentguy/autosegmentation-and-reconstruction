#!/usr/bin/env python
"""Inference script for spine segmentation."""

import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import yaml

from src.model_loader import get_model


def main():
    parser = argparse.ArgumentParser(description="Run inference on spine CT")
    parser.add_argument("--input", required=True, help="Input NIfTI file")
    parser.add_argument("--output", required=True, help="Output mask file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--config", default="configs/spine_ct_config.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model = get_model(config["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(args.device).eval()
    
    # Load and preprocess
    nii = nib.load(args.input)
    data = nii.get_fdata().astype(np.float32)
    hu_min, hu_max = config["data"]["hu_min"], config["data"]["hu_max"]
    data = np.clip(data, hu_min, hu_max)
    data = (data - hu_min) / (hu_max - hu_min)
    
    # Predict (simplified - full implementation would handle patching)
    with torch.no_grad():
        x = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(args.device)
        pred = model(x).argmax(dim=1).squeeze().cpu().numpy()
    
    # Save
    out_nii = nib.Nifti1Image(pred.astype(np.int16), nii.affine, nii.header)
    nib.save(out_nii, args.output)
    logging.info(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
