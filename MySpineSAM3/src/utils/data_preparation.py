"""
Data Preparation Utilities
==========================
DICOM to NIfTI conversion using nibabel and pydicom.
"""

import os
from pathlib import Path
from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


def dicom_to_nifti(dicom_dir: str, output_path: str, use_sitk: bool = True) -> str:
    """Convert DICOM series to NIfTI."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if use_sitk:
        try:
            import SimpleITK as sitk
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            sitk.WriteImage(image, str(output_path))
            logger.info(f"Converted {dicom_dir} -> {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"DICOM conversion failed: {e}")
            raise
    return str(output_path)


def prepare_dataset(raw_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Prepare dataset with train/val/test split."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Prepared directories in {output_dir}")
