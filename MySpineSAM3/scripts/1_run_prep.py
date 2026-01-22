#!/usr/bin/env python
"""Data preparation script - DICOM to NIfTI conversion."""

import argparse
from src.utils.data_preparation import dicom_to_nifti, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare spine CT data")
    parser.add_argument("--input", required=True, help="Input DICOM directory")
    parser.add_argument("--output", required=True, help="Output NIfTI directory")
    parser.add_argument("--split", action="store_true", help="Create train/val/test splits")
    args = parser.parse_args()
    
    if args.split:
        prepare_dataset(args.input, args.output)
    else:
        print("Use --split to prepare dataset directories")


if __name__ == "__main__":
    main()
