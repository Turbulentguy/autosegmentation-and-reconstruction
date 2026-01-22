# MySpineSAM3 Implementation Walkthrough

## Overview
Complete spine segmentation framework supporting 3D volumetric (MONAI) and 2D slice-based (SAM) training.

## Project Structure
```
MySpineSAM3/
├── train.py                    # Main entry point
├── configs/spine_ct_config.yaml
├── src/
│   ├── model_loader.py         # Factory: UNET, UNETR, SwinUNETR, MobileSAM
│   ├── dataset.py              # 3D NIfTI dataset
│   ├── ctspine1k_loader.py     # HuggingFace CTSpine1K
│   ├── slice_dataset.py        # 2D slices for SAM
│   ├── engine.py               # 3D training loop
│   ├── slice_engine.py         # 2D SAM training
│   ├── metrics.py              # DSC, IoU, HD95, ASD
│   ├── models/
│   │   ├── sam_adapter.py      # SAM with adapters
│   │   └── mobile_sam.py       # Mobile SAM
│   └── utils/
└── scripts/
```

## Training Modes

**3D Volumetric** (UNET, SwinUNETR):
```bash
python train.py --config configs/spine_ct_config.yaml --model SwinUNETR
```

**2D Slice-based** (SAM):
```yaml
model:
  architecture: "MobileSAM"
  training_mode: "2d"
```

## Available Models
| Model | Type | Use Case |
|-------|------|----------|
| UNET | 3D | Fast baseline |
| SwinUNETR | 3D | SOTA accuracy |
| MobileSAM | 2D | Efficient SAM |
| SegmentAnyBone | 2D | Bone-optimized |

## Quick Start
```bash
mamba activate verte
pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
python train.py --config configs/spine_ct_config.yaml
```
