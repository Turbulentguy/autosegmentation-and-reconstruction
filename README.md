# Automated Three-Dimensional (3D) Lumbar Spine Segmentation from Computed Tomography

---

## 1. About

This repository presents a deep learning pipeline for automated three-dimensional (3D) lumbar spine segmentation from computed tomography (CT) images.

This project aims to develop a computer vision-based model for automatic vertebral bone segmentation from CT images. The proposed approach seeks to enhance interpretability, improve diagnostic efficiency, and support more accurate clinical assessment.

Back-related disorders are increasingly prevalent and pose challenges in clinical diagnosis. Although CT imaging is widely used in clinical practice, its two-dimensional (2D) representation can limit the interpretation of complex anatomical structures. Therefore, automated three-dimensional segmentation of the lumbar spine can provide clearer anatomical visualization and assist clinicians in diagnosis and treatment planning.


## 2. Project Overview

This repository focuses on developing a computer vision-based approach for automatic vertebral bone segmentation from CT images.

This repository builds upon existing baseline segmentation models and applies them to CT volumes to develop a system capable of automated three-dimensional (3D) lumbar spine segmentation.

The output of this project is a trained model that can perform automatic segmentation of lumbar vertebrae from CT images during inference, without requiring manual masks or labels.

### Pipeline Overview Diagram
```
CT Images (3D Volume)
      ↓
Data Preprocessing
      ↓
Deep Learning Segmentation Model
      ↓
Vertebra Segmentation Mask
      ↓
3D Visualization / Reconstruction
```

## 3. Features
- Three-dimensional (3D) CT image segmentation
- Support for multiple model architectures (UNet, UNETR, SwinUNETR)
- TensorBoard logging for training monitoring
- Modular training and validation framework
- Automatic model checkpoint saving
- Configurable training via YAML configuration files

## 4. Repository Structures
```
repository_name/
├── configs/                            # Configuration Directory
│   └── configs.yaml                    # Model & Path Configurations
├── scripts/                            # Scripts Directory
│   └── main.py                         # Script to run the repository
├── src/                                # Sources Directory
│   ├── data/                           # Data Directory
│   │   ├── data_loader.py              # Script to load data
│   │   ├── data_pairs.py               # Script to pair data
│   │   └── data_splitter.py            # Script to split data (Training set, Validation set, Testing set)
|   ├── loss/                           # Loss Directory
│   │   └── loss.py                     # Script for DiceCeLoss
|   ├── metrics/                        # Metrics Directory
│   │   └── metrics.py                  # Script for Dice & IoU
|   ├── models/                         # Models Directory
│   │   ├── models.py                   # Script for Model Preferance
│   │   ├── swinunetr.py                # Script for SwinUNetR model
│   │   ├── unet.py                     # Script for UNet model
│   │   └── unetr.py                    # Script for UNetR model
|   ├── postprocess/                    # Data Postprocessing Directory
│   │   └── postprocess.py              # Script for Data Postprocessing
|   ├── preprocess/                     # Data Preprocessing Directory
│   │   └── preprocess.py               # Script for Data Preprocessing
|   ├── test/
│   │   ├── saver.py                    # Script to save files to a preferred Directory
│   │   └── test.py                     # Script for Model Testing
|   ├── training/                       # Training Directory
│   │   ├── train.py                    # Script for Model Training
│   │   └── validation.py               # Script for Model Validation
|   └── utils/                          # Utilities Directory
│       ├── configs.py                  # Script for configuration
│       ├── folder_handler.py           # Script for handling directories
│       └── logging.py                  # Script for logging
├── requirements.txt                    # Python dependencies
└── README.md                           # Main project documentation
```

## 5. Installation
### Clone repository
```
git clone https://github.com/Turbulentguy/autosegmentation-and-reconstruction.git
```
### Move into project directory
```
cd repository_name
```
### Install dependencies
```
pip install -r requirements.txt
```

## 6. Training
### How to perform model training
```
python scripts/main.py --config configs/configs.yaml --model <model_name>
```

## 7. Monitoring
### How to enable TensorBoard
```
python -m tensorboard.main --logdir outputs
```

## 8. Dataset

This repository uses the **CTSpine1K** dataset for training and evaluating the segmentation models. The dataset provides metadata that specifies which files should be used for model training, validation, and testing.

Dataset split:

- Training: 610 CT volumes
- Validation: 197 CT volumes
- Testing: 198 CT volumes

## 9. Results

| Model | Loss  | DICE  | IoU   | Validation Batch Time |
|-------|-------|-------|-------|------------------------|
| UNet | 0.7175 | 0.7750 | 0.7069 | 0.027 seconds |
| UNetR | 0.7493 | 0.7481 | 0.6735 | 0.102 seconds |
| SwinUNetR | 0.7292 | 0.7571 | 0.6923 | 0.152 seconds |

## 10. Visualization

Segmentation outputs can be imported into 3D Slicer for surface reconstruction and visualization.

## 11. Testing / Inference from CLI

### Run test only (using an existing trained checkpoint)
```
python scripts/main.py --config configs/configs.yaml --model <model_name> --mode test --best_model_path <path_to_best_model.pt>
```

Example:
```
python scripts/main.py --config configs/configs.yaml --model unet --mode test --best_model_path outputs/unet/run_001/checkpoints/unet_best_model.pt
```

### Run training and testing in one command
```
python scripts/main.py --config configs/configs.yaml --model <model_name> --mode both
```

### Supported model names
- `unet`
- `unetr`
- `swinunetr`

### Important notes
- When using `--mode test`, you must provide `--best_model_path`.
- Testing uses sliding-window inference and post-processing before saving masks.
- Output predictions are saved under the run directory in:
      - `<run_dir>/predictions/`

### Optional path overrides from CLI
You can override dataset and output paths without editing the YAML file:
```
python scripts/main.py \
      --config configs/configs.yaml \
      --model unet \
      --mode test \
      --best_model_path <path_to_best_model.pt> \
      --meta_path <path_to_data_split.txt> \
      --images_dir <path_to_images> \
      --masks_dir <path_to_masks> \
      --outputs_path <path_to_outputs>
```
