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
│   │   ├── models.py                   # Script for model preferance
│   │   ├── swinunetr.py                # Script for SwinUNetR model
│   │   ├── unet.py                     # Script for UNet model
│   │   └── unetr.py                    # Script for UNetR model
|   ├── preprocess/                     # Data Preprocessing Directory
│   │   └── preprocess.py               # Script for data preprocessing
|   ├── training/                       # Training Directory
│   │   ├── train.py                    # Script for model training
│   │   └── validation.py               # Script for model validation
|   └── utils/                          # utilities Directory
│       ├── configs.py                  # Script for configuration
│       ├── folder_handler.py           # Script for handling directories
│       └── logging.py                  # Script for logging
├── requirements.txt                    # Python dependencies
└── README.md                           # Main project documentation
```
