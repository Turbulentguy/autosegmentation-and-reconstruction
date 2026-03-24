# Automated Three-Dimensional (3D) Lumbar Spine Segmentation from Computed Tomography

---

## 1. About

This repository presents a deep learning pipeline for automated three-dimensional (3D) lumbar spine segmentation from computed tomography (CT) images.

This project aims to develop a computer vision-based model for automatic vertebral bone segmentation from CT images. The proposed approach seeks to enhance interpretability, improve diagnostic efficiency, and support more accurate clinical assessment.

Back-related disorders are increasingly prevalent and pose challenges in clinical diagnosis. Although CT imaging is widely used in clinical practice, its two-dimensional (2D) representation can limit the interpretation of complex anatomical structures. Therefore, automated three-dimensional segmentation of the lumbar spine can provide clearer anatomical visualization and assist clinicians in diagnosis and treatment planning.

## 2. Project Highlight & Features
- 3D lumbar spine segmentation from CT volumes
- Supports UNet, UNETR, and SwinUNETR architectures
- Trained on CTSpine1K dataset (1000+ volumes)
- Achieves Dice score up to 0.775
- Modular pipeline with configurable YAML setup
- TensorBoard integration for training monitoring
- Resume training from latest checkpoint
- Automatic best-model checkpointing based on validation Dice
- Sliding-window inference for large 3D volumes

## 3. Project Overview

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

## 4. Dataset

This repository uses the **CTSpine1K** dataset for training and evaluating the segmentation models. The dataset provides metadata that specifies which files should be used for model training, validation, and testing.

Dataset split:

- Training: 610 CT volumes
- Validation: 197 CT volumes
- Testing: 198 CT volumes

## 5. Results

| Model | Loss  | DICE  | IoU   | Validation Batch Time |
|-------|-------|-------|-------|------------------------|
| UNet | 0.7175 | 0.7750 | 0.7069 | 0.027 seconds |
| UNetR | 0.7493 | 0.7481 | 0.6735 | 0.102 seconds |
| SwinUNetR | 0.7292 | 0.7571 | 0.6923 | 0.152 seconds |

## 6. Installation
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

## 7. Usages (CLI)
### Run full pipeline (train + test):
```
python -m scripts.main --config configs/configs.yaml --mode both --model unet
```
### CLI Help
```
python -m scripts.main --help
```
### Train
```
python -m scripts.main --config configs/configs.yaml --mode train --model <model_name>
```
### Example:
```
python -m scripts.main --config configs/configs.yaml --mode train --model unet
```
### Test
```
python -m scripts.main --config configs/configs.yaml --mode test --model <model_name> --best_model_path <path_to_best_model.pt>
```
### Example:
```
python -m scripts.main --config configs/configs.yaml --mode test --model unet --best_model_path outputs/unet/run_001/checkpoints/unet_best_model.pt
```
### Train + Test
```
python -m scripts.main --config configs/configs.yaml --mode both --model <model_name>
```
### Example:
```
python -m scripts.main --config configs/configs.yaml --mode both --model unet
```
### Optional path overrides from CLI
You can override dataset and output paths without editing the YAML file:
```
python -m scripts.main \
      --config configs/configs.yaml \
      --mode <preferred mode (train, test, both)> \
      --model <unet, unetr, swinunetr> \
      --meta_path <path_to_data_split.txt> \
      --images_dir <path_to_images> \
      --masks_dir <path_to_masks> \
      --resume_training_path <path_to_resume_training (path to last model)> \
      --outputs_path <path_to_outputs> \
      --best_model_path <path_to_best_model.pt> \
      --random_seed <random_seed> \
      --batch_size <batch_size> \
      --num_workers <num_workers> \
      --num_epochs <num_epochs> \
      --learning_rate <learning_rate> \
      --weight_decay <weight_decay>
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

## 8. Model Monitoring
### How to enable TensorBoard
```
python -m tensorboard.main --logdir outputs
```

## 9. Repository Structures
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
│   │   ├── models.py                   # Script for Model Preference
│   │   ├── swinunetr.py                # Script for SwinUNetR model
│   │   ├── unet.py                     # Script for UNet model
│   │   └── unetr.py                    # Script for UNetR model
|   ├── postprocess/                    # Data Postprocessing Directory
│   │   └── postprocess.py              # Script for Data Postprocessing
|   ├── preprocess/                     # Data Preprocessing Directory
│   │   └── preprocess.py               # Script for Data Preprocessing
|   ├── test/                           # Testing Directory
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

## 10. Visualization

Segmentation outputs can be imported into 3D Slicer for surface reconstruction and visualization.

## 11. Training Resources

All three models (UNet, UNETR, and SwinUNETR) were trained on the **LANTA supercomputer**.

Resource setting used for training jobs:

- 1 GPU (`--gres=gpu:1`)
- 8 CPU cores (`--cpus-per-task=8`)
- 64 GB RAM (`--mem=64G`)

Using this setup, the training time for the three models was approximately **100 hours** per model.
