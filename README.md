# AI Image Detection

This project implements a baseline ResNet18 model for binary classification of real vs AI-generated images.

## Project structure
- `src/data.py` - dataset loading and transforms
- `src/model.py` - model creation
- `src/train.py` - training and validation loops
- `src/evaluate.py` - test evaluation metrics
- `notebooks/resnet18_baseline_colab.ipynb` - Colab notebook runner

## Dataset structure
The dataset should be organized as:

data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/

In Colab, the dataset is loaded from Google Drive.