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

## Quick local testing in VS Code

If the full dataset is in Google Drive/My Drive on your laptop, create a small
copy for local development first. This keeps training fast while you check that
the code works.

Example:

```powershell
python scripts\create_sample_dataset.py --source "C:\path\to\full\dataset" --output data\sample --images-per-class 20
```

The source folder must contain:

```text
train/real
train/fake
val/real
val/fake
test/real
test/fake
```

Then train on the small sample:

```powershell
$env:DATA_DIR="data\sample"
$env:EPOCHS="2"
$env:BATCH_SIZE="8"
python scripts\train_baseline.py
```

After the local test works, use the full Google Drive dataset path in Colab for
the real training run.
