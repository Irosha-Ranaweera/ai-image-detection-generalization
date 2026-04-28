from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def save_training_curves(
    history: Dict[str, list],
    output_dir: str,
    model_prefix: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_path = output_path / f"{model_prefix}_training_curves.png"

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path


def save_confusion_matrix(
    confusion_matrix,
    class_names: List[str],
    output_dir: str,
    model_prefix: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_path = output_path / f"{model_prefix}_confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path
