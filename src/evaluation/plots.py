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

    epochs = range(1, len(history["train_loss"]) + 1)
    title_prefix = model_prefix.replace("_", " ").title()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        linewidth=2.2,
        color="#2563eb",
        label="Train Loss",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="o",
        linestyle="--",
        linewidth=2.2,
        color="#f97316",
        label="Val Loss",
    )
    axes[0].set_title(f"{title_prefix} Loss", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(list(epochs))
    axes[0].legend(frameon=True)

    axes[1].plot(
        epochs,
        history["train_acc"],
        marker="o",
        linewidth=2.2,
        color="#22c55e",
        label="Train Accuracy",
    )
    axes[1].plot(
        epochs,
        history["val_acc"],
        marker="o",
        linestyle="--",
        linewidth=2.2,
        color="#dc2626",
        label="Val Accuracy",
    )
    axes[1].set_title(f"{title_prefix} Accuracy", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(list(epochs))
    axes[1].legend(frameon=True)

    for axis in axes:
        axis.grid(True, linestyle="-", alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path


def save_confusion_matrix(
    confusion_matrix,
    class_names: List[str],
    output_dir: str,
    model_prefix: str,
    metrics: Dict[str, float] = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_path = output_path / f"{model_prefix}_confusion_matrix.png"

    total = confusion_matrix.sum()
    annotations = []
    for row in confusion_matrix:
        annotation_row = []
        for value in row:
            percentage = (value / total) * 100 if total else 0
            annotation_row.append(f"{value}\n{percentage:.2f}%")
        annotations.append(annotation_row)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        confusion_matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if metrics:
        metric_text = (
            f"Accuracy={metrics['accuracy']:.4f}\n"
            f"Precision(fake)={metrics['precision']:.4f}\n"
            f"Recall(fake)={metrics['recall']:.4f}\n"
            f"F1(fake)={metrics['f1_score']:.4f}"
        )
        plt.gcf().text(0.5, 0.01, metric_text, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close()

    return figure_path
