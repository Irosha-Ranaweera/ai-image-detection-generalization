import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import binomtest
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score

from src.data.dataset import get_dataloaders
from src.evaluation.evaluate import evaluate_model
from src.evaluation.plots import save_confusion_matrix
from src.models.attention_resnet import AttentionResNet
from src.models.baseline_resnet import get_baseline_resnet


def load_state_dict(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def enable_layer4_fine_tuning(model, is_attention_model: bool):
    layer4 = model.backbone.layer4 if is_attention_model else model.layer4
    for parameter in layer4.parameters():
        parameter.requires_grad = True


def mcnemar_exact_test(y_true, baseline_pred, eca_pred):
    baseline_correct = [truth == pred for truth, pred in zip(y_true, baseline_pred)]
    eca_correct = [truth == pred for truth, pred in zip(y_true, eca_pred)]

    baseline_correct_eca_wrong = sum(
        base and not eca for base, eca in zip(baseline_correct, eca_correct)
    )
    baseline_wrong_eca_correct = sum(
        not base and eca for base, eca in zip(baseline_correct, eca_correct)
    )

    disagreements = baseline_correct_eca_wrong + baseline_wrong_eca_correct
    p_value = 1.0
    if disagreements > 0:
        p_value = binomtest(
            baseline_correct_eca_wrong,
            n=disagreements,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "baseline_correct_eca_wrong": baseline_correct_eca_wrong,
        "baseline_wrong_eca_correct": baseline_wrong_eca_correct,
        "disagreements": disagreements,
        "p_value": p_value,
    }


def save_summary_csv(summary_rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


def dataset_size(dataset):
    return len(dataset)


def save_predictions_csv(
    output_path,
    test_loader,
    class_names,
    baseline_results,
    eca_results,
):
    samples = test_loader.dataset.samples

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_path",
                "true_label",
                "baseline_pred",
                "baseline_fake_score",
                "eca_pred",
                "eca_fake_score",
            ]
        )

        for index, (image_path, true_label_index) in enumerate(samples):
            writer.writerow(
                [
                    image_path,
                    class_names[true_label_index],
                    class_names[baseline_results["y_pred"][index]],
                    baseline_results["y_score"][index],
                    class_names[eca_results["y_pred"][index]],
                    eca_results["y_score"][index],
                ]
            )


def save_roc_curve(output_path, baseline_results, eca_results):
    baseline_true = [
        1 if label == baseline_results["positive_label"] else 0
        for label in baseline_results["y_true"]
    ]
    eca_true = [
        1 if label == eca_results["positive_label"] else 0
        for label in eca_results["y_true"]
    ]

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        baseline_true,
        baseline_results["y_score"],
        name="ResNet18 baseline",
        ax=plt.gca(),
    )
    RocCurveDisplay.from_predictions(
        eca_true,
        eca_results["y_score"],
        name="ResNet18 + ECA",
        ax=plt.gca(),
    )
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_precision_recall_curve(output_path, baseline_results, eca_results):
    baseline_true = [
        1 if label == baseline_results["positive_label"] else 0
        for label in baseline_results["y_true"]
    ]
    eca_true = [
        1 if label == eca_results["positive_label"] else 0
        for label in eca_results["y_true"]
    ]

    baseline_ap = average_precision_score(baseline_true, baseline_results["y_score"])
    eca_ap = average_precision_score(eca_true, eca_results["y_score"])

    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        baseline_true,
        baseline_results["y_score"],
        name=f"ResNet18 baseline (AP={baseline_ap:.3f})",
        ax=plt.gca(),
    )
    PrecisionRecallDisplay.from_predictions(
        eca_true,
        eca_results["y_score"],
        name=f"ResNet18 + ECA (AP={eca_ap:.3f})",
        ax=plt.gca(),
    )
    plt.title("Precision-Recall Curve for Fake Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_probability_density(output_path, baseline_results, eca_results):
    baseline_true = [
        "fake" if label == baseline_results["positive_label"] else "real"
        for label in baseline_results["y_true"]
    ]
    eca_true = [
        "fake" if label == eca_results["positive_label"] else "real"
        for label in eca_results["y_true"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    plot_specs = [
        ("ResNet18 baseline", baseline_true, baseline_results["y_score"]),
        ("ResNet18 + ECA", eca_true, eca_results["y_score"]),
    ]

    for axis, (title, labels, scores) in zip(axes, plot_specs):
        sns.kdeplot(x=scores, hue=labels, fill=True, common_norm=False, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Predicted fake probability")
        axis.set_ylabel("Density")
        axis.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    model_name = os.environ.get("MODEL_NAME", "resnet18")
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    num_workers = int(os.environ.get("NUM_WORKERS", 0))
    epochs = os.environ.get("EPOCHS", "not_recorded")
    learning_rate = os.environ.get("LEARNING_RATE", "not_recorded")
    transform_mode = os.environ.get("TRANSFORM_MODE", "rgb")
    fine_tune_layer4 = os.environ.get("FINE_TUNE_LAYER4", "false").lower() == "true"
    baseline_checkpoint = os.environ.get(
        "BASELINE_CHECKPOINT", f"outputs/checkpoints/best_{model_name}.pth"
    )
    eca_checkpoint = os.environ.get(
        "ECA_CHECKPOINT", f"outputs/checkpoints/best_eca_{model_name}.pth"
    )
    analysis_dir = Path(os.environ.get("ANALYSIS_DIR", "outputs/analysis"))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_mode=transform_mode,
    )
    print("Classes:", class_names)

    metadata = {
        "data_dir": data_dir,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "transform_mode": transform_mode,
        "fine_tune_layer4": fine_tune_layer4,
        "num_workers": num_workers,
        "device": str(device),
        "class_names": class_names,
        "positive_class": "fake",
        "dataset_sizes": {
            "train": dataset_size(train_loader.dataset),
            "val": dataset_size(val_loader.dataset),
            "test": dataset_size(test_loader.dataset),
            "total": (
                dataset_size(train_loader.dataset)
                + dataset_size(val_loader.dataset)
                + dataset_size(test_loader.dataset)
            ),
        },
        "checkpoints": {
            "baseline": baseline_checkpoint,
            "eca": eca_checkpoint,
        },
    }

    baseline_model = get_baseline_resnet(model_name=model_name, num_classes=2)
    baseline_model = load_state_dict(
        baseline_model, baseline_checkpoint, device
    ).to(device)

    eca_model = AttentionResNet(model_name=model_name, num_classes=2)
    eca_model = load_state_dict(eca_model, eca_checkpoint, device).to(device)

    if fine_tune_layer4:
        enable_layer4_fine_tuning(baseline_model, is_attention_model=False)
        enable_layer4_fine_tuning(eca_model, is_attention_model=True)

    baseline_total, baseline_trainable = count_parameters(baseline_model)
    eca_total, eca_trainable = count_parameters(eca_model)

    baseline_results = evaluate_model(
        baseline_model,
        test_loader,
        device,
        class_names=class_names,
        positive_class="fake",
    )
    eca_results = evaluate_model(
        eca_model,
        test_loader,
        device,
        class_names=class_names,
        positive_class="fake",
    )

    mcnemar_results = mcnemar_exact_test(
        baseline_results["y_true"],
        baseline_results["y_pred"],
        eca_results["y_pred"],
    )

    summary_rows = [
        {
            "model": "ResNet18 baseline",
            "dataset_total": metadata["dataset_sizes"]["total"],
            "train_size": metadata["dataset_sizes"]["train"],
            "val_size": metadata["dataset_sizes"]["val"],
            "test_size": metadata["dataset_sizes"]["test"],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "transform_mode": transform_mode,
            "fine_tune_layer4": fine_tune_layer4,
            "accuracy": baseline_results["accuracy"],
            "precision_fake": baseline_results["precision"],
            "recall_fake": baseline_results["recall"],
            "f1_fake": baseline_results["f1_score"],
            "roc_auc": baseline_results["roc_auc"],
            "total_parameters": baseline_total,
            "trainable_parameters": baseline_trainable,
            "checkpoint": baseline_checkpoint,
        },
        {
            "model": "ResNet18 + ECA",
            "dataset_total": metadata["dataset_sizes"]["total"],
            "train_size": metadata["dataset_sizes"]["train"],
            "val_size": metadata["dataset_sizes"]["val"],
            "test_size": metadata["dataset_sizes"]["test"],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "transform_mode": transform_mode,
            "fine_tune_layer4": fine_tune_layer4,
            "accuracy": eca_results["accuracy"],
            "precision_fake": eca_results["precision"],
            "recall_fake": eca_results["recall"],
            "f1_fake": eca_results["f1_score"],
            "roc_auc": eca_results["roc_auc"],
            "total_parameters": eca_total,
            "trainable_parameters": eca_trainable,
            "checkpoint": eca_checkpoint,
        },
    ]

    save_summary_csv(summary_rows, analysis_dir / "model_comparison_summary.csv")
    save_predictions_csv(
        analysis_dir / "test_predictions.csv",
        test_loader,
        class_names,
        baseline_results,
        eca_results,
    )
    save_roc_curve(analysis_dir / "roc_curve.png", baseline_results, eca_results)
    save_precision_recall_curve(
        analysis_dir / "precision_recall_curve.png",
        baseline_results,
        eca_results,
    )
    save_probability_density(
        analysis_dir / "probability_density.png",
        baseline_results,
        eca_results,
    )

    save_confusion_matrix(
        baseline_results["confusion_matrix"],
        class_names,
        str(analysis_dir),
        "baseline_resnet18",
        metrics=baseline_results,
    )
    save_confusion_matrix(
        eca_results["confusion_matrix"],
        class_names,
        str(analysis_dir),
        "eca_resnet18",
        metrics=eca_results,
    )

    report = {
        "metadata": metadata,
        "summary": summary_rows,
        "mcnemar_test": mcnemar_results,
        "baseline_classification_report": baseline_results["classification_report"],
        "eca_classification_report": eca_results["classification_report"],
    }
    with (analysis_dir / "analysis_report.json").open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(report), file, indent=2)

    print("\nModel Comparison")
    for row in summary_rows:
        print(row)

    print("\nMcNemar Test")
    print(mcnemar_results)

    print("\nSaved analysis files to:", analysis_dir)


if __name__ == "__main__":
    main()
