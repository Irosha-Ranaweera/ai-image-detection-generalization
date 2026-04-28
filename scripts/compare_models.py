import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import torch
from scipy.stats import binomtest
from sklearn.metrics import RocCurveDisplay

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


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    model_name = os.environ.get("MODEL_NAME", "resnet18")
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    num_workers = int(os.environ.get("NUM_WORKERS", 0))
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

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("Classes:", class_names)

    baseline_model = get_baseline_resnet(model_name=model_name, num_classes=2)
    baseline_model = load_state_dict(
        baseline_model, baseline_checkpoint, device
    ).to(device)

    eca_model = AttentionResNet(model_name=model_name, num_classes=2)
    eca_model = load_state_dict(eca_model, eca_checkpoint, device).to(device)

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

    save_confusion_matrix(
        baseline_results["confusion_matrix"],
        class_names,
        str(analysis_dir),
        "baseline_resnet18",
    )
    save_confusion_matrix(
        eca_results["confusion_matrix"],
        class_names,
        str(analysis_dir),
        "eca_resnet18",
    )

    report = {
        "summary": summary_rows,
        "mcnemar_test": mcnemar_results,
        "baseline_classification_report": baseline_results["classification_report"],
        "eca_classification_report": eca_results["classification_report"],
    }
    with (analysis_dir / "analysis_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nModel Comparison")
    for row in summary_rows:
        print(row)

    print("\nMcNemar Test")
    print(mcnemar_results)

    print("\nSaved analysis files to:", analysis_dir)


if __name__ == "__main__":
    main()
