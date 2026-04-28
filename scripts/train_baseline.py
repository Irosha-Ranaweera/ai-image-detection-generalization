import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.evaluation.plots import save_confusion_matrix, save_training_curves
from src.models.baseline_resnet import get_baseline_resnet
from src.training.trainer import fit
from src.evaluation.evaluate import evaluate_model


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    model_name = os.environ.get("MODEL_NAME", "resnet18")
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = int(os.environ.get("EPOCHS", 10))
    learning_rate = float(os.environ.get("LEARNING_RATE", 1e-4))
    num_workers = int(os.environ.get("NUM_WORKERS", 0))
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/figures")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    save_path = os.environ.get("SAVE_PATH", f"outputs/checkpoints/best_{model_name}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print("Classes:", class_names)

    model = get_baseline_resnet(model_name=model_name, num_classes=2)
    model = model.to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint:", checkpoint_path)

    criterion = nn.CrossEntropyLoss()

    # Only train classification head at first
    if model_name in ["resnet18", "resnet50"]:
        optimizer = optim.AdamW(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path=save_path
    )

    results = evaluate_model(
        model,
        test_loader,
        device,
        class_names=class_names,
        positive_class="fake",
    )

    print("\nTest Results")
    print(
        "Positive class:",
        f"{results['positive_class']} (label {results['positive_label']})"
    )
    print("Accuracy :", round(results["accuracy"], 4))
    print("Precision:", round(results["precision"], 4))
    print("Recall   :", round(results["recall"], 4))
    print("F1 Score :", round(results["f1_score"], 4))

    print("\nClassification Report:\n")
    print(results["classification_report"])

    print("\nConfusion Matrix:\n")
    print(results["confusion_matrix"])

    curves_path = save_training_curves(
        history=history,
        output_dir=output_dir,
        model_prefix=f"baseline_{model_name}",
    )
    cm_path = save_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        class_names=class_names,
        output_dir=output_dir,
        model_prefix=f"baseline_{model_name}",
    )

    print("\nSaved Figures")
    print("Training curves:", curves_path)
    print("Confusion matrix:", cm_path)


if __name__ == "__main__":
    main()
