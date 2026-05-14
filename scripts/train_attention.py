import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.evaluation.evaluate import evaluate_model
from src.evaluation.plots import save_confusion_matrix, save_training_curves
from src.models.attention_resnet import AttentionResNet
from src.training.trainer import fit


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    model_name = os.environ.get("MODEL_NAME", "resnet18")
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = int(os.environ.get("EPOCHS", 10))
    learning_rate = float(os.environ.get("LEARNING_RATE", 1e-4))
    num_workers = int(os.environ.get("NUM_WORKERS", 0))
    early_stopping_patience = int(os.environ.get("EARLY_STOPPING_PATIENCE", 0))
    fine_tune_layer4 = os.environ.get("FINE_TUNE_LAYER4", "false").lower() == "true"
    transform_mode = os.environ.get("TRANSFORM_MODE", "rgb")
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/figures")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    save_path = os.environ.get(
        "SAVE_PATH", f"outputs/checkpoints/best_eca_{model_name}.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_mode=transform_mode,
    )

    print("Classes:", class_names)
    print("Transform mode:", transform_mode)

    model = AttentionResNet(model_name=model_name, num_classes=2)
    model = model.to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint:", checkpoint_path)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = list(model.eca.parameters()) + list(model.classifier.parameters())
    if fine_tune_layer4:
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        trainable_parameters += list(model.backbone.layer4.parameters())
        print("Fine-tuning layer4, ECA, and classifier head")
    else:
        print("Training ECA and classifier head only")

    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Trainable parameters:", trainable_count)

    optimizer = optim.AdamW(trainable_parameters, lr=learning_rate)

    model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path=save_path,
        early_stopping_patience=early_stopping_patience,
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
        f"{results['positive_class']} (label {results['positive_label']})",
    )
    print("Accuracy :", round(results["accuracy"], 4))
    print("Precision:", round(results["precision"], 4))
    print("Recall   :", round(results["recall"], 4))
    print("F1 Score :", round(results["f1_score"], 4))
    print("ROC-AUC  :", round(results["roc_auc"], 4))

    print("\nClassification Report:\n")
    print(results["classification_report"])

    print("\nConfusion Matrix:\n")
    print(results["confusion_matrix"])

    curves_path = save_training_curves(
        history=history,
        output_dir=output_dir,
        model_prefix=f"eca_{model_name}",
    )
    cm_path = save_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        class_names=class_names,
        output_dir=output_dir,
        model_prefix=f"eca_{model_name}",
        metrics=results,
    )

    print("\nSaved Figures")
    print("Training curves:", curves_path)
    print("Confusion matrix:", cm_path)


if __name__ == "__main__":
    main()
