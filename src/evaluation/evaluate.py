from typing import Dict, List, Optional

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model,
    loader,
    device,
    class_names: Optional[List[str]] = None,
    positive_class: str = "fake",
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    positive_label = 1
    if class_names is not None:
        positive_label = class_names.index(positive_class)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "positive_class": positive_class,
        "positive_label": positive_label,
        "precision": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }
