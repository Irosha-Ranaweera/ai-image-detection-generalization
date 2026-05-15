import csv
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.evaluation.evaluate import evaluate_model
from src.evaluation.plots import save_confusion_matrix, save_training_curves
from src.models.attention_resnet import AttentionResNet
from src.models.baseline_resnet import get_baseline_resnet
from src.training.trainer import fit


def parse_list(name: str, default: str) -> List[str]:
    value = os.environ.get(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def build_model(model_type: str, model_name: str, fine_tune_layer4: bool, device):
    if model_type == "baseline":
        model = get_baseline_resnet(model_name=model_name, num_classes=2)
        trainable_parameters = list(model.fc.parameters())

        if fine_tune_layer4:
            for param in model.layer4.parameters():
                param.requires_grad = True
            trainable_parameters += list(model.layer4.parameters())

    elif model_type == "eca":
        model = AttentionResNet(model_name=model_name, num_classes=2)
        trainable_parameters = list(model.eca.parameters()) + list(
            model.classifier.parameters()
        )

        if fine_tune_layer4:
            for param in model.backbone.layer4.parameters():
                param.requires_grad = True
            trainable_parameters += list(model.backbone.layer4.parameters())
    else:
        raise ValueError("GRID_MODELS must contain only 'baseline' and/or 'eca'")

    model = model.to(device)
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_count = sum(param.numel() for param in model.parameters())
    return model, trainable_parameters, total_count, trainable_count


def make_grid() -> Iterable[Dict[str, object]]:
    models = parse_list("GRID_MODELS", "baseline,eca")
    model_names = parse_list("GRID_MODEL_NAMES", os.environ.get("MODEL_NAME", "resnet18"))
    learning_rates = [float(item) for item in parse_list("GRID_LEARNING_RATES", "1e-4,5e-5")]
    batch_sizes = [int(item) for item in parse_list("GRID_BATCH_SIZES", "32")]
    epochs_values = [int(item) for item in parse_list("GRID_EPOCHS", "5,10")]
    transform_modes = parse_list("GRID_TRANSFORM_MODES", "rgb")
    fine_tune_values = [
        parse_bool(item) for item in parse_list("GRID_FINE_TUNE_LAYER4", "false,true")
    ]

    keys = [
        "model_type",
        "model_name",
        "learning_rate",
        "batch_size",
        "epochs",
        "transform_mode",
        "fine_tune_layer4",
    ]
    values = [
        models,
        model_names,
        learning_rates,
        batch_sizes,
        epochs_values,
        transform_modes,
        fine_tune_values,
    ]

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    output_dir = Path(os.environ.get("GRID_OUTPUT_DIR", "outputs/grid_search"))
    num_workers = int(os.environ.get("NUM_WORKERS", 2))
    seed = int(os.environ.get("SEED", 42))
    early_stopping_patience = int(os.environ.get("EARLY_STOPPING_PATIENCE", 0))
    evaluate_test_each_trial = parse_bool(
        os.environ.get("EVALUATE_TEST_EACH_TRIAL", "false")
    )
    save_figures = parse_bool(os.environ.get("GRID_SAVE_FIGURES", "true"))
    max_trials = int(os.environ.get("MAX_TRIALS", 0))

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Grid output directory:", output_dir)

    grid = list(make_grid())
    if max_trials > 0:
        grid = grid[:max_trials]

    print(f"Total trials: {len(grid)}")

    dataloader_cache = {}
    rows: List[Dict[str, object]] = []
    best_trial = None

    for trial_index, params in enumerate(grid, start=1):
        trial_name = (
            f"trial_{trial_index:03d}_"
            f"{params['model_type']}_{params['model_name']}_"
            f"{params['transform_mode']}_"
            f"lr{params['learning_rate']}_bs{params['batch_size']}_"
            f"ep{params['epochs']}_layer4{params['fine_tune_layer4']}"
        ).replace(".", "p")

        print("\n" + "=" * 80)
        print(f"Trial {trial_index}/{len(grid)}: {trial_name}")
        print(params)

        cache_key = (params["batch_size"], params["transform_mode"])
        if cache_key not in dataloader_cache:
            dataloader_cache[cache_key] = get_dataloaders(
                data_dir=data_dir,
                batch_size=params["batch_size"],
                num_workers=num_workers,
                transform_mode=params["transform_mode"],
            )

        train_loader, val_loader, test_loader, class_names = dataloader_cache[cache_key]
        print("Classes:", class_names)

        model, trainable_parameters, total_params, trainable_params = build_model(
            model_type=params["model_type"],
            model_name=params["model_name"],
            fine_tune_layer4=params["fine_tune_layer4"],
            device=device,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(trainable_parameters, lr=params["learning_rate"])
        checkpoint_path = output_dir / "checkpoints" / f"{trial_name}.pth"

        model, history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=params["epochs"],
            save_path=str(checkpoint_path),
            early_stopping_patience=early_stopping_patience,
        )

        best_val_acc = max(history["val_acc"])
        best_val_epoch = history["val_acc"].index(best_val_acc) + 1
        final_val_loss = history["val_loss"][-1]
        final_train_acc = history["train_acc"][-1]
        final_train_loss = history["train_loss"][-1]

        row = {
            "trial": trial_index,
            "trial_name": trial_name,
            "model": params["model_type"],
            "model_name": params["model_name"],
            "transform_mode": params["transform_mode"],
            "fine_tune_layer4": params["fine_tune_layer4"],
            "epochs_requested": params["epochs"],
            "epochs_ran": len(history["val_acc"]),
            "best_val_epoch": best_val_epoch,
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "best_val_acc": best_val_acc,
            "checkpoint": str(checkpoint_path),
        }

        if evaluate_test_each_trial:
            test_results = evaluate_model(
                model,
                test_loader,
                device,
                class_names=class_names,
                positive_class="fake",
            )
            row.update(
                {
                    "test_accuracy": test_results["accuracy"],
                    "test_precision_fake": test_results["precision"],
                    "test_recall_fake": test_results["recall"],
                    "test_f1_fake": test_results["f1_score"],
                    "test_roc_auc": test_results["roc_auc"],
                }
            )

            if save_figures:
                figure_dir = output_dir / "figures" / trial_name
                save_confusion_matrix(
                    confusion_matrix=test_results["confusion_matrix"],
                    class_names=class_names,
                    output_dir=str(figure_dir),
                    model_prefix=trial_name,
                    metrics=test_results,
                )

        if save_figures:
            save_training_curves(
                history=history,
                output_dir=str(output_dir / "figures" / trial_name),
                model_prefix=trial_name,
            )

        rows.append(row)
        if best_trial is None or row["best_val_acc"] > best_trial["best_val_acc"]:
            best_trial = row

        write_csv(output_dir / "grid_search_results.csv", rows)
        with (output_dir / "grid_search_results.json").open("w", encoding="utf-8") as file:
            json.dump(to_jsonable(rows), file, indent=2)

    rows = sorted(rows, key=lambda item: item["best_val_acc"], reverse=True)
    write_csv(output_dir / "grid_search_results_ranked.csv", rows)

    print("\nBest validation trial")
    print(best_trial)

    # Formal final evaluation: evaluate only the best validation checkpoint on test set.
    best_params = next(row for row in rows if row["trial_name"] == best_trial["trial_name"])
    model, _, _, _ = build_model(
        model_type=best_params["model"],
        model_name=best_params["model_name"],
        fine_tune_layer4=parse_bool(str(best_params["fine_tune_layer4"])),
        device=device,
    )
    model.load_state_dict(torch.load(best_params["checkpoint"], map_location=device))

    cache_key = (int(best_params["batch_size"]), best_params["transform_mode"])
    _, _, test_loader, class_names = dataloader_cache[cache_key]

    final_results = evaluate_model(
        model,
        test_loader,
        device,
        class_names=class_names,
        positive_class="fake",
    )

    final_summary = {
        **best_params,
        "test_accuracy": final_results["accuracy"],
        "test_precision_fake": final_results["precision"],
        "test_recall_fake": final_results["recall"],
        "test_f1_fake": final_results["f1_score"],
        "test_roc_auc": final_results["roc_auc"],
        "confusion_matrix": final_results["confusion_matrix"],
        "classification_report": final_results["classification_report"],
    }

    with (output_dir / "best_model_test_results.json").open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(final_summary), file, indent=2)

    save_confusion_matrix(
        confusion_matrix=final_results["confusion_matrix"],
        class_names=class_names,
        output_dir=str(output_dir / "best_model_figures"),
        model_prefix="best_grid_model",
        metrics=final_results,
    )

    print("\nBest model final test results")
    print("Accuracy:", round(final_results["accuracy"], 4))
    print("Precision(fake):", round(final_results["precision"], 4))
    print("Recall(fake):", round(final_results["recall"], 4))
    print("F1(fake):", round(final_results["f1_score"], 4))
    print("ROC-AUC:", round(final_results["roc_auc"], 4))
    print("\nSaved grid-search outputs to:", output_dir)


if __name__ == "__main__":
    main()
