import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def make_case_grid(rows, title, output_path, max_images):
    selected_rows = rows[:max_images]
    if not selected_rows:
        print(f"No examples found for: {title}")
        return

    columns = min(4, len(selected_rows))
    grid_rows = (len(selected_rows) + columns - 1) // columns
    fig, axes = plt.subplots(grid_rows, columns, figsize=(columns * 3.2, grid_rows * 3.6))

    if grid_rows == 1 and columns == 1:
        axes = [[axes]]
    elif grid_rows == 1:
        axes = [axes]
    elif columns == 1:
        axes = [[axis] for axis in axes]

    for axis_row in axes:
        for axis in axis_row:
            axis.axis("off")

    for index, row in enumerate(selected_rows):
        axis = axes[index // columns][index % columns]
        image = Image.open(row["image_path"]).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            "True: {true}\nBase: {base} | ECA: {eca}".format(
                true=row["true_label"],
                base=row["baseline_pred"],
                eca=row["eca_pred"],
            ),
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", output_path)


def main():
    predictions_path = Path(
        os.environ.get("PREDICTIONS_CSV", "outputs/analysis/test_predictions.csv")
    )
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/error_cases"))
    max_images = int(os.environ.get("MAX_IMAGES", 12))
    seed = int(os.environ.get("SEED", 42))

    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(predictions_path)
    rng = random.Random(seed)

    baseline_correct = data["baseline_pred"] == data["true_label"]
    eca_correct = data["eca_pred"] == data["true_label"]

    case_sets = {
        "baseline_wrong_eca_correct": data[~baseline_correct & eca_correct],
        "baseline_correct_eca_wrong": data[baseline_correct & ~eca_correct],
        "both_wrong": data[~baseline_correct & ~eca_correct],
    }

    titles = {
        "baseline_wrong_eca_correct": "Baseline Wrong / ECA Correct",
        "baseline_correct_eca_wrong": "Baseline Correct / ECA Wrong",
        "both_wrong": "Both Models Wrong",
    }

    for case_name, case_data in case_sets.items():
        rows = case_data.to_dict("records")
        rng.shuffle(rows)
        make_case_grid(
            rows=rows,
            title=titles[case_name],
            output_path=output_dir / f"{case_name}.png",
            max_images=max_images,
        )


if __name__ == "__main__":
    main()
