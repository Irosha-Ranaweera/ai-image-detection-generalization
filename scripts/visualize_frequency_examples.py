import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
from PIL import Image

from src.data.transforms import FFTHighPass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path):
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    split = os.environ.get("SPLIT", "test")
    samples_per_class = int(os.environ.get("SAMPLES_PER_CLASS", 3))
    seed = int(os.environ.get("SEED", 42))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/frequency_visuals"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    high_pass_transform = FFTHighPass()
    classes = ["fake", "real"]

    selected = []
    for class_name in classes:
        folder = data_dir / split / class_name
        images = list_images(folder)
        if len(images) < samples_per_class:
            raise ValueError(f"Not enough images in {folder}")
        for image_path in rng.sample(images, samples_per_class):
            selected.append((class_name, image_path))

    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 3))

    for row_index, (class_name, image_path) in enumerate(selected):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        high_pass = high_pass_transform(image)

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{class_name}: RGB")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(high_pass)
        axes[row_index, 1].set_title(f"{class_name}: FFT high-pass")
        axes[row_index, 1].axis("off")

    plt.tight_layout()
    output_path = output_dir / f"{split}_fft_highpass_examples.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved frequency visualization to:", output_path)


if __name__ == "__main__":
    main()
