import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
CLASSES = ("real", "fake")


def list_images(folder: Path):
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_subset(source_dir: Path, output_dir: Path, images_per_class: int, seed: int):
    rng = random.Random(seed)

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Choose a new output folder or delete it first."
        )

    total_copied = 0

    for split in SPLITS:
        for class_name in CLASSES:
            src_folder = source_dir / split / class_name
            dst_folder = output_dir / split / class_name

            if not src_folder.exists():
                raise FileNotFoundError(f"Missing expected folder: {src_folder}")

            images = list_images(src_folder)
            if not images:
                raise ValueError(f"No supported image files found in: {src_folder}")

            selected = rng.sample(images, k=min(images_per_class, len(images)))
            dst_folder.mkdir(parents=True, exist_ok=True)

            for image_path in selected:
                shutil.copy2(image_path, dst_folder / image_path.name)

            print(f"{split}/{class_name}: copied {len(selected)} images")
            total_copied += len(selected)

    print(f"\nDone. Copied {total_copied} images to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a small balanced sample dataset for quick local testing."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the full dataset containing train/val/test folders.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/sample"),
        type=Path,
        help="Where to create the sample dataset. Default: data/sample",
    )
    parser.add_argument(
        "--images-per-class",
        default=50,
        type=int,
        help="Number of images to copy for each class in each split. Default: 50",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducible sampling. Default: 42",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    copy_subset(
        source_dir=args.source,
        output_dir=args.output,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
