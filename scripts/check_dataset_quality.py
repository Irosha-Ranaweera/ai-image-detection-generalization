import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
CLASSES = ("fake", "real")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def summarize_numbers(values):
    if not values:
        return {
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def scan_dataset(data_dir: Path, check_duplicates: bool):
    split_rows = []
    corrupted = []
    widths = []
    heights = []
    aspect_ratios = []
    extension_counts = Counter()
    hash_to_paths = defaultdict(list)

    for split in SPLITS:
        for class_name in CLASSES:
            folder = data_dir / split / class_name
            if not folder.exists():
                raise FileNotFoundError(f"Missing expected folder: {folder}")

            image_paths = [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            split_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "count": len(image_paths),
                }
            )

            for image_path in tqdm(image_paths, desc=f"{split}/{class_name}"):
                extension_counts[image_path.suffix.lower()] += 1

                try:
                    with Image.open(image_path) as image:
                        image.verify()

                    with Image.open(image_path) as image:
                        width, height = image.size
                        widths.append(width)
                        heights.append(height)
                        if height:
                            aspect_ratios.append(width / height)
                except Exception as error:
                    corrupted.append(
                        {
                            "path": str(image_path),
                            "error": str(error),
                        }
                    )
                    continue

                if check_duplicates:
                    hash_to_paths[file_sha256(image_path)].append(str(image_path))

    duplicate_groups = [
        paths for paths in hash_to_paths.values() if len(paths) > 1
    ]

    summary = {
        "data_dir": str(data_dir),
        "total_images": sum(row["count"] for row in split_rows),
        "split_class_counts": split_rows,
        "extensions": dict(extension_counts),
        "width": summarize_numbers(widths),
        "height": summarize_numbers(heights),
        "aspect_ratio": summarize_numbers(aspect_ratios),
        "corrupted_count": len(corrupted),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_image_count": sum(len(group) for group in duplicate_groups),
    }

    return summary, corrupted, duplicate_groups


def save_split_counts_csv(split_rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "class", "count"])
        writer.writeheader()
        writer.writerows(split_rows)


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/dataset_quality"))
    check_duplicates = (
        os.environ.get("CHECK_DUPLICATES", "false").lower() == "true"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary, corrupted, duplicate_groups = scan_dataset(
        data_dir=data_dir,
        check_duplicates=check_duplicates,
    )

    with (output_dir / "dataset_quality_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    save_split_counts_csv(
        summary["split_class_counts"],
        output_dir / "split_class_counts.csv",
    )

    with (output_dir / "corrupted_images.json").open("w", encoding="utf-8") as file:
        json.dump(corrupted, file, indent=2)

    with (output_dir / "duplicate_groups.json").open("w", encoding="utf-8") as file:
        json.dump(duplicate_groups, file, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved dataset quality outputs to:", output_dir)


if __name__ == "__main__":
    main()
