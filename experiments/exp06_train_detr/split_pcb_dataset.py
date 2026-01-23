"""Split the PCB dataset into ImageFolder splits for HF Datasets.

Example:
    uv run experiments/exp06_train_detr/split_pcb_dataset.py \
        --base-dir data/PCB_DATASET \
        --output-dir data/pcb_splits_imagefolder
"""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer()


@dataclass
class AnnotationItem:
    class_name: str
    xml_path: Path


def parse_annotation(xml_path: Path) -> tuple[str, int, int, list[dict]]:
    """Parse VOC XML and return filename, width, height, and bbox list.

    Returns:
        filename: Image filename
        width: Image width
        height: Image height
        objects: List of {label, xmin, ymin, xmax, ymax}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"No filename in {xml_path}")

    size = root.find("size")
    if size is None:
        raise ValueError(f"No size in {xml_path}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    objects = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "")
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            objects.append(
                {
                    "label": normalize_label(label),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

    return filename, width, height, objects


def normalize_label(label: str) -> str:
    """Normalize label: 'short' -> 'Short', 'Missing_Hole' -> 'Missing_hole'."""
    return label.strip().replace("_", " ").title().replace(" ", "_")


def collect_items(annotations_dir: Path) -> list[AnnotationItem]:
    items: list[AnnotationItem] = []
    for class_dir in sorted(p for p in annotations_dir.iterdir() if p.is_dir()):
        for xml_path in sorted(class_dir.glob("*.xml")):
            items.append(AnnotationItem(class_name=class_dir.name, xml_path=xml_path))
    return items


def split_items(
    items: list[AnnotationItem],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[AnnotationItem]]:
    rng = random.Random(seed)
    train_items: list[AnnotationItem] = []
    val_items: list[AnnotationItem] = []
    test_items: list[AnnotationItem] = []

    items_by_class: dict[str, list[AnnotationItem]] = {}
    for item in items:
        items_by_class.setdefault(item.class_name, []).append(item)

    for class_name, class_items in items_by_class.items():
        class_items = class_items[:]
        rng.shuffle(class_items)
        total = len(class_items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_items.extend(class_items[:train_end])
        val_items.extend(class_items[train_end:val_end])
        test_items.extend(class_items[val_end:])
        print(
            f"[INFO] {class_name}: train={train_end}, val={val_end - train_end}, "
            f"test={total - val_end}"
        )

    return {"train": train_items, "val": val_items, "test": test_items}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_object_metadata(
    objects: list[dict],
    category_name_to_id: dict[str, int],
) -> dict[str, list]:
    bboxes: list[list[float]] = []
    categories: list[int] = []
    for obj in objects:
        label = obj["label"]
        if label not in category_name_to_id:
            print(f"[WARNING] Unknown label: {label}")
            continue
        xmin = float(obj["xmin"])
        ymin = float(obj["ymin"])
        xmax = float(obj["xmax"])
        ymax = float(obj["ymax"])
        bbox_width = max(0.0, xmax - xmin)
        bbox_height = max(0.0, ymax - ymin)
        bboxes.append([xmin, ymin, bbox_width, bbox_height])
        categories.append(category_name_to_id[label])
    return {"bbox": bboxes, "categories": categories}


def create_imagefolder_split(
    split_items: list[AnnotationItem],
    images_dir: Path,
    output_dir: Path,
    category_name_to_id: dict[str, int],
) -> list[dict[str, object]]:
    metadata_entries: list[dict[str, object]] = []
    for image_id, item in enumerate(split_items, start=1):
        filename, _, _, objects = parse_annotation(item.xml_path)
        image_path = images_dir / item.class_name / filename
        if not image_path.exists():
            print(f"[WARNING] Image not found: {image_path}")
            continue

        suffix = image_path.suffix or Path(filename).suffix or ".jpg"
        new_filename = f"{image_id:06d}{suffix}"
        shutil.copy2(image_path, output_dir / new_filename)
        metadata_entries.append(
            {
                "file_name": new_filename,
                "objects": build_object_metadata(objects, category_name_to_id),
            }
        )
    return metadata_entries


@app.command()
def main(
    base_dir: Path = typer.Option(
        "data/PCB_DATASET", help="Base PCB dataset directory"
    ),
    output_dir: Path = typer.Option(
        "data/pcb_splits_imagefolder", help="Output directory for ImageFolder data"
    ),
    train_ratio: float = typer.Option(0.8, help="Train split ratio"),
    val_ratio: float = typer.Option(0.1, help="Validation split ratio"),
    test_ratio: float = typer.Option(0.1, help="Test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Split the PCB dataset into train/val/test ImageFolder splits."""
    annotations_dir = base_dir / "Annotations"
    images_dir = base_dir / "images"

    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(
            "[ERROR] base_dir must contain Annotations/ and images/ directories."
        )

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    items = collect_items(annotations_dir)
    if not items:
        raise ValueError("[ERROR] No annotation files found.")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique class names and create category mapping
    class_names = sorted({normalize_label(item.class_name) for item in items})
    category_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    print(f"[INFO] Categories: {class_names}")

    # Save category mapping
    with open(output_dir / "categories.json", "w") as f:
        json.dump(category_name_to_id, f, indent=2)
    print(f"[INFO] Categories saved: {output_dir / 'categories.json'}")

    # Split items
    split_map = split_items(items, train_ratio, val_ratio, test_ratio, seed)

    # Process each split
    for split_name, items_list in split_map.items():
        print(f"\n[INFO] Processing {split_name} split ({len(items_list)} images)...")

        split_output_dir = output_dir / split_name
        ensure_dir(split_output_dir)

        metadata_entries = create_imagefolder_split(
            items_list,
            images_dir,
            split_output_dir,
            category_name_to_id,
        )

        metadata_file = split_output_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for entry in metadata_entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

        print(
            f"[INFO] {split_name}: {len(metadata_entries)} samples saved to {split_output_dir}"
        )
        print(f"[INFO] Saved: {metadata_file}")

    print(f"\n[SUCCESS] ImageFolder dataset created: {output_dir}")
    print("  Structure:")
    print(f"    {output_dir}/")
    print("      train/              # Training images + metadata.jsonl")
    print("      val/                # Validation images + metadata.jsonl")
    print("      test/               # Test images + metadata.jsonl")
    print("      categories.json")


if __name__ == "__main__":
    app()
