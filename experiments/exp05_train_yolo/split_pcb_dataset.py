"""Split the PCB dataset into train/val/test.

Example:
    uv run experiments/exp05/split_pcb_dataset.py --base-dir data/PCB_DATASET --output-dir data/pcb_splits
"""

from __future__ import annotations

import csv
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


def parse_annotation(xml_path: Path) -> tuple[str, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    labels = [obj.findtext("name") for obj in root.findall("object")]
    return filename, labels


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


def copy_item(
    item: AnnotationItem,
    images_dir: Path,
    output_dir: Path,
    split: str,
) -> tuple[Path, Path]:
    filename, _ = parse_annotation(item.xml_path)
    image_path = images_dir / item.class_name / filename
    if not image_path.exists():
        raise FileNotFoundError(f"[ERROR] Missing image file: {image_path}")

    ann_out_dir = output_dir / "annotations" / split / item.class_name
    img_out_dir = output_dir / "images" / split / item.class_name
    ensure_dir(ann_out_dir)
    ensure_dir(img_out_dir)

    ann_output = ann_out_dir / item.xml_path.name
    img_output = img_out_dir / image_path.name
    shutil.copy2(item.xml_path, ann_output)
    shutil.copy2(image_path, img_output)
    return img_output, ann_output


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["split", "class", "image_path", "annotation_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Split manifest saved: {manifest_path}")


@app.command()
def main(
    base_dir: Path = typer.Option(
        "data/PCB_DATASET", help="Base PCB dataset directory"
    ),
    output_dir: Path = typer.Option(
        "data/pcb_splits", help="Output directory for split data"
    ),
    train_ratio: float = typer.Option(0.8, help="Train split ratio"),
    val_ratio: float = typer.Option(0.1, help="Validation split ratio"),
    test_ratio: float = typer.Option(0.1, help="Test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Split the PCB dataset into train/val/test."""
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

    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted({item.class_name for item in items})
    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n")
    print(f"[INFO] Classes saved: {output_dir / 'classes.txt'}")

    split_map = split_items(items, train_ratio, val_ratio, test_ratio, seed)
    manifest_rows: list[dict[str, str]] = []

    for split, split_items_list in split_map.items():
        for item in split_items_list:
            img_output, ann_output = copy_item(item, images_dir, output_dir, split)
            manifest_rows.append(
                {
                    "split": split,
                    "class": item.class_name,
                    "image_path": str(img_output),
                    "annotation_path": str(ann_output),
                }
            )

    write_manifest(manifest_rows, output_dir)
    print(f"[INFO] Done. Output: {output_dir}")


if __name__ == "__main__":
    app()
