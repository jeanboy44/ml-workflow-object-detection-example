"""Convert PCB VOC annotations into YOLO format.

Example:
    uv run experiments/exp05/prepare_yolo_dataset.py --split-dir data/pcb_splits --output-dir data/pcb_yolo
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import typer

app = typer.Typer()


def load_classes(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        raise FileNotFoundError(
            "[ERROR] classes.txt not found. Run split_pcb_dataset.py first."
        )
    return [line.strip() for line in classes_path.read_text().splitlines() if line]


def parse_annotation(
    xml_path: Path,
) -> tuple[str, int, int, list[tuple[str, list[int]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    if size is None:
        raise ValueError(f"[ERROR] Missing size metadata: {xml_path}")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects: list[tuple[str, list[int]]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name")
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.findtext("xmin"))
        ymin = int(bndbox.findtext("ymin"))
        xmax = int(bndbox.findtext("xmax"))
        ymax = int(bndbox.findtext("ymax"))
        objects.append((label, [xmin, ymin, xmax, ymax]))
    return filename, width, height, objects


def to_yolo_bbox(
    box: list[int], width: int, height: int
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend([f"  - {name}" for name in class_names])
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")


@app.command()
def main(
    split_dir: Path = typer.Option("data/pcb_splits", help="Split dataset directory"),
    output_dir: Path = typer.Option("data/pcb_yolo", help="Output directory"),
):
    """Convert PCB VOC annotations into YOLO format."""
    class_names = load_classes(split_dir / "classes.txt")
    label2id = {name: idx for idx, name in enumerate(class_names)}
    label2id_lower = {name.lower(): idx for name, idx in label2id.items()}

    for split in ["train", "val", "test"]:
        ann_split_dir = split_dir / "annotations" / split
        img_split_dir = split_dir / "images" / split
        if not ann_split_dir.exists():
            print(f"[WARN] Missing {ann_split_dir}, skipping.")
            continue

        for class_dir in sorted(p for p in ann_split_dir.iterdir() if p.is_dir()):
            for xml_path in sorted(class_dir.glob("*.xml")):
                filename, width, height, objects = parse_annotation(xml_path)
                image_path = img_split_dir / class_dir.name / filename
                if not image_path.exists():
                    raise FileNotFoundError(f"[ERROR] Missing image file: {image_path}")

                label_out_dir = output_dir / "labels" / split / class_dir.name
                image_out_dir = output_dir / "images" / split / class_dir.name
                ensure_dir(label_out_dir)
                ensure_dir(image_out_dir)

                label_path = label_out_dir / f"{image_path.stem}.txt"
                lines = []
                for label, box in objects:
                    label_key = label.strip().lower()
                    if label_key not in label2id_lower:
                        raise ValueError(f"[ERROR] Unknown label: {label}")
                    class_id = label2id_lower[label_key]
                    x_center, y_center, box_w, box_h = to_yolo_bbox(box, width, height)
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{box_w:.6f} {box_h:.6f}"
                    )
                label_path.write_text("\n".join(lines) + "\n")

                image_out_path = image_out_dir / image_path.name
                if not image_out_path.exists():
                    image_out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, image_out_path)

    write_data_yaml(output_dir, class_names)
    print(f"[INFO] YOLO dataset saved: {output_dir}")


if __name__ == "__main__":
    app()
