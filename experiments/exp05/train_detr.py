"""
Train DETR on the PCB dataset (fine-tune or from scratch).

Example:
    uv run experiments/exp05/train_detr.py --data-dir data/pcb_splits --output-dir artifacts/exp05/detr_finetune --pretrained
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlflow
import torch
import typer
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

app = typer.Typer()


@dataclass
class VocTarget:
    image_id: int
    image_size: tuple[int, int]
    annotations: list[dict[str, float]]


class VocDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        label2id: dict[str, int],
    ) -> None:
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.label2id = label2id
        self.annotation_files = sorted(self.annotations_dir.rglob("*.xml"))
        if not self.annotation_files:
            raise ValueError(f"No XML files found in {annotations_dir}")

    def __len__(self) -> int:
        return len(self.annotation_files)

    def __getitem__(self, idx: int) -> tuple[Image.Image, VocTarget]:
        xml_path = self.annotation_files[idx]
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename")
        image_path = self.images_dir / xml_path.parent.name / filename
        image = Image.open(image_path).convert("RGB")

        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        annotations: list[dict[str, float]] = []
        for obj in root.findall("object"):
            label = obj.findtext("name")
            if label not in self.label2id:
                raise ValueError(f"Unknown label: {label}")
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
            box_w = xmax - xmin
            box_h = ymax - ymin
            annotations.append(
                {
                    "bbox": [xmin, ymin, box_w, box_h],
                    "area": box_w * box_h,
                    "category_id": self.label2id[label],
                    "iscrowd": 0,
                }
            )

        target = VocTarget(
            image_id=idx,
            image_size=(height, width),
            annotations=annotations,
        )
        return image, target


def collate_fn(
    processor: DetrImageProcessor,
):
    def _collate(batch: Iterable[tuple[Image.Image, VocTarget]]):
        images, targets = zip(*batch)
        annotations_payload = [
            {"image_id": target.image_id, "annotations": target.annotations}
            for target in targets
        ]
        processed = processor(
            images=list(images),
            annotations=annotations_payload,
            return_tensors="pt",
        )
        processed["raw_targets"] = list(targets)
        return processed

    return _collate


def move_to_device(batch: dict, device: torch.device) -> dict:
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    labels = []
    for target in batch["labels"]:
        labels.append(
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}
        )
    batch["labels"] = labels
    return batch


def build_model(
    pretrained: bool,
    label2id: dict[str, int],
    id2label: dict[int, str],
    freeze_backbone: bool,
) -> DetrForObjectDetection:
    if pretrained:
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        config = DetrConfig.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        model = DetrForObjectDetection(config)
    if freeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    return model


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def evaluate_model(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, float]:
    model.eval()
    total_gt = 0
    total_pred = 0
    true_positive = 0
    iou_sum = 0.0
    matched = 0
    loss_sum = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            raw_targets: list[VocTarget] = batch.pop("raw_targets")
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            loss_sum += outputs.loss.item()
            steps += 1

            target_sizes = torch.tensor(
                [target.image_size for target in raw_targets], device=device
            )
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )

            for result, target in zip(results, raw_targets, strict=True):
                pred_boxes = result["boxes"].to(device)
                pred_scores = result["scores"].to(device)
                pred_boxes = pred_boxes[pred_scores >= score_threshold]

                gt_boxes = torch.tensor(
                    [
                        [
                            ann["bbox"][0],
                            ann["bbox"][1],
                            ann["bbox"][0] + ann["bbox"][2],
                            ann["bbox"][1] + ann["bbox"][3],
                        ]
                        for ann in target.annotations
                    ],
                    device=device,
                )

                total_gt += len(gt_boxes)
                total_pred += len(pred_boxes)

                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                for gt_box in gt_boxes:
                    ious = box_iou(gt_box, pred_boxes)
                    max_iou = float(ious.max().item()) if len(ious) > 0 else 0.0
                    if max_iou >= iou_threshold:
                        true_positive += 1
                        iou_sum += max_iou
                        matched += 1

    precision = true_positive / total_pred if total_pred else 0.0
    recall = true_positive / total_gt if total_gt else 0.0
    mean_iou = iou_sum / matched if matched else 0.0
    mean_loss = loss_sum / max(steps, 1)

    return {
        "loss": mean_loss,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
    }


def train_one_epoch(
    model: DetrForObjectDetection,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    steps = 0
    for batch in dataloader:
        batch.pop("raw_targets", None)
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item()
        steps += 1
    return loss_sum / max(steps, 1)


@app.command()
def main(
    data_dir: Path = typer.Option("data/pcb_splits", help="Split dataset directory"),
    output_dir: Path = typer.Option(
        "artifacts/exp05/detr_finetune", help="Output directory for the model"
    ),
    pretrained: bool = typer.Option(True, help="Use pretrained weights"),
    epochs: int = typer.Option(10, help="Number of epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    seed: int = typer.Option(42, help="Random seed"),
    device: str = typer.Option("cpu", help="Training device"),
    score_threshold: float = typer.Option(0.5, help="Score threshold for eval"),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for eval"),
    tracking_uri: str | None = typer.Option("databricks", help="MLflow tracking URI"),
    experiment_name: str = typer.Option(
        "/Shared/Experiments/ml-workflow-object-detection-example/exp05_train_detr",
        help="MLflow experiment name",
    ),
    run_name: str = typer.Option("detr_train", help="MLflow run name"),
    freeze_backbone: bool = typer.Option(
        False, help="Freeze backbone for transfer learning"
    ),
):
    """Train DETR with MLflow tracking."""
    load_dotenv()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    random.seed(seed)
    torch.manual_seed(seed)

    classes_path = data_dir / "classes.txt"
    class_names = [
        line.strip() for line in classes_path.read_text().splitlines() if line.strip()
    ]
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for name, idx in label2id.items()}

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    train_dataset = VocDetectionDataset(
        data_dir / "images" / "train",
        data_dir / "annotations" / "train",
        label2id,
    )
    val_dataset = VocDetectionDataset(
        data_dir / "images" / "val",
        data_dir / "annotations" / "val",
        label2id,
    )
    test_dataset = VocDetectionDataset(
        data_dir / "images" / "test",
        data_dir / "annotations" / "test",
        label2id,
    )

    collate = collate_fn(processor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = build_model(pretrained, label2id, id2label, freeze_backbone)
    device_obj = torch.device(device)
    model.to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "pretrained": pretrained,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "freeze_backbone": freeze_backbone,
                "score_threshold": score_threshold,
                "iou_threshold": iou_threshold,
            }
        )

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device_obj)
            val_metrics = evaluate_model(
                model,
                processor,
                val_loader,
                device_obj,
                score_threshold,
                iou_threshold,
            )
            mlflow.log_metric("train/loss", train_loss, step=epoch)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val/{key}", value, step=epoch)
            print(
                f"[INFO] epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
            )

        test_metrics = evaluate_model(
            model,
            processor,
            test_loader,
            device_obj,
            score_threshold,
            iou_threshold,
        )
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test/{key}", value)

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        print(f"[INFO] Model saved: {output_dir}")


if __name__ == "__main__":
    app()
