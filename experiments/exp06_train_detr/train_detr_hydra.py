"""Train DETR on the PCB dataset with Hydra config and COCO format.

Example:
    uv run experiments/exp06_train_detr/train_detr_hydra.py
    uv run experiments/exp06_train_detr/train_detr_hydra.py train.epochs=20 train.batch_size=8
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput


def build_model(
    pretrained: bool,
    label2id: dict[str, int],
    id2label: dict[int, str],
    freeze_backbone: bool,
) -> DetrForObjectDetection:
    if pretrained:
        model = DetrForObjectDetection.from_pretrained(
            "artifacts/facebook/detr-resnet-50",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            use_pretrained_backbone=False,
        )
    else:
        config = DetrConfig.from_pretrained(
            "artifacts/facebook/detr-resnet-50",
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


def rescale_bboxes(boxes: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    img_h, img_w = size.tolist()
    x_c, y_c, w, h = boxes.unbind(1)
    x0 = (x_c - 0.5 * w) * img_w
    y0 = (y_c - 0.5 * h) * img_h
    x1 = (x_c + 0.5 * w) * img_w
    y1 = (y_c + 0.5 * h) * img_h
    return torch.stack([x0, y0, x1, y1], dim=1)


class CocoDetrDataset(Dataset):
    def __init__(
        self, root: Path, ann_file: Path, processor: DetrImageProcessor
    ) -> None:
        self.dataset = CocoDetection(root=str(root), annFile=str(ann_file))
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, target = self.dataset[idx]
        image_id = target[0]["image_id"] if target else idx
        encoding = self.processor(
            images=image,
            annotations={"image_id": image_id, "annotations": target},
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items() if k != "labels"}
        item["labels"] = encoding["labels"][0]
        return item


def build_collate_fn(processor: DetrImageProcessor):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        batch_encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": batch_encoding["pixel_values"],
            "pixel_mask": batch_encoding["pixel_mask"],
            "labels": labels,
        }

    return _collate


def build_compute_metrics(
    processor: DetrImageProcessor, score_threshold: float, iou_threshold: float
):
    def _compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        if isinstance(predictions, (list, tuple)):
            logits, pred_boxes = predictions[:2]
        else:
            logits, pred_boxes = predictions

        outputs = DetrObjectDetectionOutput(
            logits=torch.tensor(logits),
            pred_boxes=torch.tensor(pred_boxes),
        )

        if isinstance(label_ids, np.ndarray):
            labels = label_ids.tolist()
        else:
            labels = label_ids

        targets = []
        for label in labels:
            target = {k: torch.tensor(v) for k, v in label.items()}
            targets.append(target)

        target_sizes = torch.stack([target["orig_size"] for target in targets])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_threshold
        )

        total_gt = 0
        total_pred = 0
        true_positive = 0
        iou_sum = 0.0
        matched = 0

        for result, target in zip(results, targets, strict=True):
            pred_boxes_abs = result["boxes"]
            gt_boxes_abs = rescale_bboxes(target["boxes"], target["orig_size"])

            total_gt += len(gt_boxes_abs)
            total_pred += len(pred_boxes_abs)

            if len(gt_boxes_abs) == 0 or len(pred_boxes_abs) == 0:
                continue

            for gt_box in gt_boxes_abs:
                ious = box_iou(gt_box, pred_boxes_abs)
                max_iou = float(ious.max().item()) if len(ious) > 0 else 0.0
                if max_iou >= iou_threshold:
                    true_positive += 1
                    iou_sum += max_iou
                    matched += 1

        precision = true_positive / total_pred if total_pred else 0.0
        recall = true_positive / total_gt if total_gt else 0.0
        mean_iou = iou_sum / matched if matched else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "mean_iou": mean_iou,
        }

    return _compute_metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get(
        "experiment_name",
        "/Shared/Experiments/ml-workflow-object-detection-example/exp06_train_detr",
    )
    run_name = mlflow_cfg.get("run_name")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_cfg = dict(cfg.get("train", {}))
    data_dir = Path(train_cfg.get("data_dir", "data/pcb_splits"))
    output_dir = Path(train_cfg.get("output_dir", "artifacts/exp06/detr_finetune"))
    pretrained = bool(train_cfg.get("pretrained", True))
    epochs = int(train_cfg.get("epochs", 10))
    batch_size = int(train_cfg.get("batch_size", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    seed = int(train_cfg.get("seed", 42))
    device = train_cfg.get("device", "cpu")
    score_threshold = float(train_cfg.get("score_threshold", 0.5))
    iou_threshold = float(train_cfg.get("iou_threshold", 0.5))
    freeze_backbone = bool(train_cfg.get("freeze_backbone", False))

    random.seed(seed)
    torch.manual_seed(seed)

    # Load category mapping from COCO format dataset
    categories_path = data_dir / "categories.json"
    with open(categories_path) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    processor = DetrImageProcessor.from_pretrained("artifacts/facebook/detr-resnet-50")

    model = build_model(pretrained, label2id, id2label, freeze_backbone)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_run_name = "detr_train"
    if run_name is None:
        run_name = default_run_name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra/config.yaml")
        mlflow.log_params({f"train.{key}": value for key, value in train_cfg.items()})

        hydra_cfg = HydraConfig.get()
        hydra_output_dir = Path(hydra_cfg.runtime.output_dir)
        hydra_dir = hydra_output_dir / ".hydra"

        if hydra_dir.exists():
            for yaml_file in hydra_dir.glob("*.yaml"):
                mlflow.log_artifact(str(yaml_file), artifact_path="hydra")

        train_dataset = CocoDetrDataset(
            root=data_dir / "train",
            ann_file=data_dir / "annotations_train.json",
            processor=processor,
        )
        val_dataset = CocoDetrDataset(
            root=data_dir / "val",
            ann_file=data_dir / "annotations_val.json",
            processor=processor,
        )
        test_dataset = CocoDetrDataset(
            root=data_dir / "test",
            ann_file=data_dir / "annotations_test.json",
            processor=processor,
        )

        use_mps = device == "mps"
        use_cpu = device == "cpu"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            remove_unused_columns=False,
            seed=seed,
            dataloader_drop_last=False,
            run_name=run_name,
            use_cpu=use_cpu,
            use_mps_device=use_mps,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=build_collate_fn(processor),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor,
            compute_metrics=build_compute_metrics(
                processor, score_threshold, iou_threshold
            ),
            callbacks=[MLflowCallback()],
        )

        trainer.train()
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        print(f"[INFO] Model saved: {output_dir}")


if __name__ == "__main__":
    main()
