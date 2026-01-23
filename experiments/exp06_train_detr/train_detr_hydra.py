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
from torch.utils.data import Dataset, Subset
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
    model_path: str | None,
) -> DetrForObjectDetection:
    backbone_path = model_path or "artifacts/facebook/detr-resnet-50"
    if pretrained:
        model = DetrForObjectDetection.from_pretrained(
            backbone_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            use_pretrained_backbone=False,
        )
    else:
        config = DetrConfig.from_pretrained(
            backbone_path,
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
    def _extract_predictions(predictions):
        if isinstance(predictions, dict):
            return predictions.get("logits"), predictions.get("pred_boxes")

        if isinstance(predictions, (list, tuple)):
            values = []
            for item in predictions:
                if isinstance(item, dict):
                    if "logits" in item and "pred_boxes" in item:
                        return item["logits"], item["pred_boxes"]
                    continue
                values.append(item)
            if len(values) >= 2:
                return values[0], values[1]

        return None, None

    def _compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        logits, pred_boxes = _extract_predictions(predictions)

        if logits is None or pred_boxes is None:
            return {"precision": 0.0, "recall": 0.0, "mean_iou": 0.0}

        outputs = DetrObjectDetectionOutput(
            logits=torch.tensor(logits),
            pred_boxes=torch.tensor(pred_boxes),
        )

        if isinstance(label_ids, tuple) and len(label_ids) == 1:
            label_ids = label_ids[0]

        if isinstance(label_ids, np.ndarray):
            labels = label_ids.tolist()
        else:
            labels = label_ids

        if isinstance(labels, dict):
            labels = [labels]
        elif labels and isinstance(labels[0], (list, tuple)):
            flattened = []
            for item in labels:
                flattened.extend(item)
            labels = flattened

        targets = []
        for label in labels:
            target = {k: torch.tensor(v) for k, v in label.items()}
            targets.append(target)

        target_sizes = torch.stack([target["orig_size"] for target in targets])
        if target_sizes.shape[0] != outputs.logits.shape[0]:
            return {"precision": 0.0, "recall": 0.0, "mean_iou": 0.0}
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


def maybe_subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        return dataset
    return Subset(dataset, list(range(min(len(dataset), max_samples))))


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig) -> None:
    load_dotenv()

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get(
        "experiment_name",
        "/Shared/Experiments/ml-workflow-object-detection-example/exp06_train_detr",
    )
    run_name = mlflow_cfg.get("run_name")
    registry_uri = mlflow_cfg.get("registry_uri")
    catalog = mlflow_cfg.get("catalog")
    schema = mlflow_cfg.get("schema")
    model_name = mlflow_cfg.get("model_name")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)

    train_cfg = OmegaConf.to_container(cfg.get("train", {}), resolve=True) or {}
    data_dir = Path(train_cfg["data_dir"])
    output_dir = Path(train_cfg["output_dir"])
    seed = train_cfg["seed"]
    device = train_cfg["device"]

    training_args_cfg = dict(train_cfg["training_args"])
    # epochs = training_args_cfg["num_train_epochs"]
    batch_size = training_args_cfg["per_device_train_batch_size"]

    random.seed(seed)
    torch.manual_seed(seed)

    # Load category mapping from COCO format dataset
    categories_path = data_dir / "categories.json"
    with open(categories_path) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    processor_kwargs = {}
    if train_cfg.get("image_size"):
        processor_kwargs = {
            "size": {"shortest_edge": train_cfg["image_size"]},
            "max_size": train_cfg["image_size"],
        }
    processor = DetrImageProcessor.from_pretrained(
        "artifacts/facebook/detr-resnet-50", **processor_kwargs
    )

    model = build_model(
        train_cfg["pretrained"],
        label2id,
        id2label,
        train_cfg["freeze_backbone"],
        train_cfg.get("model_path"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    default_run_name = "detr_train"
    if run_name is None:
        run_name = default_run_name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra/config.yaml")
        for key, value in train_cfg.items():
            if key == "training_args":
                continue
            mlflow.log_param(f"train.{key}", value)
        for key, value in training_args_cfg.items():
            mlflow.log_param(f"train.training_args.{key}", value)

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

        fast_dev_run = train_cfg.get("fast_dev_run", False)
        max_train_samples = train_cfg.get("max_train_samples")
        max_eval_samples = train_cfg.get("max_eval_samples")
        max_test_samples = train_cfg.get("max_test_samples")

        if fast_dev_run:
            if max_train_samples is None:
                max_train_samples = batch_size * 2
            if max_eval_samples is None:
                max_eval_samples = batch_size * 2
            if max_test_samples is None:
                max_test_samples = batch_size * 2

        train_dataset = maybe_subset(train_dataset, max_train_samples)
        val_dataset = maybe_subset(val_dataset, max_eval_samples)
        test_dataset = maybe_subset(test_dataset, max_test_samples)

        use_mps = device == "mps"
        use_cpu = device == "cpu"

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            use_cpu=use_cpu,
            use_mps_device=use_mps,
            **training_args_cfg,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=build_collate_fn(processor),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor,
            compute_metrics=build_compute_metrics(
                processor,
                train_cfg.get("score_threshold", 0.5),
                train_cfg.get("iou_threshold", 0.5),
            ),
            callbacks=[MLflowCallback()],
        )

        trainer.train()

        best_model_path = trainer.state.best_model_checkpoint
        if best_model_path:
            best_model = DetrForObjectDetection.from_pretrained(best_model_path)
        else:
            best_model = trainer.model
        registered_model_name = None
        if catalog and schema and model_name:
            registered_model_name = f"{catalog}.{schema}.{model_name}"
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name=registered_model_name,
        )
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        print(f"[INFO] Model saved: {output_dir}")


if __name__ == "__main__":
    main()
