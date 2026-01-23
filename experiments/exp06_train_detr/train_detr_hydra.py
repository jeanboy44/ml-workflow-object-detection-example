"""Train DETR on the PCB dataset with Hydra config and COCO format.

Example:
    uv run experiments/exp06_train_detr/train_detr_hydra.py
    uv run experiments/exp06_train_detr/train_detr_hydra.py train.epochs=20 train.batch_size=8
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
import mlflow
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from mlflow.models import infer_signature
from omegaconf import MISSING, DictConfig, OmegaConf
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput


@dataclass
class TrainingArgsConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-4
    eval_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    logging_steps: Optional[int] = None
    max_steps: Optional[int] = None
    remove_unused_columns: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    seed: int = 42
    report_to: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    data_dir: str = MISSING
    output_dir: str = MISSING
    seed: int = 42
    device: str = "cpu"
    training_args: TrainingArgsConfig = field(default_factory=TrainingArgsConfig)


@dataclass
class MlflowConfig:
    tracking_uri: str = "databricks"
    experiment_name: str = MISSING
    run_name: Optional[str] = None
    model_name: Optional[str] = None  # {catalog}.{schema}.{model} format


@dataclass
class AppConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


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


def resolve_image_size(train_cfg: dict[str, Any], processor: DetrImageProcessor) -> int:
    if train_cfg.get("image_size"):
        return int(train_cfg["image_size"])
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        return int(size.get("shortest_edge") or size.get("max_size") or 800)
    if isinstance(size, int):
        return size
    return 800


def build_signature(
    model: DetrForObjectDetection,
    image_size: int,
) -> tuple[Any, dict[str, np.ndarray]]:
    pixel_values = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    pixel_mask = torch.ones((1, image_size, image_size), dtype=torch.int64)
    input_example = {
        "pixel_values": pixel_values.cpu().numpy(),
        "pixel_mask": pixel_mask.cpu().numpy(),
    }
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    output_example = {
        "logits": outputs.logits.detach().cpu().numpy(),
        "pred_boxes": outputs.pred_boxes.detach().cpu().numpy(),
    }
    signature = infer_signature(input_example, output_example)
    return signature, input_example


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


def transform_for_detr(processor: DetrImageProcessor):
    """Create a transform function for DETR that processes images and annotations."""

    def _transform(example_batch):
        images = example_batch["image"]
        batch_annotations = []

        for idx, image in enumerate(images):
            objects = (
                example_batch.get("objects", [{}])[idx]
                if example_batch.get("objects")
                else {}
            )
            bboxes = objects.get("bbox", []) if objects else []
            categories = objects.get("categories", []) if objects else []

            annotations = []
            for ann_id, (bbox, category_id) in enumerate(
                zip(bboxes, categories, strict=False), start=1
            ):
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": int(category_id),
                        "bbox": [float(value) for value in bbox],
                        "area": float(bbox[2]) * float(bbox[3])
                        if len(bbox) == 4
                        else 0.0,
                        "iscrowd": 0,
                    }
                )
            batch_annotations.append({"image_id": idx, "annotations": annotations})

        encoding = processor(
            images=images,
            annotations=batch_annotations,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": encoding["labels"],
        }

    return _transform


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


def resolve_split(dataset_dict, primary: str, fallback: str | None = None):
    """Resolve dataset split by primary name or fallback."""
    if primary in dataset_dict:
        return dataset_dict[primary]
    if fallback and fallback in dataset_dict:
        return dataset_dict[fallback]
    raise KeyError(f"[ERROR] Missing {primary} split in dataset.")


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    schema = OmegaConf.structured(AppConfig)
    OmegaConf.set_struct(schema, False)
    OmegaConf.set_struct(schema.train, False)
    OmegaConf.set_struct(schema.mlflow, False)
    cfg = OmegaConf.merge(schema, cfg)

    try:
        OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        )
    except Exception as exc:
        print(f"[ERROR] Config validation failed: {exc}")
        return

    OmegaConf.set_readonly(cfg.train, True)
    OmegaConf.set_readonly(cfg.mlflow, True)

    mlflow_cfg = cfg.mlflow

    train_cfg = OmegaConf.to_container(cfg.train, resolve=True) or {}
    data_dir = Path(train_cfg["data_dir"])
    output_dir = Path(train_cfg["output_dir"])
    seed = train_cfg["seed"]
    device = train_cfg["device"]

    training_args_cfg = {
        k: v for k, v in train_cfg["training_args"].items() if v is not None
    }
    # batch_size = training_args_cfg["per_device_train_batch_size"]

    random.seed(seed)
    torch.manual_seed(seed)

    # Load category mapping from COCO format dataset
    categories_path = data_dir / "categories.json"
    with open(categories_path) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    processor = DetrImageProcessor.from_pretrained("artifacts/facebook/detr-resnet-50")

    model = build_model(
        pretrained=True,
        label2id=label2id,
        id2label=id2label,
        freeze_backbone=False,
        model_path=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    default_run_name = "detr_train"
    if mlflow_cfg.run_name is None:
        run_name = default_run_name
    else:
        run_name = mlflow_cfg.run_name

    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)
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

        dataset_dict = load_dataset("imagefolder", data_dir=str(data_dir))
        train_split = resolve_split(dataset_dict, "train")
        val_split = resolve_split(dataset_dict, "val", "validation")
        test_split = resolve_split(dataset_dict, "test")

        transform = transform_for_detr(processor)
        train_dataset = train_split.with_transform(transform)
        val_dataset = val_split.with_transform(transform)
        test_dataset = test_split.with_transform(transform)

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
            compute_metrics=build_compute_metrics(processor, 0.5, 0.5),
            callbacks=[MLflowCallback()],
        )

        trainer.train()

        best_model_path = trainer.state.best_model_checkpoint
        if best_model_path:
            best_model = DetrForObjectDetection.from_pretrained(best_model_path)
        else:
            best_model = trainer.model

        registered_model_name = None
        if mlflow_cfg.model_name:
            parts = mlflow_cfg.model_name.split(".")
            if len(parts) != 3:
                raise ValueError(
                    "mlflow.model_name should be in {catalog}.{schema}.{model} format"
                )
            registered_model_name = mlflow_cfg.model_name

        image_size = resolve_image_size(train_cfg, processor)
        signature, input_example = build_signature(best_model, image_size)
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        print(f"[INFO] Model saved: {output_dir}")


if __name__ == "__main__":
    main()
