from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import mlflow
from loguru import logger
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load(model_uri: str, tracking_uri: str = "databricks") -> YOLO:
    logger.info("Loading MLflow model: {}", model_uri)
    mlflow.set_tracking_uri(tracking_uri)
    local_dir = mlflow.artifacts.download_artifacts(model_uri)
    weights = list(Path(local_dir).rglob("*.pt"))
    if not weights:
        logger.error("No weights found under {}", local_dir)
        raise FileNotFoundError(f"No .pt weights found under {local_dir}")
    logger.info("Loaded YOLO weights: {}", weights[0])
    return YOLO(str(weights[0]))


def predict(model: YOLO, image: Image.Image, threshold: float = 0.25) -> DetectionResult:
    logger.info("Running detection with threshold {}", threshold)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name, conf=threshold)[0]
    detections = []
    for box, score, cls in zip(
        results.boxes.xyxy.cpu().tolist(),
        results.boxes.conf.cpu().tolist(),
        results.boxes.cls.cpu().tolist(),
        strict=True,
    ):
        label = model.names[int(cls)]
        detections.append(Detection(label=label, score=score, box=tuple(box)))
    logger.info("Detections: {}", len(detections))
    return DetectionResult(detections=detections)
