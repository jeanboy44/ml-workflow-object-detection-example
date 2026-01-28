from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import mlflow
from dotenv import load_dotenv
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


def _model_cache_dir(cache_root: Path, model_uri: str) -> Path:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in model_uri).strip("_")
    digest = hashlib.sha256(model_uri.encode("utf-8")).hexdigest()[:8]
    name = f"{sanitized}-{digest}" if sanitized else digest
    return cache_root / name


def _find_weight_file(search_dir: Path) -> Path | None:
    weights = sorted(search_dir.rglob("*.pt"))
    for weight in weights:
        if weight.is_file() and weight.stat().st_size > 0:
            return weight
    return None


def load(model_uri: str, tracking_uri: str = "databricks") -> YOLO:
    load_dotenv()
    cache_root = os.getenv("ML_OBJECT_DETECTOR_CACHE_DIR")
    cache_dir = Path(cache_root).expanduser() if cache_root else None
    if cache_dir:
        cache_dir = _model_cache_dir(cache_dir, model_uri)
        cached_weight = _find_weight_file(cache_dir)
        if cached_weight:
            logger.info("Using cached YOLO weights: {}", cached_weight)
            return YOLO(str(cached_weight))
    logger.info("Loading MLflow model: {}", model_uri)
    mlflow.set_tracking_uri(tracking_uri)
    local_dir = mlflow.artifacts.download_artifacts(model_uri)
    weight = _find_weight_file(Path(local_dir))
    if not weight:
        logger.error("No weights found under {}", local_dir)
        raise FileNotFoundError(f"No .pt weights found under {local_dir}")
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / weight.name
        shutil.copy2(weight, cached_path)
        weight = cached_path
    logger.info("Loaded YOLO weights: {}", weight)
    return YOLO(str(weight))


def predict(
    model: YOLO, image: Image.Image, threshold: float = 0.25
) -> DetectionResult:
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
