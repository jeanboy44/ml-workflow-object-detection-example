from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from PIL import Image


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


def _resolve_model_uri(
    model_name: str,
    model_version: str | None,
    model_alias: str | None,
) -> str:
    if model_version and model_alias:
        raise ValueError("model_version 또는 model_alias 중 하나만 입력해야 합니다.")
    if not model_version and not model_alias:
        raise ValueError("model_version 또는 model_alias 중 하나는 필수입니다.")
    suffix = model_version if model_version else model_alias
    return f"models:/{model_name}/{suffix}"


def load(
    model_name: str,
    tracking_uri: str = "databricks",
    model_version: str | None = None,
    model_alias: str | None = None,
):
    load_dotenv()
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks" and model_name.count(".") != 2:
        raise ValueError("model_name은 {catalog}.{schema}.{name} 형식이어야 합니다.")
    model_uri = _resolve_model_uri(model_name, model_version, model_alias)
    logger.info("Loading MLflow pyfunc model: {}", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def predict(model, image: Image.Image, threshold: float = 0.25) -> DetectionResult:
    array = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)
    outputs = model.predict(array)
    if not outputs:
        return DetectionResult(detections=[])

    try:
        detections_raw = json.loads(outputs[0])
    except json.JSONDecodeError as exc:
        raise ValueError("예측 결과는 JSON 문자열이어야 합니다.") from exc

    detections = []
    for item in detections_raw:
        box_values = [float(coord) for coord in item.get("box", [])]
        if len(box_values) != 4:
            continue
        detections.append(
            Detection(
                label=str(item.get("label", "")),
                score=float(item.get("score", 0.0)),
                box=(box_values[0], box_values[1], box_values[2], box_values[3]),
            )
        )
    logger.info("Detections: {}", len(detections))
    return DetectionResult(detections=detections)
