from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image

from ml_object_detector.logger import logger  # type: ignore


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]


ALLOWED_MODEL_NAMES = (
    "study.object_detection.yolo26n_finetuned_onnx",
    "study.object_detection.yolo26n_pretrained_onnx",
)


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
    if model_alias:
        return f"models:/{model_name}@{model_alias}"
    return f"models:/{model_name}/{model_version}"


def list_models() -> list[str]:
    return list(ALLOWED_MODEL_NAMES)


def list_model_versions(
    model_name: str,
    tracking_uri: str = "databricks",
) -> list[str]:
    load_dotenv()
    if model_name not in ALLOWED_MODEL_NAMES:
        allowed = ", ".join(ALLOWED_MODEL_NAMES)
        raise ValueError(
            f"허용되지 않은 모델: {model_name}. 사용 가능한 모델: {allowed}"
        )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    version_numbers = sorted({str(mv.version) for mv in versions}, key=int)
    return list(version_numbers)


def _cache_root() -> Path:
    cache_root = os.getenv(
        "ML_OBJECT_DETECTOR_CACHE_DIR",
        str(Path.home() / ".cache" / "ml-object-detector"),
    )
    return Path(cache_root).expanduser()


def _cache_dir(
    model_name: str,
    model_version: str | None,
    model_alias: str | None,
) -> Path:
    suffix = model_version if model_version else model_alias
    cache_key = f"{model_name}-{suffix}"
    return _cache_root() / cache_key


def clear_cache(
    model_name: str | None = None,
    model_version: str | None = None,
    model_alias: str | None = None,
) -> list[Path]:
    removed: list[Path] = []
    if model_name is None:
        cache_dir = _cache_root()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed.append(cache_dir)
        return removed

    if model_name not in ALLOWED_MODEL_NAMES:
        allowed = ", ".join(ALLOWED_MODEL_NAMES)
        raise ValueError(
            f"허용되지 않은 모델: {model_name}. 사용 가능한 모델: {allowed}"
        )

    if model_version is None and model_alias is None:
        prefix = f"{model_name}-"
        cache_root = _cache_root()
        if cache_root.exists():
            for path in cache_root.iterdir():
                if path.is_dir() and path.name.startswith(prefix):
                    shutil.rmtree(path)
                    removed.append(path)
        return removed

    cache_dir = _cache_dir(model_name, model_version, model_alias)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        removed.append(cache_dir)
    return removed


def load(
    model_name: str,
    tracking_uri: str = "databricks",
    model_version: str | None = None,
    model_alias: str | None = None,
):
    load_dotenv()
    if model_name not in ALLOWED_MODEL_NAMES:
        allowed = ", ".join(ALLOWED_MODEL_NAMES)
        raise ValueError(
            f"허용되지 않은 모델: {model_name}. 사용 가능한 모델: {allowed}"
        )
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks" and model_name.count(".") != 2:
        raise ValueError("model_name은 {catalog}.{schema}.{name} 형식이어야 합니다.")
    model_uri = _resolve_model_uri(model_name, model_version, model_alias)
    cache_dir = _cache_dir(model_name, model_version, model_alias)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = cache_dir / "MLmodel"
    if cached_model.exists():
        logger.info("Loading cached MLflow model: {}", cache_dir)
        return mlflow.pyfunc.load_model(str(cache_dir))
    logger.info("Downloading MLflow model to cache: {}", cache_dir)
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=str(cache_dir))
    logger.info("Loading MLflow pyfunc model: {}", local_path)
    return mlflow.pyfunc.load_model(local_path)


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
