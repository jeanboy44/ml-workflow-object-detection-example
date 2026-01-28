from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class TrainConfig:
    data: str = MISSING
    model: str = MISSING
    project: str = MISSING
    epochs: int = MISSING
    imgsz: int = MISSING
    batch: int = MISSING
    from_scratch: bool = False
    device: Optional[str] = None
    name: Optional[str] = None
    exist_ok: bool = True
    freeze: int = 0
    mosaic: float = 0.0


@dataclass
class MlflowConfig:
    tracking_uri: Optional[str] = "databricks"
    experiment_name: str = MISSING
    run_name: Optional[str] = None
    model_catalog: Optional[str] = None
    model_schema: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class AppConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
