from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import hydra
import mlflow
import numpy as np
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from mlflow.models import infer_signature
from omegaconf import MISSING, DictConfig, OmegaConf
from ultralytics import YOLO, settings


@dataclass
class TrainConfig:
    data: str = MISSING
    model: str = MISSING
    project: str = MISSING
    name: str = MISSING
    epochs: int = MISSING
    imgsz: int = MISSING
    batch: int = MISSING
    lr0: Optional[float] = None
    warmup_epochs: int = 0
    device: Optional[str] = None
    exist_ok: bool = True
    freeze: int = 0
    mosaic: float = 0.0


@dataclass
class MlflowConfig:
    tracking_uri: Optional[str] = "databricks"
    experiment_name: str = MISSING
    run_name: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class AppConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def _epoch_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            clean[key] = float(value)
    return clean


def on_fit_epoch_end(trainer) -> None:
    metrics = _epoch_metrics(getattr(trainer, "metrics", {}))
    epoch = int(getattr(trainer, "epoch", 0))
    for key, value in metrics.items():
        mlflow.log_metric(f"epoch/{key}", value, step=epoch)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        weight_path = Path(save_dir) / "weights" / "last.pt"
        if weight_path.exists():
            mlflow.log_artifact(
                str(weight_path), artifact_path=f"epoch_weights/epoch_{epoch}"
            )


def build_signature(yolo_model: YOLO, imgsz: int) -> tuple[Any, Any]:
    input_example = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    results = yolo_model(input_example)
    output_example = np.empty((0, 6), dtype=np.float32)
    if results:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and getattr(boxes, "data", None) is not None:
            output_example = boxes.data.cpu().numpy()
    signature = infer_signature(input_example, output_example)
    return signature, input_example


def log_best_and_last_models(
    save_dir: Path,
    registered_model_name: str | None,
    imgsz: int,
) -> None:
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if not best_path.exists() and not last_path.exists():
        print(f"[WARN] No model checkpoints found in {save_dir}")
        return
    if best_path.exists():
        mlflow.log_artifact(str(best_path), artifact_path="weights/best_model")
    if last_path.exists():
        mlflow.log_artifact(str(last_path), artifact_path="weights/last_model")


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # argument parsing and config validation
    load_dotenv()
    schema = OmegaConf.structured(AppConfig)
    OmegaConf.set_struct(schema, False)
    OmegaConf.set_struct(schema.train, False)
    OmegaConf.set_struct(schema.mlflow, False)
    cfg = OmegaConf.merge(schema, cfg)

    try:
        final_config = OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        )
    except Exception as exc:
        print(f"[ERROR] Config validation failed: {exc}")
        return

    if cfg.mlflow.model_name:
        # check if model_name is in {catalog}.{schema}.{model} format
        parts = cfg.mlflow.model_name.split(".")
        if len(parts) != 3:
            raise ValueError(
                "mlflow.model_name should be in {catalog}.{schema}.{model} format"
            )

    OmegaConf.set_readonly(cfg.train, True)
    OmegaConf.set_readonly(cfg.mlflow, True)

    # training with mlflow logging
    settings.update({"mlflow": False})
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        if isinstance(final_config, dict):
            mlflow.log_params(cast(dict[str, Any], final_config))

        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        hydra_dir = output_dir / ".hydra"

        if hydra_dir.exists():
            for yaml_file in hydra_dir.glob("*.yaml"):
                mlflow.log_artifact(str(yaml_file), artifact_path="hydra")

        yolo_model = YOLO(cfg.train.model)
        yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        yolo_model.train(**OmegaConf.to_container(cfg.train, resolve=True))

        save_dir = Path("runs/detect") / cfg.train.project / cfg.train.name
        if cfg.mlflow.model_name:
            log_best_and_last_models(
                Path(save_dir),
                cfg.mlflow.model_name,
                int(cfg.train.get("imgsz", 640)),
            )


if __name__ == "__main__":
    main()
