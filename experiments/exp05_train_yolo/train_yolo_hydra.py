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


def register_callbacks(yolo_model: YOLO) -> None:
    def on_fit_epoch_end(trainer) -> None:
        metrics = _epoch_metrics(getattr(trainer, "metrics", {}))
        epoch = int(getattr(trainer, "epoch", 0))
        for key, value in metrics.items():
            mlflow.log_metric(f"epoch/{key}", value, step=epoch)

    yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


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


def log_best_model(
    save_dir: Path,
    registered_model_name: str | None,
    imgsz: int,
) -> None:
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    model_path = best_path if best_path.exists() else last_path
    if not model_path.exists():
        print(f"[WARN] No model checkpoint found in {save_dir}")
        return
    best_model = YOLO(str(model_path))
    signature, input_example = build_signature(best_model, imgsz)
    mlflow.pytorch.log_model(
        best_model.model,
        artifact_path="best_model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )


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
        register_callbacks(yolo_model)
        yolo_model.train(**OmegaConf.to_container(cfg.train, resolve=True))

        save_dir = Path("runs/detect") / cfg.train.project / cfg.train.name
        if cfg.mlflow.model_name:
            log_best_model(
                Path(save_dir),
                cfg.mlflow.model_name,
                int(cfg.train.get("imgsz", 640)),
            )


if __name__ == "__main__":
    main()
