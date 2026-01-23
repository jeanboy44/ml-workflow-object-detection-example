from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import mlflow
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from ultralytics import YOLO, settings


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


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    settings.update({"mlflow": False})

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get(
        "experiment_name",
        "/Shared/Experiments/ml-workflow-object-detection-example/exp05_train_yolo",
    )
    run_name = mlflow_cfg.get("run_name")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_cfg = dict(cfg.get("train", {}))
    from_scratch = bool(train_cfg.pop("from_scratch", False))
    model_path = train_cfg.get("model")
    default_name = "yolo_from_scratch" if from_scratch else "yolo_finetune"
    if not model_path:
        train_cfg["model"] = (
            "yolov8n.yaml" if from_scratch else "artifacts/yolo/yolo26n.pt"
        )
    if "name" not in train_cfg or train_cfg["name"] is None:
        train_cfg["name"] = default_name
    if run_name is None:
        run_name = default_name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dict(cfg))

        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        hydra_dir = output_dir / ".hydra"

        if hydra_dir.exists():
            for yaml_file in hydra_dir.glob("*.yaml"):
                mlflow.log_artifact(str(yaml_file), artifact_path="hydra")

        yolo_model = YOLO(train_cfg["model"])
        register_callbacks(yolo_model)
        yolo_model.train(**train_cfg)


if __name__ == "__main__":
    main()
