from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from mlflow.models import infer_signature
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
    load_dotenv()
    settings.update({"mlflow": False})

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get(
        "experiment_name",
        "/Shared/Experiments/exp05_train_yolo",
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

        registered_model_name = None
        if catalog and schema and model_name:
            registered_model_name = f"{catalog}.{schema}.{model_name}"
        save_dir = getattr(getattr(yolo_model, "trainer", None), "save_dir", None)
        if save_dir is None:
            save_dir = Path(train_cfg.get("project", "runs/exp05")) / train_cfg["name"]
        log_best_model(
            Path(save_dir),
            registered_model_name,
            int(train_cfg.get("imgsz", 640)),
        )


if __name__ == "__main__":
    main()
