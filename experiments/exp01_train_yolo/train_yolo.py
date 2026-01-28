"""
Train YOLO on the PCB dataset (fine-tune or from scratch).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import typer
from dotenv import load_dotenv
from mlflow.models import infer_signature
from ultralytics import YOLO, settings

app = typer.Typer()


def read_last_metrics(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    with csv_path.open() as csvfile:
        rows = list(csv.DictReader(csvfile))
    if not rows:
        return {}
    last_row = rows[-1]
    metrics: dict[str, float] = {}
    for key, value in last_row.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}{key}", value)


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
    print(f"[INFO] Registering model from {model_path}")
    if registered_model_name:
        print(f"[INFO] MLflow registered_model_name={registered_model_name}")
    best_model = YOLO(str(model_path))
    signature, input_example = build_signature(best_model, imgsz)
    mlflow.pytorch.log_model(
        best_model.model,
        artifact_path="best_model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )
    print("[INFO] Model registration complete")


@app.command()
def main(
    data_yaml: Path = typer.Option(..., help="Path to YOLO data.yaml"),
    model: str | None = typer.Option(None, help="Pretrained weights or model yaml"),
    from_scratch: bool = typer.Option(False, help="Train from scratch"),
    epochs: int = typer.Option(50, help="Number of epochs"),
    imgsz: int = typer.Option(640, help="Image size"),
    batch: int = typer.Option(16, help="Batch size"),
    device: str | None = typer.Option(None, help="Training device"),
    project: str = typer.Option("runs/exp05", help="YOLO output project path"),
    name: str | None = typer.Option(None, help="YOLO run name"),
    tracking_uri: str | None = typer.Option("databricks", help="MLflow tracking URI"),
    experiment_name: str = typer.Option(
        "/Shared/Experiments/exp05_train_yolo",
        help="MLflow experiment name",
    ),
    run_name: str | None = typer.Option(None, help="MLflow run name"),
    catalog: str = typer.Option("study", help="Unity Catalog name"),
    schema: str = typer.Option("object_detection", help="Unity schema name"),
    model_name: str = typer.Option("yolo_best", help="Registered model name"),
    freeze_backbone: int = typer.Option(
        0, help="Freeze backbone layers for transfer learning"
    ),
    fraction: float = typer.Option(1.0, help="Fraction of training data to use"),
):
    """Train YOLO with MLflow tracking."""
    load_dotenv()
    settings.update({"mlflow": False})
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if from_scratch:
        model_path = model or "yolov8n.yaml"
        default_name = "yolo_from_scratch"
    else:
        model_path = model or "artifacts/yolo/yolo26n.pt"
        default_name = "yolo_finetune"

    run_name = run_name or default_name
    yolo_run_name = name or default_name
    yolo_log_dir = Path("runs/detect") / project / yolo_run_name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_path": model_path,
                "from_scratch": from_scratch,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "project": project,
                "name": yolo_run_name,
                "freeze_backbone": freeze_backbone,
            }
        )

        print("[INFO] Starting YOLO training")
        yolo_model = YOLO(model_path)
        yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        yolo_model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=yolo_run_name,
            exist_ok=True,
            freeze=freeze_backbone,
            fraction=fraction,
        )
        print("[INFO] YOLO training complete")

        registered_model_name = f"{catalog}.{schema}.{model_name}"
        print(f"[INFO] Looking for checkpoints in {yolo_log_dir}")
        log_best_model(yolo_log_dir, registered_model_name, imgsz)

        results_csv = yolo_log_dir / "results.csv"
        train_metrics = read_last_metrics(results_csv)
        log_metrics("train/", train_metrics)

        try:
            val_results = yolo_model.val(data=str(data_yaml), split="test")
            results_dict = getattr(val_results, "results_dict", None)
            if isinstance(results_dict, dict):
                log_metrics("test/", results_dict)
        except Exception as exc:
            print(f"[WARN] validation step failed: {exc}")

        if results_csv.exists():
            mlflow.log_artifact(str(results_csv))

    print("[INFO] Training completed")


if __name__ == "__main__":
    app()
