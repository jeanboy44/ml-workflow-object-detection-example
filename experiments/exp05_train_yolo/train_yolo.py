"""
Train YOLO on the PCB dataset (fine-tune or from scratch).

Example:
    uv run experiments/exp05/train_yolo.py --data-yaml data/pcb_yolo/data.yaml --model artifacts/yolo/yolo26n.pt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import mlflow
import typer
from dotenv import load_dotenv
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


def register_callbacks(yolo_model: YOLO) -> None:
    def on_fit_epoch_end(trainer) -> None:
        metrics = _epoch_metrics(getattr(trainer, "metrics", {}))
        epoch = int(getattr(trainer, "epoch", 0))
        for key, value in metrics.items():
            mlflow.log_metric(f"epoch/{key}", value, step=epoch)

    yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


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
        "/Shared/Experiments/ml-workflow-object-detection-example/exp05_train_yolo",
        help="MLflow experiment name",
    ),
    run_name: str | None = typer.Option(None, help="MLflow run name"),
    freeze_backbone: int = typer.Option(
        0, help="Freeze backbone layers for transfer learning"
    ),
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

        yolo_model = YOLO(model_path)
        register_callbacks(yolo_model)
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
        )

        results_csv = Path(project) / yolo_run_name / "results.csv"
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
