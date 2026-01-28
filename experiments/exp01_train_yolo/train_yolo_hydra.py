from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import numpy as np
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO, settings  # type: ignore[attr-defined]


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


def log_best_and_last_models(save_dir: Path) -> None:
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    if not best_path.exists() and not last_path.exists():
        print(f"[WARN] No model checkpoints found in {save_dir}")
        return
    if best_path.exists():
        mlflow.log_artifact(str(best_path), artifact_path="weights/best_model")
    if last_path.exists():
        mlflow.log_artifact(str(last_path), artifact_path="weights/last_model")


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    final_config = OmegaConf.to_container(cfg, resolve=True)

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
        train_params = cast(
            dict[str, Any], OmegaConf.to_container(cfg.train, resolve=True)
        )
        yolo_model.train(**train_params)

        save_dir = Path("../../runs/detect") / cfg.train.project / cfg.train.name
        log_best_and_last_models(Path(save_dir))


if __name__ == "__main__":
    main()
