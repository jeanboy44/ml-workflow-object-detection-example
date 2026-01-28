# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow[databricks]>=3.4.0",
#     "pillow>=12.1.0",
#     "python-dotenv>=1.0.1",
#     "typer>=0.21.1",
#     "ultralytics>=8.4.3",
# ]
# ///
# pyright: reportIncompatibleMethodOverride=false
# pyright: reportMissingImports=false
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import mlflow
import numpy as np
import typer
from dotenv import load_dotenv
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema, TensorSpec
from PIL import Image
from ultralytics import YOLO  # type: ignore[attr-defined]

load_dotenv()

app = typer.Typer(help="YOLO MLflow 레지스트리 등록 스크립트")


@dataclass
class Config:
    confidence: float = 0.25


class YoloPredictModel(PythonModel):
    def __init__(self, config: Config) -> None:
        self.config = config

    def load_context(self, context) -> None:
        weights_path = context.artifacts["weights"]
        self.model = YOLO(weights_path)
        self.names = (
            self.model.names
            if isinstance(self.model.names, dict)
            else {idx: name for idx, name in enumerate(self.model.names)}
        )

    def _iter_images(self, model_input) -> Iterable[Image.Image]:
        if isinstance(model_input, dict):
            if "input_image" not in model_input:
                raise ValueError("model_input은 'input_image' 키를 포함해야 합니다.")
            return self._iter_images(model_input["input_image"])
        if isinstance(model_input, np.ndarray):
            return self._images_from_array(model_input)
        if isinstance(model_input, list):
            images = []
            for item in model_input:
                images.extend(self._images_from_array(np.asarray(item)))
            return images
        to_numpy = getattr(model_input, "to_numpy", None)
        if to_numpy is not None:
            return self._images_from_array(np.asarray(to_numpy()))
        raise ValueError("model_input은 이미지 텐서(numpy 배열)를 필요로 합니다.")

    def _images_from_array(self, array: np.ndarray) -> list[Image.Image]:
        if array.ndim == 3:
            array = np.expand_dims(array, axis=0)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError("입력 텐서 shape은 (N, H, W, 3)이어야 합니다.")
        images = []
        for item in array:
            images.append(Image.fromarray(item.astype(np.uint8), mode="RGB"))
        return images

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        images = self._iter_images(model_input)
        outputs = []

        for image in images:
            results = self.model(
                image,
                conf=self.config.confidence,
                verbose=False,
            )[0]
            detections = []
            for box, score, cls in zip(
                results.boxes.xyxy.cpu().tolist(),
                results.boxes.conf.cpu().tolist(),
                results.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                detections.append(
                    {
                        "label": self.names[int(cls)],
                        "score": float(score),
                        "box": [float(coord) for coord in box],
                    }
                )
            outputs.append(json.dumps(detections))

        return outputs


def _validate_model(weights_path: Path, config: Config) -> None:
    model = YoloPredictModel(config)
    context = SimpleNamespace(artifacts={"weights": str(weights_path)})
    model.load_context(context)

    sample = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    output = model.predict(context, {"input_image": sample})

    if not isinstance(output, list) or len(output) != 1:
        raise ValueError("예측 결과는 길이 1의 리스트여야 합니다.")

    try:
        detections = json.loads(output[0])
    except json.JSONDecodeError as exc:
        raise ValueError("예측 결과는 JSON 문자열이어야 합니다.") from exc

    if not isinstance(detections, list):
        raise ValueError("예측 결과 JSON은 리스트여야 합니다.")


@app.command()
def main(
    model_name: str = typer.Argument(..., help="등록할 모델 이름"),
    weights_path: Path = typer.Argument(..., help="YOLO.pt 가중치 경로"),
    run_name: str | None = typer.Option(None, help="MLflow run 이름"),
    confidence: float = typer.Option(0.25, help="YOLO confidence threshold"),
    model_catalog: str = typer.Option("study", help="모델 카탈로그 이름"),
    model_schema: str = typer.Option("object_detection", help="모델 스키마 이름"),
):
    """
    YOLO 모델을 MLflow에 등록하고 이미지 텐서 입력용 pyfunc 모델을 생성합니다.
    """
    model_full_name = f"{model_catalog}.{model_schema}.{model_name}"

    mlflow.set_tracking_uri("databricks")

    signature = ModelSignature(
        inputs=Schema(
            [TensorSpec(np.dtype(np.uint8), (-1, -1, -1, 3), name="input_image")]
        ),
        outputs=Schema([ColSpec("string", "detections_json")]),
    )

    mlflow.set_experiment("/Shared/Models/yolo_models")
    with mlflow.start_run(run_name=run_name) as run:
        config = Config(confidence=confidence)
        try:
            _validate_model(weights_path, config)
            typer.secho("모델 검증 통과", fg=typer.colors.GREEN)
        except Exception as exc:
            typer.secho(f"모델 검증 실패: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        mlflow.pyfunc.log_model(
            name=model_name,
            python_model=YoloPredictModel(config),
            artifacts={"weights": str(weights_path)},
            signature=signature,
            pip_requirements=[
                "mlflow[databricks]>=3.4.0",
                "pillow>=12.1.0",
                "ultralytics>=8.4.3",
            ],
        )
        model_uri = f"runs:/{run.info.run_id}/{model_name}"

    try:
        client = MlflowClient(tracking_uri="databricks")
        try:
            client.create_registered_model(name=model_full_name)
        except Exception as exc:
            message = str(exc)
            if (
                "RESOURCE_ALREADY_EXISTS" not in message
                and "already exists" not in message
            ):
                raise
        result = client.create_model_version(
            name=model_full_name,
            source=model_uri,
            run_id=run.info.run_id,
        )
    except Exception as exc:
        typer.secho(f"등록 실패: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(
        f"등록 요청 완료: {result.name} v{result.version}",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()
