# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow[databricks]>=3.4.0",
#     "onnxruntime>=1.17.0",
#     "pillow>=12.1.0",
#     "python-dotenv>=1.0.1",
#     "typer>=0.21.1",
# ]
# ///
# pyright: reportIncompatibleMethodOverride=false
# pyright: reportMissingImports=false
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import mlflow
import numpy as np
import onnxruntime as ort
import typer
from dotenv import load_dotenv
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema, TensorSpec
from PIL import Image

load_dotenv()

app = typer.Typer(help="YOLO ONNX 모델을 MLflow 레지스트리에 등록합니다.")


@dataclass
class Config:
    confidence: float = 0.25
    imgsz: int = 640


class YoloOnnxPredictModel(PythonModel):
    def __init__(self, config: Config) -> None:
        self.config = config

    def load_context(self, context) -> None:
        weights_path = context.artifacts["weights"]
        self.session = ort.InferenceSession(
            str(weights_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.input_shape = self.input_meta.shape
        self.names = self._resolve_names(context)

    def _resolve_names(self, context) -> dict[int, str]:
        names = None
        names_path = context.artifacts.get("names")
        if names_path:
            try:
                with Path(names_path).open("r", encoding="utf-8") as handle:
                    names = json.load(handle)
            except Exception:
                names = None
        if isinstance(names, dict):
            return {int(key): value for key, value in names.items()}
        if isinstance(names, list):
            return {idx: name for idx, name in enumerate(names)}

        return {}

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

    def _closest_stride_size(self, size: int, stride: int = 32) -> int:
        return max(stride, (size // stride) * stride)

    def _resolve_input_size(self, image: Image.Image) -> tuple[int, int]:
        if len(self.input_shape) >= 4:
            height = self.input_shape[2]
            width = self.input_shape[3]
            if isinstance(height, int) and isinstance(width, int):
                return width, height
        size = self._closest_stride_size(self.config.imgsz)
        return size, size

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, int, int]:
        original_width, original_height = image.size
        width, height = self._resolve_input_size(image)
        resized = image.resize((width, height))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0), original_width, original_height

    def _extract_detections(self, outputs: Sequence[object]) -> np.ndarray:
        if not outputs:
            return np.empty((0, 6), dtype=np.float32)
        detections = np.asarray(outputs[0])
        if detections.ndim == 3:
            detections = detections[0]
        if detections.ndim == 2 and detections.shape[-1] >= 6:
            return detections[:, :6]
        return np.empty((0, 6), dtype=np.float32)

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        resized_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        resized_w, resized_h = resized_size
        original_w, original_h = original_size

        boxes = boxes.copy()
        if np.max(boxes[:, :4]) <= 1.5:
            boxes[:, [0, 2]] *= resized_w
            boxes[:, [1, 3]] *= resized_h
        boxes[:, [0, 2]] *= original_w / resized_w
        boxes[:, [1, 3]] *= original_h / resized_h
        return boxes

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        images = self._iter_images(model_input)
        outputs = []

        for image in images:
            input_tensor, original_w, original_h = self._preprocess(image)
            resized_w = input_tensor.shape[3]
            resized_h = input_tensor.shape[2]
            outputs_raw = self.session.run(None, {self.input_name: input_tensor})
            detections = self._extract_detections(outputs_raw)
            detections = detections[detections[:, 4] >= self.config.confidence]
            detections = self._scale_boxes(
                detections,
                (resized_w, resized_h),
                (original_w, original_h),
            )
            detections_list = []
            for det in detections:
                x1, y1, x2, y2, score, cls = det[:6]
                label = self.names.get(int(cls), str(int(cls)))
                detections_list.append(
                    {
                        "label": label,
                        "score": float(score),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )
            outputs.append(json.dumps(detections_list))

        return outputs


def _validate_model(predict_model: PythonModel, weights_path: Path) -> None:
    context = SimpleNamespace(artifacts={"weights": str(weights_path)})
    predict_model.load_context(context)

    sample = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    output = predict_model.predict(context, sample)

    if not isinstance(output, list) or len(output) != 1:
        raise ValueError("예측 결과는 길이 1의 리스트여야 합니다.")

    try:
        detections = json.loads(output[0])
    except json.JSONDecodeError as exc:
        raise ValueError("예측 결과는 JSON 문자열이어야 합니다.") from exc

    if not isinstance(detections, list):
        raise ValueError("예측 결과 JSON은 리스트여야 합니다.")


def _build_pip_requirements() -> list[str]:
    return [
        "mlflow[databricks]>=3.4.0",
        "onnxruntime>=1.17.0",
        "pillow>=12.1.0",
    ]


@app.command()
def main(
    model_name: str = typer.Argument(..., help="등록할 모델 이름"),
    model_path: Path = typer.Argument(..., help="YOLO .onnx 경로"),
    run_name: str | None = typer.Option(None, help="MLflow run 이름"),
    confidence: float = typer.Option(0.25, help="YOLO confidence threshold"),
    imgsz: int = typer.Option(640, help="ONNX 입력 이미지 사이즈"),
    model_catalog: str = typer.Option("study", help="모델 카탈로그 이름"),
    model_schema: str = typer.Option("object_detection", help="모델 스키마 이름"),
):
    """
    YOLO ONNX 모델을 MLflow에 등록하고 이미지 텐서 입력용 pyfunc 모델을 생성합니다.
    """
    model_full_name = f"{model_catalog}.{model_schema}.{model_name}"

    mlflow.set_tracking_uri("databricks")

    signature = ModelSignature(
        inputs=Schema(
            [TensorSpec(np.dtype(np.uint8), (-1, -1, -1, 3), name="input_image")]
        ),
        outputs=Schema([ColSpec("string", "detections_json")]),
    )

    input_example = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    if input_example.dtype != np.uint8 or input_example.ndim != 4:
        raise ValueError("input_example은 uint8의 (N, H, W, 3) 텐서여야 합니다.")

    if not model_path.exists():
        raise typer.BadParameter(f"모델 파일을 찾을 수 없습니다: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise typer.BadParameter(f"ONNX 모델만 지원합니다: {model_path}")

    mlflow.set_experiment("/Shared/Models/yolo_models")
    with mlflow.start_run(run_name=run_name) as run:
        config = Config(confidence=confidence, imgsz=imgsz)
        try:
            _validate_model(YoloOnnxPredictModel(config), model_path)
            typer.secho("모델 검증 통과", fg=typer.colors.GREEN)
        except Exception as exc:
            typer.secho(f"모델 검증 실패: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        predict_model = YoloOnnxPredictModel(config)

        mlflow.pyfunc.log_model(
            name=model_name,
            python_model=predict_model,
            artifacts={"weights": str(model_path)},
            signature=signature,
            code_paths=[str(Path(__file__))],
            input_example=input_example,
            pip_requirements=_build_pip_requirements(),
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
