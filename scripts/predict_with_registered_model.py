#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow[databricks]>=3.4.0",
#     "numpy>=2.1.0",
#     "pillow>=12.1.0",
#     "python-dotenv>=1.2.1",
#     "typer>=0.21.1",
#     "ultralytics>=8.4.8",
# ]
# ///
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import typer
from dotenv import load_dotenv
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

app = typer.Typer(help="MLflow 레지스트리 모델 다운로드 및 예측 예제")


def _resolve_model_version(
    client: MlflowClient,
    model_full_name: str,
    model_version: str | None,
    model_stage: str | None,
) -> str:
    if model_version and model_stage:
        raise typer.BadParameter(
            "model_version과 model_stage는 동시에 지정할 수 없습니다."
        )
    if model_version:
        return model_version
    if model_stage:
        return model_stage

    versions = client.search_model_versions(f"name = '{model_full_name}'")
    if not versions:
        raise typer.BadParameter(f"등록된 버전을 찾을 수 없습니다: {model_full_name}")
    latest = max(versions, key=lambda item: int(item.version))
    return latest.version


def _draw_detections(
    image: Image.Image,
    detections: list[dict[str, object]],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = [
        "#e24a33",
        "#348abd",
        "#988ed5",
        "#777777",
        "#fbc15e",
        "#8eba42",
        "#ffb5b8",
    ]

    for index, detection in enumerate(detections):
        box = detection.get("box")
        label = detection.get("label", "unknown")
        score = detection.get("score", 0.0)
        if not isinstance(box, list) or len(box) != 4:
            continue
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"
        text_x = x1 + 4
        text_y = max(y1 + 4, 0.0)
        text_bbox = draw.textbbox((text_x, text_y), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_x, text_y), text, fill="white", font=font)

    return image


@app.command()
def main(
    model_name: str = typer.Argument(..., help="등록된 모델 이름"),
    image_path: Path = typer.Argument(..., help="예측에 사용할 이미지 경로"),
    model_version: str | None = typer.Option(None, help="사용할 모델 버전"),
    model_stage: str | None = typer.Option(None, help="사용할 모델 스테이지"),
    model_catalog: str = typer.Option("study", help="모델 카탈로그 이름"),
    model_schema: str = typer.Option("object_detection", help="모델 스키마 이름"),
    output_path: Path | None = typer.Option(
        None, help="시각화 이미지 저장 경로 (기본: data/predictions)"
    ),
    download_dir: Path = typer.Option(
        Path("artifacts/registered_models"), help="모델 다운로드 디렉토리"
    ),
):
    """
    MLflow 레지스트리에서 모델을 다운로드하고 단일 이미지 예측을 수행합니다.
    """

    model_full_name = f"{model_catalog}.{model_schema}.{model_name}"

    if not image_path.exists():
        raise typer.BadParameter(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    mlflow.set_tracking_uri("databricks")
    client = MlflowClient(tracking_uri="databricks")
    resolved_version = _resolve_model_version(
        client=client,
        model_full_name=model_full_name,
        model_version=model_version,
        model_stage=model_stage,
    )

    model_uri = f"models:/{model_full_name}/{resolved_version}"
    download_dir.mkdir(parents=True, exist_ok=True)
    local_path = download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(download_dir),
    )

    typer.secho(f"모델 다운로드 완료: {local_path}", fg=typer.colors.GREEN)

    model = mlflow.pyfunc.load_model(local_path)
    image = Image.open(image_path).convert("RGB")
    input_tensor = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)

    predictions = model.predict(input_tensor)
    if not predictions:
        typer.echo("예측 결과가 없습니다.")
        raise typer.Exit(code=0)

    typer.echo("예측 결과:")
    for idx, item in enumerate(predictions, start=1):
        detections = json.loads(item)
        typer.echo(f"- image[{idx}]: {detections}")

        annotated = _draw_detections(image.copy(), detections)
        if output_path is None:
            output_dir = Path("data/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_pred.jpg"
        annotated.save(output_path)
        typer.secho(f"시각화 저장 완료: {output_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
