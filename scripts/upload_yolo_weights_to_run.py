# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow[databricks]>=3.4.0",
#     "python-dotenv>=1.2.1",
#     "typer>=0.21.1",
# ]
# ///
from __future__ import annotations

from pathlib import Path

import mlflow
import typer
from dotenv import load_dotenv

app = typer.Typer(help="MLflow run에 YOLO 가중치 아티팩트 업로드")


@app.command()
def main(
    run_id: str = typer.Argument(..., help="MLflow run ID"),
    weights_path: Path = typer.Argument(..., help="YOLO 가중치 경로"),
    artifact_path: str = typer.Option(
        "yolo_weights",
        help="run 내 아티팩트 저장 경로",
    ),
):
    """
    로컬 YOLO 가중치를 지정한 run의 아티팩트로 업로드합니다.
    """
    load_dotenv()
    mlflow.set_tracking_uri("databricks")

    if not weights_path.exists():
        typer.secho(f"파일 없음: {weights_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with mlflow.start_run(run_id=run_id):
        if weights_path.is_dir():
            mlflow.log_artifacts(str(weights_path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(weights_path), artifact_path=artifact_path)

    typer.secho(
        f"업로드 완료: {artifact_path}",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()
