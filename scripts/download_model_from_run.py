from __future__ import annotations

from pathlib import Path

import mlflow
import typer
from dotenv import load_dotenv
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

load_dotenv()

app = typer.Typer(help="Databricks MLflow run에서 모델 아티팩트 다운로드")


@app.command()
def main(
    run_id: str = typer.Argument(..., help="MLflow run ID"),
    model_path: str = typer.Argument(..., help="run 내 모델 경로"),
    download_dir: Path = typer.Option(Path("artifacts/runs"), help="다운로드 디렉토리"),
):
    """
    Databricks MLflow run에서 모델 아티팩트를 로컬로 다운로드합니다.
    """
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient(tracking_uri="databricks")

    try:
        client.get_run(run_id)
    except Exception as exc:
        typer.secho(f"run 조회 실패: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    model_uri = f"runs:/{run_id}/{model_path}"

    destination = download_dir / run_id
    destination.mkdir(parents=True, exist_ok=True)

    local_path = download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(destination),
    )

    typer.secho(f"다운로드 완료: {local_path}", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
