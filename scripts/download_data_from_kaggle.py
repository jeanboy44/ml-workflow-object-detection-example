"""
KAGGLE_API_TOKEN 설정 필요

https://www.kaggle.com/datasets/akhatova/pcb-defects
"""

from pathlib import Path

import typer
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()  # .env 파일에서 환경 변수 로드
app = typer.Typer(help="PCB Defect Dataset Downloader")


@app.command()
def download_data(
    output_dir: Path = typer.Argument(
        ...,
        help="데이터셋을 저장할 경로 (예: ./data/pcb)",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    dataset_slug: str = typer.Option(
        "akhatova/pcb-defects", help="Kaggle 데이터셋 Slug (유저명/데이터셋명)"
    ),
    unzip: bool = typer.Option(True, help="다운로드 후 압축 해제 여부"),
):
    """
    Kaggle API를 사용하여 PCB 결함 데이터셋을 다운로드합니다.
    """
    try:
        # 1. 경로 생성 (pathlib 사용)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            typer.secho(f"📂 디렉토리 생성 완료: {output_dir}", fg=typer.colors.GREEN)

        # 2. Kaggle API 인증
        typer.echo("🔑 Kaggle API 인증 중...")
        api = KaggleApi()
        api.authenticate()

        # 3. 데이터셋 다운로드
        typer.secho(
            f"⬇️  다운로드 시작: {dataset_slug} -> {output_dir}", fg=typer.colors.BLUE
        )

        # Kaggle API는 기본적으로 zip 파일을 다운로드합니다.
        # unzip=True 옵션을 주면 라이브러리가 알아서 해제하지만,
        # 진행 상황 제어 등을 위해 직접 처리할 수도 있습니다. 여기서는 라이브러리 기능 활용.
        api.dataset_download_files(
            dataset_slug, path=output_dir, unzip=unzip, quiet=False
        )

        typer.secho("✅ 다운로드 및 압축 해제 완료!", fg=typer.colors.GREEN, bold=True)

        # 다운로드된 파일 목록 보여주기
        typer.echo("\n[다운로드된 파일 목록]")
        for file_path in output_dir.iterdir():
            typer.echo(f" - {file_path.name}")

    except Exception as e:
        typer.secho(f"❌ 오류 발생: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
