# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fsspec>=2026.1.0",
#     "python-dotenv>=1.2.1",
#     "typer>=0.21.1",
# ]
# ///
from pathlib import Path

import fsspec
import typer
from dotenv import load_dotenv
from fsspec.callbacks import TqdmCallback

# Typer 앱 초기화
app = typer.Typer(help="Azure Blob Storage 업로드 유틸리티")

load_dotenv()


@app.command()
def main(
    container: str = typer.Argument(..., help="Azure Container 이름"),
    src_path: str = typer.Option(..., help="로컬 파일 또는 폴더 경로 (파일명/폴더명)"),
    blob_path: str = typer.Option(..., help="업로드될 Blob 경로 (파일 및 폴더명 포함)"),
    connection_string: str = typer.Option(
        None,
        envvar="AZURE_STORAGE_CONNECTION_STRING",
        help="Azure Connection String (환경변수 설정 권장)",
    ),
):
    """
    로컬 파일 또는 폴더를 Azure Blob Storage로 업로드합니다.
    """

    # 1. 연결 문자열 확인
    if not connection_string:
        typer.secho(
            "Error: Connection String이 필요합니다. --connection-string 옵션이나 AZURE_STORAGE_CONNECTION_STRING 환경변수를 설정해주세요.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    src = Path(src_path)
    if not src.exists():
        typer.secho(
            f"Error: 소스 경로를 찾을 수 없습니다: {src_path}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    try:
        # 2. Azure 파일시스템 연결
        fs = fsspec.filesystem("az", connection_string=connection_string)

        full_remote_path = f"{container}/{blob_path}"

        typer.secho(
            f"업로드 시작: {src_path} -> {full_remote_path}", fg=typer.colors.GREEN
        )

        callback = TqdmCallback(
            tqdm_kwargs={"desc": "Uploading", "unit": "b", "unit_scale": True}
        )

        # 3. 업로드 수행 (recursive=True로 폴더/파일 모두 대응)
        fs.put(str(src), full_remote_path, recursive=True, callback=callback)

        typer.secho("업로드 완료!", fg=typer.colors.BLUE, bold=True)

    except Exception as e:
        typer.secho(f"오류 발생: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
