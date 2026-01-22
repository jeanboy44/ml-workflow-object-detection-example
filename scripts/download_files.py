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
app = typer.Typer(help="Azure Blob Storage 다운로드 유틸리티")

load_dotenv()


@app.command()
def main(
    container: str = typer.Argument(..., help="Azure Container 이름"),
    blob_path: str = typer.Option(..., help="다운로드할 Blob 경로 (폴더 또는 파일명)"),
    dst_path: str = typer.Option(
        ..., help="로컬 저장 경로 (저장될 파일명 또는 폴더명)"
    ),
    connection_string: str = typer.Option(
        None,
        envvar="AZURE_STORAGE_CONNECTION_STRING",
        help="Azure Connection String (환경변수 설정 권장)",
    ),
):
    """
    Azure Blob Storage에서 파일 또는 폴더를 다운로드합니다.
    """

    # 1. 연결 문자열 확인
    if not connection_string:
        typer.secho(
            "Error: Connection String이 필요합니다. --connection-string 옵션이나 AZURE_STORAGE_CONNECTION_STRING 환경변수를 설정해주세요.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        # 2. Azure 파일시스템 연결
        fs = fsspec.filesystem("az", connection_string=connection_string)

        full_remote_path = f"{container}/{blob_path}"

        # 3. 경로 존재 여부 확인 (선택 사항이지만 안전을 위해)
        if not fs.exists(full_remote_path):
            typer.secho(
                f"Error: 원격 경로를 찾을 수 없습니다: {full_remote_path}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        typer.secho(
            f"다운로드 시작: {full_remote_path} -> {dst_path}", fg=typer.colors.GREEN
        )

        # 4. 다운로드 수행 (recursive=True로 폴더/파일 모두 대응)
        parent = Path(dst_path).parent
        if not parent.exists() and str(parent) != "":
            parent.mkdir(parents=True, exist_ok=True)

        callback = TqdmCallback(
            tqdm_kwargs={"desc": "Downloading", "unit": "b", "unit_scale": True}
        )
        fs.get(full_remote_path, dst_path, recursive=True, callback=callback)

        typer.secho("다운로드 완료!", fg=typer.colors.BLUE, bold=True)

    except Exception as e:
        typer.secho(f"오류 발생: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
