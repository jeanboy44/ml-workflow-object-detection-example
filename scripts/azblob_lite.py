import os
from pathlib import Path

import typer
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv

app = typer.Typer(help="Azure Blob Storage 간단 CLI: list, upload, download 명령 지원")

# .env 환경변수 로딩
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv(
    "AZBLOB_LITE_AZURE_STORAGE_CONNECTION_STRING"
)
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZBLOB_LITE_AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_KEY = os.getenv("AZBLOB_LITE_AZURE_STORAGE_KEY")

# 연결


def get_blob_service_client():
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    elif AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_KEY:
        account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=AZURE_STORAGE_KEY)
    else:
        raise RuntimeError("환경변수에서 Azure Blob 연결 정보가 없습니다.")


def get_container_client(container_name: str) -> ContainerClient:
    service_client = get_blob_service_client()
    return service_client.get_container_client(container_name)


@app.command(help="컨테이너 내부 blob 파일/폴더 리스트 조회")
def list(
    container: str = typer.Option(..., help="대상 Azure Blob 컨테이너명"),
    path: str = typer.Option(None, help="조회할 blob 경로(예: images/)"),
):
    cc = get_container_client(container)
    blobs = cc.list_blobs(name_starts_with=path) if path else cc.list_blobs()
    typer.echo("[Blob 리스트]")
    for blob in blobs:
        typer.echo(blob.name)


# 업로드 logic (src_path → cloud, container_path)
def upload_path(cc: ContainerClient, src_path: Path, container_path: str = ""):
    if src_path.is_file():
        blob_name = (
            str(Path(container_path) / src_path.name)
            if container_path
            else src_path.name
        )
        with open(src_path, "rb") as f:
            cc.upload_blob(name=blob_name, data=f, overwrite=True)
        typer.echo(f"업로드: {blob_name}")
    elif src_path.is_dir():
        for child in src_path.rglob("*"):
            if child.is_file():
                rel_path = child.relative_to(src_path)
                blob_name = (
                    str(Path(container_path) / rel_path)
                    if container_path
                    else str(rel_path)
                )
                with open(child, "rb") as f:
                    cc.upload_blob(name=blob_name, data=f, overwrite=True)
                typer.echo(f"업로드: {blob_name}")
    else:
        raise typer.BadParameter(f"경로가 존재하지 않습니다: {src_path}")


@app.command(help="로컬 파일/폴더를 Azure Blob Storage 컨테이너에 업로드")
def upload(
    container: str = typer.Argument(..., help="대상 Azure Blob 컨테이너명"),
    src_path: str = typer.Option(..., help="로컬 파일/폴더 경로"),
    container_path: str = typer.Option(
        "", help="컨테이너 내 업로드 경로(예: images2026/)"
    ),
):
    """
    예) 파일:
      python scripts/azblob_lite.py upload --src-path ./test.txt --container foo
    예) 폴더:
      python scripts/azblob_lite.py upload --src-path images --container foo --container-path artifacts/google/owlvit-base-patch32
    """
    cc = get_container_client(container)
    upload_path(cc, Path(src_path), container_path)


def download_blob(
    cc,
    src_path: str,
    dst_path: Path,
    container_name: str = "dcim-llm-dev",
):
    """
    Azure Blob Storage의 특정 파일을 로컬로 다운로드합니다.

    Args:
        src_path (str): 컨테이너 내에서 다운로드할 파일의 경로
        dst_path (Path): 로컬 디렉토리로 파일을 저장할 경로
        container_name (str): 컨테이너 이름
    """

    blob_client = cc.get_blob_client(src_path)

    if not blob_client.exists():
        raise FileNotFoundError(f"Blob {src_path} does not exist.")

    with dst_path.open("wb") as f:
        f.write(blob_client.download_blob().readall())


def download_blobs(
    cc,
    prefix: str,
    dst_dir: Path,
):
    blobs = cc.list_blobs(name_starts_with=prefix)
    count = 0
    skipped_conflicts = set()

    for blob in blobs:
        blob_name = blob.name
        typer.secho(f"Found blob: {blob_name} (size: {blob.size})")

        # ✅ 1. 디렉토리처럼 보이는 빈 blob은 저장하지 않음
        if blob.size == 0 and (
            blob_name.endswith("/")
            or "/" in blob_name
            and "." not in blob_name.split("/")[-1]
        ):
            typer.secho(f"Skipping directory-like blob: {blob_name}")
            continue

        dst_path = dst_dir / blob_name

        try:
            # ✅ 2. 상위 경로 중 파일이 존재하면 제거 (파일->디렉토리 전환 허용)
            for parent in reversed(dst_path.parents):
                if parent.exists() and parent.is_file():
                    # logger.warning(
                    #     f"Removing file conflicting with directory: {parent}"
                    # )
                    parent.unlink()  # 파일 삭제
                    break  # 한 번만 처리하면 됨

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # ✅ 3. blob 다운로드
            blob_client = cc.get_blob_client(blob_name)
            with dst_path.open("wb") as f:
                f.write(blob_client.download_blob().readall())
            count += 1

        except Exception:
            # logger.error(f"Failed to download blob: {blob_name}: {e}")
            skipped_conflicts.add(blob_name)

    if count == 0:
        typer.secho(f"No valid blobs found under prefix: {prefix}")
    else:
        typer.secho(f"Downloaded {count} files from {prefix}")

    if skipped_conflicts:
        typer.secho(f"Skipped {len(skipped_conflicts)} blobs due to errors.")


@app.command(help="Azure Blob Storage 컨테이너 파일/폴더를 로컬에 다운로드")
def download(
    container: str = typer.Argument(..., help="다운로드할 Azure Blob 컨테이너명"),
    container_path: str = typer.Option("", help="다운로드할 경로(예: images2026/)"),
    dst_path: str = typer.Option(..., help="로컬 저장 폴더 경로"),
):
    """
    예)
      python scripts/azblob_lite.py download --container foo --container-path images2026/ --dst-path ./downloads
      python scripts/azblob_lite.py download --container foo --dst-path ./allblobs
    """
    cc = get_container_client(container)
    dst = Path(dst_path)
    dst.mkdir(parents=True, exist_ok=True)
    download_blobs(cc, container_path, dst)


if __name__ == "__main__":
    app()
