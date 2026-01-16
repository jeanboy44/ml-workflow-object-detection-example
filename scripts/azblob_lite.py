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
    src_path: str = typer.Argument(..., help="로컬 파일/폴더 경로"),
    container: str = typer.Option(..., help="대상 Azure Blob 컨테이너명"),
    container_path: str = typer.Option(
        "", help="컨테이너 내 업로드 경로(예: images2026/)"
    ),
):
    """
    예) 파일:
      python scripts/azblob_lite.py upload --src-path ./test.txt --container foo
    예) 폴더:
      python scripts/azblob_lite.py upload --src-path images --container foo --container-path 2024imgs
    """
    cc = get_container_client(container)
    upload_path(cc, Path(src_path), container_path)


# 다운로드 (cloud→local)
def download_blobs(cc: ContainerClient, container_path: str, dst_path: Path):
    # container_path(=다운로드 기준경로) 길이만큼 blob.name 앞부분을 제거하여 저장
    prefix_len = len(container_path) if container_path else 0
    blobs = (
        cc.list_blobs(name_starts_with=container_path)
        if container_path
        else cc.list_blobs()
    )
    for blob in blobs:
        # blob.name 에서 container_path(prefix) 부분을 제거
        local_rel_path = (
            blob.name[prefix_len:]
            if prefix_len and blob.name.startswith(container_path)
            else blob.name
        )
        local_rel_path = local_rel_path.lstrip("/")
        local_path = dst_path / local_rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            data = cc.download_blob(blob.name).readall()
            f.write(data)
        typer.echo(f"다운로드: {local_path}")


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
