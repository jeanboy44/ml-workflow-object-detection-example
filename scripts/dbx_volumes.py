"""Databricks Volume 파일 유틸리티."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from rich.logging import RichHandler
from tqdm.auto import tqdm

load_dotenv()

logger.remove()
logger.add(
    RichHandler(rich_tracebacks=True, markup=False),
    format="{message}",
    level="INFO",
)

app = typer.Typer(help="Databricks Volume 파일 유틸리티입니다.")


def list_volume_entries(client: WorkspaceClient, src_path: str) -> list:
    """Databricks Volume에서 파일/디렉터리를 재귀적으로 수집합니다."""

    def collect_entries(path: str) -> list:
        entries: list = []
        for entry in client.files.list_directory_contents(path):
            if entry.path is None:
                continue
            entries.append(entry)
            if entry.is_directory:
                entries.extend(collect_entries(entry.path))
        return entries

    return collect_entries(src_path)


@app.command("download")
def download_resources(
    src_path: Annotated[
        str,
        typer.Argument(
            help="원본 경로 (예: /Volumes/catalog/schema/volume/folder)입니다.",
        ),
    ],
    dst_path: Annotated[Path, typer.Option(help="로컬 저장 디렉터리입니다.")] = Path(
        "./data"
    ),
    workers: Annotated[
        int,
        typer.Option(help="동시 다운로드 수입니다."),
    ] = 16,
) -> None:
    """Databricks Volume의 파일을 내려받습니다."""
    client = WorkspaceClient()
    dst_path.mkdir(parents=True, exist_ok=True)

    entries = list_volume_entries(client, src_path)
    files = [entry for entry in entries if not entry.is_directory]
    logger.info(f"Files to download: {len(files)}")

    def build_local_path(entry_path: str) -> Path:
        src_root = Path(src_path).parent
        rel_path = Path(entry_path).relative_to(src_root)
        return dst_path / rel_path

    def download_file(entry_path: str) -> str | None:
        local_path = build_local_path(entry_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        worker_client = WorkspaceClient()
        response = worker_client.files.download(entry_path)
        if response.contents is None:
            return entry_path
        with response.contents as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        return None

    total = len(files)
    skipped: list[str] = []
    failed: list[tuple[str, Exception]] = []
    file_paths = [entry.path for entry in files]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(download_file, path): path for path in file_paths}
        for index, future in enumerate(
            tqdm(
                as_completed(futures), total=total, desc="Downloading", colour="green"
            ),
            start=1,
        ):
            path = futures[future]
            try:
                skipped_path = future.result()
            except Exception as exc:
                failed.append((path, exc))
                continue
            if skipped_path:
                skipped.append(skipped_path)

    for path in skipped:
        logger.warning(f"Skipping empty download: {path}")
    if failed:
        logger.error(f"Failed downloads: {len(failed)}")
        for path, exc in failed:
            logger.error(f"Download error: {path} -> {exc}")

    logger.success(f"Download complete: {dst_path}")


@app.command("list")
def list_resources(
    src_path: Annotated[
        str,
        typer.Argument(
            help="원본 경로 (예: /Volumes/catalog/schema/volume/folder)입니다.",
        ),
    ],
) -> None:
    """Databricks Volume의 파일 목록을 출력한다."""
    client = WorkspaceClient()
    entries = list_volume_entries(client, src_path)
    logger.info(f"Entries found: {len(entries)}")
    for entry in entries:
        logger.info(entry.path)


if __name__ == "__main__":
    app()
