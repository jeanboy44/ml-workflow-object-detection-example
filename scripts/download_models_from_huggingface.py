# 모델 및 관련 파일을 Hugging Face에서 미리 다운로드 (OWL-ViT 예시)
"""
실행 예시:
    uv run scripts/download_models_from_huggingface.py --model-id google/owlvit-base-patch32 --cache-dir ./models/owlvit
"""

from pathlib import Path

import typer
from huggingface_hub import hf_hub_download, list_repo_files

DEFAULT_MODEL_ID = "google/owlvit-base-patch32"
DEFAULT_FILES = ["pytorch_model.bin", "config.json", "preprocessor_config.json"]


def main(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        help="Hugging Face 모델 리포지토리 ID (ex: google/owlvit-base-patch32)",
    ),
    cache_dir: Path = typer.Option("./models/owlvit", help="모델을 저장할 디렉토리"),
):
    """지정한 모델의 주요 파일을 huggingface에서 다운로드합니다."""
    print(f"[INFO] 다운로드: {model_id} → {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 모델 repo에 어떤 파일이 있는지 확인 (필수 파일만 다운로드)
    all_files = list_repo_files(model_id)
    targets = [f for f in DEFAULT_FILES if f in all_files]
    if not targets:
        print(
            f"다운로드할 핵심 파일이 없습니다. 직접 파일명을 지정하세요. Candidate files: {all_files}"
        )
        return
    for filename in targets:
        local_path = hf_hub_download(
            repo_id=model_id, filename=filename, cache_dir=cache_dir
        )
        print(f"✔ {filename} → {local_path}")
    print(f"[완료] 모든 핵심 모델 파일이 {cache_dir}에 저장되었습니다.")


if __name__ == "__main__":
    typer.run(main)
