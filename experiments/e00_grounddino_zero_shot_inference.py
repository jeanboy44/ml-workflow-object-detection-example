"""
Zero-Shot Object Detection with Hugging Face GroundingDINO

[실행 예시]
uv run experiments/e00_grounddino_zero_shot_inference.py data/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg --prompt "missing_hole" --save-path data/output_grounddino/sample1.jpg --threshold 0.7

* 모델 경로/이미지 경로는 workspace 내 실제 데이터 기준으로 입력 필요

사전 준비:
  uv run scripts/download_files.py jgryu --blob-path ml-workflow-object-detection-example/IDEA-Research/grounding-dino-base/ --dst-path artifacts/IDEA-Research/grounding-dino-base/

* GroundingDINO, transformers, torch, typer 등 설치 필요 (에러 발생시 pip install)
"""

import sys
from pathlib import Path

import torch
import typer
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from experiments.utils import load_image, plot_detections


def main(
    image_path: Path = typer.Argument(
        ...,
        help="입력 이미지 경로 (ex: data/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg)",
    ),
    prompt: str = typer.Option(..., help="탐지할 텍스트 (쉼표로 구분, ex: 'cat,dog')"),
    model_name: str = typer.Option(
        "artifacts/IDEA-Research/grounding-dino-base",
        help="huggingface 저장소명 또는 로컬 폴더 (ex: artifacts/IDEA-Research/grounding-dino-base)",
    ),
    threshold: float = typer.Option(0.3, help="신뢰도 임계값 (0~1, ex: 0.3)"),
    save_path: Path = typer.Option(
        "result.jpg", help="검출 결과 이미지 경로 (ex: result.jpg)"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", help="cpu|cuda 자동 선택 기본"
    ),
):
    # 입력 경로들이 상대경로/절대경로 혼용돼도 안정 동작하도록 보정
    image_path = Path(image_path)
    save_path = Path(save_path)
    model_name = str(model_name)
    if not image_path.exists():
        typer.echo(f"[ERROR] 이미지 파일이 존재하지 않습니다: {image_path}")
        sys.exit(1)
    if not Path(model_name).exists():
        typer.echo(
            f"[WARNING] 모델 경로가 존재하지 않습니다: {model_name} (transformers repo명일 수 있음)"
        )
    processor = GroundingDinoProcessor.from_pretrained(model_name)
    model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(device)
    image = load_image(image_path)
    text_query = [t.strip() for t in prompt.split(",") if t.strip()]
    prompt_str = ". ".join(text_query)
    inputs = processor(images=image, text=prompt_str, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [image.size[::-1]]
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        box_threshold=threshold,
        text_threshold=threshold,
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = [results["labels"][i] for i in range(len(results["labels"]))]
    plot_detections(
        image, boxes, scores, labels, score_threshold=threshold, save_path=save_path
    )
    typer.echo(f"[INFO] Detect result saved: {save_path}")


if __name__ == "__main__":
    typer.run(main)
