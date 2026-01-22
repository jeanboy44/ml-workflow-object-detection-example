"""
# Zero-Shot Object Detection with Grounding DINO

## 사전 준비
1. 필요 리소스 다운로드
uv run scripts/download_files.py jgryu --blob-path ml-workflow-object-detection-example/IDEA-Research/grounding-dino-base/ --dst-path artifacts/IDEA-Research/grounding-dino-base/

## 실험 실행
    uv run experiments/e00_grounddino_zero_shot_inference.py experiments/sample_data/cat_03.jpg "cat" --threshold 0.2 --save-path data/output_ground_dino/cat_03.jpg
"""

import sys
from pathlib import Path

import torch
import typer
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from experiments.utils import load_image, plot_detections

MODEL_NAME = "artifacts/IDEA-Research/grounding-dino-base"


def main(
    image_path: Path = typer.Argument(..., help="입력 이미지 경로"),
    prompt: str = typer.Argument(..., help="탐지할 텍스트 (ex: cat,dog)"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="신뢰도 임계값"),
    save_path: Path = typer.Option(
        "result.jpg", "--save-path", "-o", help="결과 이미지 경로"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", "--device", help="cpu/cuda 선택"
    ),
):
    image_path = Path(image_path)
    save_path = Path(save_path)
    if not image_path.exists():
        typer.echo(f"[ERROR] 이미지 파일 없음: {image_path}")
        sys.exit(1)

    processor = GroundingDinoProcessor.from_pretrained(MODEL_NAME)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_NAME).to(device)
    image = load_image(image_path)
    text_query = [t.strip() for t in prompt.split(",") if t.strip()]
    prompt_str = ". ".join(text_query)
    inputs = processor(images=image, text=prompt_str, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1]]
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=target_sizes,
    )
    if isinstance(results, list):
        results = results[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_detections(
        image, boxes, scores, labels, score_threshold=threshold, save_path=save_path
    )
    typer.echo(f"[INFO] Detect result saved: {save_path}")


if __name__ == "__main__":
    typer.run(main)
