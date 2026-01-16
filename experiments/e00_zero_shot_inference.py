"""
# Zero-Shot Object Detection with Hugging Face OWL-ViT

## 사전 준비
1. 필요 리소스 다운로드
    uv run scripts/azblob_lite.py download jgryu --container-path ml-workflow-object-detection-example/IDEA-Research/grounding-dino-base/ --dst-path artifacts/IDEA-Research/grounding-dino-base/

## 실험 실행
    uv run experiments/e01_pre_trained_model.py experiments/sample_data/ --output-dir data/detr_results --threshold 0.7
"""

from pathlib import Path

import torch
import typer
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from experiments.utils import load_image, plot_detections


def main(
    image_path: Path = typer.Argument(..., help="입력 이미지 경로"),
    prompt: str = typer.Option(..., help="탐지할 텍스트 (쉼표로 구분)"),
    model_name: str = typer.Option(
        "IDEA-Research/grounding-dino-base", help="huggingface 모델명 또는 폴더"
    ),
    threshold: float = typer.Option(0.3, help="바운딩박스 신뢰도 임계값"),
    save_path: Path = typer.Option("result.jpg", help="결과 이미지 파일"),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", help="cpu|cuda"
    ),
):
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
