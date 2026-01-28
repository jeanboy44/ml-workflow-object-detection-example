import sys
from pathlib import Path

import torch
import typer
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from utils import load_image, plot_detections


def main(
    image_path: Path = typer.Argument(..., help="입력 이미지 경로"),
    prompt: str = typer.Argument(..., help="탐지할 텍스트 (ex: cat,dog)"),
    threshold: float = typer.Option(0.1, "--threshold", "-t", help="신뢰도 임계값"),
    model_path: Path = typer.Option(
        Path("../../artifacts/IDEA-Research/grounding-dino-base"),
        "--model-path",
        "-m",
        help="사전학습된 모델",
    ),
    save_path: Path = typer.Option(None, "--save-path", "-o", help="결과 이미지 경로"),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", "--device", help="cpu/cuda 선택"
    ),
):
    if not image_path.exists():
        typer.echo(f"[ERROR] 이미지 파일 없음: {image_path}")
        sys.exit(1)

    processor = GroundingDinoProcessor.from_pretrained(model_path)
    model = GroundingDinoForObjectDetection.from_pretrained(model_path).to(device)
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
    plot_detections(
        image, boxes, scores, labels, score_threshold=threshold, save_path=save_path
    )
    typer.echo(f"[INFO] Detect result saved: {save_path}")


if __name__ == "__main__":
    typer.run(main)
