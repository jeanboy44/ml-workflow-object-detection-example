# Zero-Shot Object Detection with Hugging Face OWL-ViT
"""
실행 예시:
    uv run experiments/e01_zero_shot_inference.py --image-path experiments/cat_01.jpg --phrases "a dog"
    # 결과 저장도 가능
    uv run experiments/e01_zero_shot_inference.py --image-path path/to/image.jpg --phrases "a dog" "a person" --save-path result.jpg
"""

from pathlib import Path

import torch
import typer
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from experiments.utils import load_image, plot_detections


def zero_shot_detection(
    img_path: Path,
    texts: list[str],
    model_name: str = "google/owlvit-base-patch32",
    threshold: float = 0.1,
    device: str = "cpu",
    save_path: Path = None,
):
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    model.to(device)
    image = load_image(img_path)
    inputs = processor(text=texts, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_size = torch.tensor(image.size[::-1]).unsqueeze(0)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_size
    )
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = [texts[label] for label in results[0]["labels"].cpu().numpy()]
    plot_detections(
        image, boxes, scores, labels, score_threshold=threshold, save_path=save_path
    )
    return boxes, scores, labels


def main(
    image_path: Path = typer.Option(..., help="입력 이미지 경로"),
    phrases: list[str] = typer.Option(..., help="탐지할 텍스트 프롬프트 목록"),
    threshold: float = typer.Option(0.2, help="신뢰도 임계치 (0~1)"),
    save_path: Path = typer.Option(
        None, help="탐지결과를 저장할 경로 (지정 시 show하지 않음)"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="실행 디바이스 (cpu 또는 cuda)",
    ),
):
    """Zero-shot Object Detection (Open-Vocabulary) 실험 예시."""
    zero_shot_detection(
        img_path=image_path,
        texts=phrases,
        threshold=threshold,
        save_path=save_path,
        device=device,
    )


if __name__ == "__main__":
    typer.run(main)
