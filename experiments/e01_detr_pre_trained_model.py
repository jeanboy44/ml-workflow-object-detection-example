"""
# Zero-Shot Object Detection with Hugging Face OWL-ViT

## 사전 준비
1. 필요 리소스 다운로드
uv run scripts/download_files.py jgryu --blob-path ml-workflow-object-detection-example/facebook/detr-resnet-50/ --dst-path artifacts/facebook/detr-resnet-50/

## 실험 실행
    uv run experiments/e01_detr_pre_trained_model.py experiments/sample_data/ --output-dir data/detr_results --threshold 0.7

"""

from pathlib import Path

import torch
import typer
from transformers import DetrForObjectDetection, DetrImageProcessor

from experiments.utils import load_image, plot_detections

app = typer.Typer()


def run_detr(
    img_path: Path,
    save_path: Path = None,
    device: str = "cpu",
    score_threshold: float = 0.1,
):
    processor = DetrImageProcessor.from_pretrained("artifacts/facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "artifacts/facebook/detr-resnet-50",
        use_pretrained_backbone=False,  # https://github.com/huggingface/transformers/issues/15764
    )
    model.to(device)

    image = load_image(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (height, width)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=score_threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    id2label = model.config.id2label
    str_labels = [id2label[label] for label in labels]

    plot_detections(
        image,
        boxes,
        scores,
        str_labels,
        score_threshold=score_threshold,
        save_path=save_path,
    )


@app.command()
def main(
    input: Path = typer.Argument(..., help="Path to image or image folder"),
    output_dir: Path = typer.Option(
        None, help="Output dir for visualization (optional)"
    ),
    device: str = typer.Option("cpu", help="Device to run on (cpu or cuda)"),
    score_threshold: float = typer.Option(0.1, help="Score threshold for detections"),
):
    """Run facebook DETR on image or directory."""
    input_path = input
    output = output_dir
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    def accepted_ext(f: Path):
        return f.suffix.lower() in [".jpg", ".jpeg", ".png"]

    if input_path.is_dir():
        for img_file in filter(accepted_ext, input_path.iterdir()):
            save_path = output / f"result_{img_file.name}" if output else None
            print(f"[INFO] Running detection on {img_file}")
            run_detr(
                img_file,
                save_path=save_path,
                device=device,
                score_threshold=score_threshold,
            )
    else:
        save_path = output / f"result_{input_path.name}" if output else None
        run_detr(
            input_path,
            save_path=save_path,
            device=device,
            score_threshold=score_threshold,
        )


if __name__ == "__main__":
    app()
