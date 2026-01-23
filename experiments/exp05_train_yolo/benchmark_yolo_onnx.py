"""Benchmark YOLO (.pt) vs ONNX on a single image.

Example:
    uv run experiments/exp05/benchmark_yolo_onnx.py \
      --pt-model runs/exp05/yolo_finetune/weights/best.pt \
      --onnx-model artifacts/exp05/yolo_onnx/best.onnx \
      --image-path data/PCB_DATASET/images/Short/01_short_01.jpg
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import typer
from ultralytics import YOLO

app = typer.Typer()


@dataclass(frozen=True)
class BenchResult:
    avg_latency_ms: float
    avg_detections: float
    avg_score: float


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def run_inference(
    model: YOLO,
    image_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
) -> tuple[list[list[float]], list[int], list[float]]:
    results = model(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
    )[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    labels = results.boxes.cls.cpu().tolist()
    scores = results.boxes.conf.cpu().tolist()
    return boxes, [int(label) for label in labels], scores


def benchmark(
    model: YOLO,
    image_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    runs: int,
    warmup: int,
) -> tuple[BenchResult, list[list[float]], list[int], list[float]]:
    for _ in range(warmup):
        run_inference(model, image_path, conf, imgsz, device)

    timings = []
    detections_counts = []
    avg_scores = []
    last_boxes: list[list[float]] = []
    last_labels: list[int] = []
    last_scores: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        last_boxes, last_labels, last_scores = run_inference(
            model, image_path, conf, imgsz, device
        )
        timings.append((time.perf_counter() - start) * 1000.0)
        detections_counts.append(len(last_scores))
        avg_scores.append(
            sum(last_scores) / len(last_scores) if last_scores else 0.0
        )

    avg_latency = sum(timings) / max(len(timings), 1)
    avg_detections = sum(detections_counts) / max(len(detections_counts), 1)
    avg_score = sum(avg_scores) / max(len(avg_scores), 1)
    return (
        BenchResult(
            avg_latency_ms=avg_latency,
            avg_detections=avg_detections,
            avg_score=avg_score,
        ),
        last_boxes,
        last_labels,
        last_scores,
    )


def compare_predictions(
    pt_boxes: list[list[float]],
    pt_labels: list[int],
    onnx_boxes: list[list[float]],
    onnx_labels: list[int],
    iou_threshold: float,
) -> tuple[float, float]:
    matches = 0
    iou_sum = 0.0
    used = set()
    for box, label in zip(pt_boxes, pt_labels, strict=True):
        best_iou = 0.0
        best_idx = None
        for idx, (onnx_box, onnx_label) in enumerate(zip(onnx_boxes, onnx_labels)):
            if idx in used or label != onnx_label:
                continue
            iou = box_iou(box, onnx_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            used.add(best_idx)
            matches += 1
            iou_sum += best_iou

    match_rate = matches / len(pt_boxes) if pt_boxes else 0.0
    mean_iou = iou_sum / matches if matches else 0.0
    return match_rate, mean_iou


@app.command()
def main(
    pt_model: Path = typer.Option(..., help="YOLO .pt model path"),
    onnx_model: Path = typer.Option(..., help="YOLO .onnx model path"),
    image_path: Path = typer.Option(..., help="Input image path"),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
    imgsz: int = typer.Option(640, help="Inference image size"),
    device: str | None = typer.Option(None, help="Device (cpu/cuda/mps)"),
    runs: int = typer.Option(20, help="Benchmark runs"),
    warmup: int = typer.Option(3, help="Warmup runs"),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for match rate"),
):
    """Compare inference speed and output similarity for PT vs ONNX."""
    pt_model = pt_model.expanduser()
    onnx_model = onnx_model.expanduser()
    image_path = image_path.expanduser()

    if not pt_model.exists():
        raise FileNotFoundError(f"Missing pt model: {pt_model}")
    if not onnx_model.exists():
        raise FileNotFoundError(f"Missing onnx model: {onnx_model}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    pt = YOLO(str(pt_model))
    onnx = YOLO(str(onnx_model))

    pt_result, pt_boxes, pt_labels, pt_scores = benchmark(
        pt, image_path, conf, imgsz, device, runs, warmup
    )
    onnx_result, onnx_boxes, onnx_labels, onnx_scores = benchmark(
        onnx, image_path, conf, imgsz, device, runs, warmup
    )

    match_rate, mean_iou = compare_predictions(
        pt_boxes, pt_labels, onnx_boxes, onnx_labels, iou_threshold
    )

    print("[INFO] PT latency: {:.2f} ms".format(pt_result.avg_latency_ms))
    print("[INFO] ONNX latency: {:.2f} ms".format(onnx_result.avg_latency_ms))
    print("[INFO] PT avg detections: {:.2f}, avg score: {:.3f}".format(
        pt_result.avg_detections, pt_result.avg_score
    ))
    print("[INFO] ONNX avg detections: {:.2f}, avg score: {:.3f}".format(
        onnx_result.avg_detections, onnx_result.avg_score
    ))
    print("[INFO] Match rate: {:.2%}, mean IoU: {:.3f}".format(match_rate, mean_iou))


if __name__ == "__main__":
    app()
