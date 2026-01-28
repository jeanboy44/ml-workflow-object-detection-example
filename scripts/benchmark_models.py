from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import typer
from ultralytics import YOLO

app = typer.Typer(help="PT/ONNX 모델 성능 및 예측 비교 벤치마크")


@dataclass(frozen=True)
class BenchmarkResult:
    model_load_seconds: float
    avg_infer_ms_per_image: float
    total_images: int


@dataclass(frozen=True)
class PredictionDiffResult:
    images: int
    total_pt_detections: int
    total_onnx_detections: int
    match_rate: float
    mean_iou: float
    within_eps_rate: float
    max_coord_diff: float
    max_score_diff: float


def _list_images(sample_dir: Path, max_images: int | None) -> list[Path]:
    image_paths = sorted(
        [
            path
            for path in sample_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return image_paths


def _load_model(model_path: Path, device: str | None) -> tuple[YOLO, float]:
    start = perf_counter()
    model = YOLO(str(model_path))
    if device is not None:
        model.to(device)
    elapsed = perf_counter() - start
    return model, elapsed


def _run_inference(
    model: YOLO,
    image_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    normalized: bool,
) -> tuple[list[list[float]], list[int], list[float]]:
    results = model(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    boxes_tensor = results.boxes.xyxyn if normalized else results.boxes.xyxy
    boxes = boxes_tensor.cpu().tolist()
    labels = results.boxes.cls.cpu().tolist()
    scores = results.boxes.conf.cpu().tolist()
    return boxes, [int(label) for label in labels], scores


def _measure_inference_time(
    model: YOLO,
    image_paths: list[Path],
    conf: float,
    imgsz: int,
    device: str | None,
    warmup: int,
    runs: int,
) -> float:
    if not image_paths:
        return 0.0

    warmup_image = image_paths[0]
    for _ in range(max(0, warmup)):
        model(
            str(warmup_image),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

    start = perf_counter()
    for _ in range(max(1, runs)):
        for image_path in image_paths:
            model(
                str(image_path),
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
    elapsed = perf_counter() - start
    total_images = max(1, runs) * len(image_paths)
    return (elapsed / total_images) * 1000.0


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
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


def _match_predictions(
    pt_boxes: list[list[float]],
    pt_labels: list[int],
    onnx_boxes: list[list[float]],
    onnx_labels: list[int],
    iou_threshold: float,
) -> list[tuple[int, int | None, float]]:
    matches: list[tuple[int, int | None, float]] = []
    used = set()
    for pt_idx, (pt_box, pt_label) in enumerate(zip(pt_boxes, pt_labels, strict=True)):
        best_iou = 0.0
        best_idx: int | None = None
        for onnx_idx, (onnx_box, onnx_label) in enumerate(zip(onnx_boxes, onnx_labels)):
            if onnx_idx in used or pt_label != onnx_label:
                continue
            iou = _box_iou(pt_box, onnx_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = onnx_idx
        if best_idx is not None and best_iou >= iou_threshold:
            used.add(best_idx)
            matches.append((pt_idx, best_idx, best_iou))
        else:
            matches.append((pt_idx, None, best_iou))
    return matches


def _compare_predictions(
    pt_model: YOLO,
    onnx_model: YOLO,
    image_paths: list[Path],
    conf: float,
    imgsz: int,
    device: str | None,
    iou_threshold: float,
    epsilon: float,
    normalized: bool,
) -> PredictionDiffResult:
    total_pt = 0
    total_onnx = 0
    total_matches = 0
    total_iou = 0.0
    within_eps = 0
    max_coord_diff = 0.0
    max_score_diff = 0.0

    for image_path in image_paths:
        pt_boxes, pt_labels, pt_scores = _run_inference(
            pt_model, image_path, conf, imgsz, device, normalized
        )
        onnx_boxes, onnx_labels, onnx_scores = _run_inference(
            onnx_model, image_path, conf, imgsz, device, normalized
        )

        total_pt += len(pt_scores)
        total_onnx += len(onnx_scores)

        matches = _match_predictions(
            pt_boxes, pt_labels, onnx_boxes, onnx_labels, iou_threshold
        )
        for pt_idx, onnx_idx, iou in matches:
            if onnx_idx is None:
                continue
            total_matches += 1
            total_iou += iou
            pt_box = pt_boxes[pt_idx]
            onnx_box = onnx_boxes[onnx_idx]
            pt_score = pt_scores[pt_idx]
            onnx_score = onnx_scores[onnx_idx]
            diffs = [
                abs(pt_box[0] - onnx_box[0]),
                abs(pt_box[1] - onnx_box[1]),
                abs(pt_box[2] - onnx_box[2]),
                abs(pt_box[3] - onnx_box[3]),
            ]
            max_coord_diff = max(max_coord_diff, max(diffs, default=0.0))
            score_diff = abs(pt_score - onnx_score)
            max_score_diff = max(max_score_diff, score_diff)
            if max(diffs, default=0.0) <= epsilon and score_diff <= epsilon:
                within_eps += 1

    match_rate = total_matches / total_pt if total_pt else 0.0
    mean_iou = total_iou / total_matches if total_matches else 0.0
    within_eps_rate = within_eps / total_matches if total_matches else 0.0
    return PredictionDiffResult(
        images=len(image_paths),
        total_pt_detections=total_pt,
        total_onnx_detections=total_onnx,
        match_rate=match_rate,
        mean_iou=mean_iou,
        within_eps_rate=within_eps_rate,
        max_coord_diff=max_coord_diff,
        max_score_diff=max_score_diff,
    )


@app.command()
def main(
    pt_model: Path = typer.Argument(..., help="YOLO .pt 모델 경로"),
    onnx_model: Path = typer.Argument(..., help="ONNX 모델 경로"),
    image_dir: Path = typer.Argument(..., help="이미지 데이터 디렉토리"),
    imgsz: int = typer.Option(640, help="추론 이미지 사이즈"),
    conf: float = typer.Option(0.25, help="예측 confidence threshold"),
    iou_threshold: float = typer.Option(0.5, help="예측 비교 IoU threshold"),
    epsilon: float = typer.Option(0.01, help="좌표/점수 허용 오차"),
    normalized: bool = typer.Option(True, help="좌표를 0~1로 정규화해서 비교"),
    warmup: int = typer.Option(1, help="모델 워밍업 횟수"),
    runs: int = typer.Option(3, help="추론 반복 횟수"),
    max_images: int | None = typer.Option(
        None, help="벤치마크에 사용할 최대 이미지 수"
    ),
    device: str | None = typer.Option(None, help="device (cpu/cuda/mps)"),
):
    """모델 로드/추론/예측 결과 차이를 벤치마크합니다."""
    pt_model = pt_model.expanduser()
    onnx_model = onnx_model.expanduser()
    image_dir = image_dir.expanduser()

    if not pt_model.exists():
        raise typer.BadParameter(f"모델 파일을 찾을 수 없습니다: {pt_model}")
    if not onnx_model.exists():
        raise typer.BadParameter(f"모델 파일을 찾을 수 없습니다: {onnx_model}")
    if not image_dir.exists():
        raise typer.BadParameter(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir}")

    image_paths = _list_images(image_dir, max_images)
    if not image_paths:
        raise typer.BadParameter(f"이미지 파일이 없습니다: {image_dir}")

    pt, pt_load = _load_model(pt_model, device)
    onnx, onnx_load = _load_model(onnx_model, device)

    pt_avg_ms = _measure_inference_time(
        pt, image_paths, conf, imgsz, device, warmup, runs
    )
    onnx_avg_ms = _measure_inference_time(
        onnx, image_paths, conf, imgsz, device, warmup, runs
    )

    diff = _compare_predictions(
        pt,
        onnx,
        image_paths,
        conf,
        imgsz,
        device,
        iou_threshold,
        epsilon,
        normalized,
    )

    typer.secho("모델 로드 시간", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"PT:   {pt_load:.4f}s")
    typer.echo(f"ONNX: {onnx_load:.4f}s")

    typer.secho("추론 평균 시간 (ms/image)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"PT:   {pt_avg_ms:.2f}ms")
    typer.echo(f"ONNX: {onnx_avg_ms:.2f}ms")

    typer.secho("예측 차이 요약", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        "images={} pt_det={} onnx_det={} match_rate={:.2%} mean_iou={:.3f} "
        "within_eps={:.2%} max_coord_diff={:.4f} max_score_diff={:.4f}".format(
            diff.images,
            diff.total_pt_detections,
            diff.total_onnx_detections,
            diff.match_rate,
            diff.mean_iou,
            diff.within_eps_rate,
            diff.max_coord_diff,
            diff.max_score_diff,
        )
    )


if __name__ == "__main__":
    app()
