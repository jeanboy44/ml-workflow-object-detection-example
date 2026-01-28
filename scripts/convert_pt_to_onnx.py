from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
import typer
from ultralytics import YOLO  # type: ignore

app = typer.Typer(help="YOLO .pt 모델을 ONNX로 변환하고 검증합니다.")


@dataclass(frozen=True)
class PredictionValidationResult:
    images: int
    avg_match_rate: float
    avg_abs_dx1: float
    avg_abs_dy1: float
    avg_abs_dx2: float
    avg_abs_dy2: float
    avg_abs_dscore: float


def convert_to_onnx(
    pt_model: Path,
    output_dir: Path,
    imgsz: int,
    device: str | None,
    opset: int,
    simplify: bool,
    dynamic: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO(str(pt_model))
    export_path = yolo_model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )
    export_path = Path(export_path)
    onnx_path = output_dir / export_path.name
    if export_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(export_path.read_bytes())
    return onnx_path


def validate_onnx(onnx_path: Path) -> tuple[int, int, int]:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph
    return len(graph.input), len(graph.output), len(graph.node)


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


def _run_inference(
    model: YOLO,
    image_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    normalized: bool = False,
) -> tuple[list[list[float]], list[int], list[float]]:
    results = model(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
    )[0]
    boxes_tensor = results.boxes.xyxyn if normalized else results.boxes.xyxy
    boxes = boxes_tensor.cpu().tolist()
    labels = results.boxes.cls.cpu().tolist()
    scores = results.boxes.conf.cpu().tolist()
    return boxes, [int(label) for label in labels], scores


def _compare_predictions_by_delta(
    pt_boxes: list[list[float]],
    pt_labels: list[int],
    pt_scores: list[float],
    onnx_boxes: list[list[float]],
    onnx_labels: list[int],
    onnx_scores: list[float],
) -> tuple[float, dict[str, float], int]:
    matches = 0
    used = set()
    diffs = {
        "dx1": 0.0,
        "dy1": 0.0,
        "dx2": 0.0,
        "dy2": 0.0,
        "dscore": 0.0,
    }

    for pt_box, pt_label, pt_score in zip(pt_boxes, pt_labels, pt_scores, strict=True):
        best_delta = float("inf")
        best_idx = None
        for idx, (onnx_box, onnx_label) in enumerate(zip(onnx_boxes, onnx_labels)):
            if idx in used or pt_label != onnx_label:
                continue
            delta = (
                abs(pt_box[0] - onnx_box[0])
                + abs(pt_box[1] - onnx_box[1])
                + abs(pt_box[2] - onnx_box[2])
                + abs(pt_box[3] - onnx_box[3])
            )
            if delta < best_delta:
                best_delta = delta
                best_idx = idx

        if best_idx is None:
            continue

        used.add(best_idx)
        matches += 1
        onnx_box = onnx_boxes[best_idx]
        onnx_score = onnx_scores[best_idx] if best_idx < len(onnx_scores) else 0.0
        diffs["dx1"] += abs(pt_box[0] - onnx_box[0])
        diffs["dy1"] += abs(pt_box[1] - onnx_box[1])
        diffs["dx2"] += abs(pt_box[2] - onnx_box[2])
        diffs["dy2"] += abs(pt_box[3] - onnx_box[3])
        diffs["dscore"] += abs(pt_score - onnx_score)

    match_rate = matches / len(pt_boxes) if pt_boxes else 0.0
    return match_rate, diffs, matches


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


def _format_row(values: list[str], widths: list[int]) -> str:
    return " | ".join(value.ljust(width) for value, width in zip(values, widths))


def _float_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _compare_tables(
    pt_model: Path,
    onnx_model: Path,
    sample_dir: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    iou_threshold: float,
    epsilon: float,
    normalized: bool,
    max_images: int | None,
) -> str:
    if not sample_dir.exists():
        raise typer.BadParameter(
            f"샘플 데이터 디렉토리를 찾을 수 없습니다: {sample_dir}"
        )

    image_paths = sorted(
        [
            path
            for path in sample_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not image_paths:
        raise typer.BadParameter(f"샘플 이미지가 없습니다: {sample_dir}")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    pt = YOLO(str(pt_model))
    onnx = YOLO(str(onnx_model))

    headers = [
        "image",
        "pt_idx",
        "label",
        "pt_x1",
        "pt_y1",
        "pt_x2",
        "pt_y2",
        "pt_score",
        "onnx_x1",
        "onnx_y1",
        "onnx_x2",
        "onnx_y2",
        "onnx_score",
        "iou",
        "within_eps",
    ]

    rows: list[list[str]] = []
    for image_path in image_paths:
        pt_boxes, pt_labels, pt_scores = _run_inference(
            pt, image_path, conf, imgsz, device, normalized
        )
        onnx_boxes, onnx_labels, onnx_scores = _run_inference(
            onnx, image_path, conf, imgsz, device, normalized
        )
        matches = _match_predictions(
            pt_boxes, pt_labels, onnx_boxes, onnx_labels, iou_threshold
        )

        for pt_idx, onnx_idx, iou in matches:
            pt_box = pt_boxes[pt_idx]
            pt_label = pt_labels[pt_idx]
            pt_score = pt_scores[pt_idx]

            onnx_box = None
            onnx_score: float | None = None
            if onnx_idx is not None:
                onnx_box = onnx_boxes[onnx_idx]
                onnx_score = onnx_scores[onnx_idx]

            within_eps = ""
            if onnx_box is not None:
                onnx_score_val = onnx_score if onnx_score is not None else 0.0
                diffs = [
                    abs(pt_box[0] - onnx_box[0]),
                    abs(pt_box[1] - onnx_box[1]),
                    abs(pt_box[2] - onnx_box[2]),
                    abs(pt_box[3] - onnx_box[3]),
                    abs(pt_score - onnx_score_val),
                ]
                within_eps = (
                    "true" if all(diff <= epsilon for diff in diffs) else "false"
                )

            row = [
                image_path.name,
                str(pt_idx),
                str(pt_label),
                _float_or_blank(pt_box[0]),
                _float_or_blank(pt_box[1]),
                _float_or_blank(pt_box[2]),
                _float_or_blank(pt_box[3]),
                _float_or_blank(pt_score),
                _float_or_blank(onnx_box[0] if onnx_box else None),
                _float_or_blank(onnx_box[1] if onnx_box else None),
                _float_or_blank(onnx_box[2] if onnx_box else None),
                _float_or_blank(onnx_box[3] if onnx_box else None),
                _float_or_blank(onnx_score),
                f"{iou:.3f}",
                within_eps,
            ]
            rows.append(row)

    widths = [max(len(header), 10) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    lines = [
        _format_row(headers, widths),
        _format_row(["-" * width for width in widths], widths),
    ]
    for row in rows:
        lines.append(_format_row(row, widths))
    return "\n".join(lines)


def validate_predictions(
    pt_model: Path,
    onnx_model: Path,
    sample_dir: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    max_images: int | None,
) -> PredictionValidationResult:
    if not sample_dir.exists():
        raise typer.BadParameter(
            f"샘플 데이터 디렉토리를 찾을 수 없습니다: {sample_dir}"
        )

    image_paths = sorted(
        [
            path
            for path in sample_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not image_paths:
        raise typer.BadParameter(f"샘플 이미지가 없습니다: {sample_dir}")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    pt = YOLO(str(pt_model))
    onnx = YOLO(str(onnx_model))

    total_match_rate = 0.0
    total_dx1 = 0.0
    total_dy1 = 0.0
    total_dx2 = 0.0
    total_dy2 = 0.0
    total_dscore = 0.0
    total_matches = 0
    effective_conf = min(conf, 0.001)

    for image_path in image_paths:
        pt_boxes, pt_labels, pt_scores = _run_inference(
            pt, image_path, effective_conf, imgsz, device
        )
        onnx_boxes, onnx_labels, onnx_scores = _run_inference(
            onnx, image_path, effective_conf, imgsz, device
        )
        match_rate, diffs, matches = _compare_predictions_by_delta(
            pt_boxes,
            pt_labels,
            pt_scores,
            onnx_boxes,
            onnx_labels,
            onnx_scores,
        )
        total_match_rate += match_rate
        total_dx1 += diffs["dx1"]
        total_dy1 += diffs["dy1"]
        total_dx2 += diffs["dx2"]
        total_dy2 += diffs["dy2"]
        total_dscore += diffs["dscore"]
        total_matches += matches

    images = len(image_paths)
    avg_divisor = total_matches if total_matches else 1
    return PredictionValidationResult(
        images=images,
        avg_match_rate=total_match_rate / images,
        avg_abs_dx1=total_dx1 / avg_divisor,
        avg_abs_dy1=total_dy1 / avg_divisor,
        avg_abs_dx2=total_dx2 / avg_divisor,
        avg_abs_dy2=total_dy2 / avg_divisor,
        avg_abs_dscore=total_dscore / avg_divisor,
    )


@app.command()
def main(
    pt_model: Path = typer.Argument(..., help="YOLO .pt 모델 경로"),
    output_dir: Path = typer.Option(Path("artifacts/onnx"), help="ONNX 저장 디렉토리"),
    sample_dir: Path = typer.Option(
        Path("data/sample_data"), help="예측 검증 샘플 이미지 디렉토리"
    ),
    imgsz: int = typer.Option(640, help="변환/추론 이미지 사이즈"),
    conf: float = typer.Option(0.25, help="예측 confidence threshold"),
    iou_threshold: float = typer.Option(0.5, help="예측 비교 IoU threshold"),
    device: str | None = typer.Option(None, help="device (cpu/cuda/mps)"),
    opset: int = typer.Option(12, help="ONNX opset 버전"),
    simplify: bool = typer.Option(True, help="ONNX 그래프 단순화"),
    dynamic: bool = typer.Option(True, help="동적 입력 크기 허용"),
    max_images: int | None = typer.Option(
        None, help="예측 검증에 사용할 최대 이미지 수"
    ),
):
    """PT 모델을 ONNX로 변환하고 유효성/예측 결과를 검증합니다."""
    pt_model = pt_model.expanduser()
    output_dir = output_dir.expanduser()
    sample_dir = sample_dir.expanduser()

    if not pt_model.exists():
        raise typer.BadParameter(f"모델 파일을 찾을 수 없습니다: {pt_model}")

    onnx_path = convert_to_onnx(
        pt_model,
        output_dir,
        imgsz,
        device,
        opset,
        simplify,
        dynamic,
    )
    typer.secho(f"ONNX 변환 완료: {onnx_path}", fg=typer.colors.GREEN)

    inputs, outputs, nodes = validate_onnx(onnx_path)
    typer.secho(
        f"ONNX 유효성 검증 완료: inputs={inputs}, outputs={outputs}, nodes={nodes}",
        fg=typer.colors.GREEN,
    )

    stats = validate_predictions(
        pt_model,
        onnx_path,
        sample_dir,
        conf,
        imgsz,
        device,
        max_images,
    )
    typer.secho(
        "예측 결과 검증 완료 (PT vs ONNX)\n"
        "- 비교 기준: 동일 클래스 박스끼리 좌표/점수 절대 차이\n"
        "- match_rate: PT 박스 중 매칭된 비율\n"
        "- confidence: 비교용 예측 임계값({conf:.3f})\n"
        "결과: images={images} match_rate={match_rate:.2%} "
        "dx1={dx1:.3f} dy1={dy1:.3f} dx2={dx2:.3f} dy2={dy2:.3f} dscore={dscore:.4f}".format(
            conf=min(conf, 0.001),
            images=stats.images,
            match_rate=stats.avg_match_rate,
            dx1=stats.avg_abs_dx1,
            dy1=stats.avg_abs_dy1,
            dx2=stats.avg_abs_dx2,
            dy2=stats.avg_abs_dy2,
            dscore=stats.avg_abs_dscore,
        ),
        fg=typer.colors.GREEN,
    )


@app.command()
def compare(
    pt_model: Path = typer.Argument(..., help="YOLO .pt 모델 경로"),
    onnx_model: Path = typer.Argument(..., help="ONNX 모델 경로"),
    sample_dir: Path = typer.Option(
        Path("data/sample_data"), help="예측 검증 샘플 이미지 디렉토리"
    ),
    imgsz: int = typer.Option(640, help="추론 이미지 사이즈"),
    conf: float = typer.Option(0.25, help="예측 confidence threshold"),
    iou_threshold: float = typer.Option(0.5, help="매칭 IoU threshold"),
    epsilon: float = typer.Option(1.0, help="좌표/점수 허용 오차"),
    normalized: bool = typer.Option(True, help="좌표를 0~1로 정규화해서 비교"),
    device: str | None = typer.Option(None, help="device (cpu/cuda/mps)"),
    max_images: int | None = typer.Option(
        None, help="예측 검증에 사용할 최대 이미지 수"
    ),
):
    """PT/ONNX 결과를 테이블로 비교합니다."""
    pt_model = pt_model.expanduser()
    onnx_model = onnx_model.expanduser()
    sample_dir = sample_dir.expanduser()

    if not pt_model.exists():
        raise typer.BadParameter(f"모델 파일을 찾을 수 없습니다: {pt_model}")
    if not onnx_model.exists():
        raise typer.BadParameter(f"ONNX 파일을 찾을 수 없습니다: {onnx_model}")

    table = _compare_tables(
        pt_model,
        onnx_model,
        sample_dir,
        conf,
        imgsz,
        device,
        iou_threshold,
        epsilon,
        normalized,
        max_images,
    )
    typer.echo(table)


if __name__ == "__main__":
    app()
