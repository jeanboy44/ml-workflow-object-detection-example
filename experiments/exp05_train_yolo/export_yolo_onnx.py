"""Export a YOLO model to ONNX.

Example:
    uv run experiments/exp05/export_yolo_onnx.py \
      --model-path runs/exp05/yolo_finetune/weights/best.pt \
      --model-name exp05_yolo_onnx
"""

from __future__ import annotations

import shutil
from pathlib import Path

import onnx
import typer
from ultralytics import YOLO

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(..., help="YOLO weights (.pt) path"),
    output_dir: Path = typer.Option(
        "artifacts/exp05/yolo_onnx", help="Output directory for ONNX export"
    ),
    imgsz: int = typer.Option(640, help="Export image size"),
    device: str | None = typer.Option(None, help="Export device"),
):
    """Export YOLO weights to ONNX and validate the file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO(str(model_path))
    export_path = yolo_model.export(format="onnx", imgsz=imgsz, device=device)
    export_path = Path(export_path)
    onnx_path = output_dir / export_path.name
    if export_path.resolve() != onnx_path.resolve():
        shutil.copy2(export_path, onnx_path)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph
    print(f"[INFO] Exported ONNX: {onnx_path}")
    print(
        "[INFO] Graph inputs={} outputs={} nodes={}".format(
            len(graph.input), len(graph.output), len(graph.node)
        )
    )


if __name__ == "__main__":
    app()
