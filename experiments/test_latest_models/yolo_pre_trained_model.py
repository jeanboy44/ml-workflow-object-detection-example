from pathlib import Path

import typer
from ultralytics import YOLO

from utils import load_image, plot_detections

app = typer.Typer()


def run_yolo(
    img_path: Path,
    save_path: Path = None,
    model_path: str = "../../artifacts/yolo/yolo26n.pt",
):
    assert img_path.exists(), f"[ERROR] 입력 이미지가 존재하지 않습니다: {img_path}"
    image = load_image(img_path)
    model = YOLO(model_path)
    results = model(str(img_path))
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = [model.names[int(i)] for i in result.boxes.cls.cpu().numpy()]
    plot_detections(image, boxes, scores, labels, save_path=save_path)


@app.command()
def main(
    input: Path = typer.Argument(..., help="입력 이미지/폴더 경로"),
    output_dir: Path = typer.Option(None, help="결과 저장 폴더(없으면 show만)"),
    model_path: str = typer.Option(
        "../../artifacts/yolo/yolo26n.pt", help="YOLO 모델 가중치 파일"
    ),
):
    """YOLO 사전학습모델로 객체 탐지를 실행합니다."""
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
            run_yolo(img_file, save_path=save_path, model_path=model_path)
    else:
        save_path = output / f"result_{input_path.name}" if output else None
        run_yolo(input_path, save_path=save_path, model_path=model_path)


if __name__ == "__main__":
    app()
