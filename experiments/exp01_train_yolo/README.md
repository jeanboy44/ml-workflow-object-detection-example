# Exp01: YOLO Training (Typer)

PCB 결함 데이터셋을 YOLO로 학습하는 기본 CLI 파이프라인입니다. 데이터 분할, YOLO 포맷 변환, 학습과 MLflow 로깅까지 한 번에 수행합니다.

## Workflow

### 1) 데이터 분할 (train/val/test)
```sh
uv run experiments/exp01_train_yolo/split_pcb_dataset.py \
  --base-dir data/PCB_DATASET \
  --output-dir data/pcb_splits
```

### 2) YOLO 데이터셋 변환
```sh
uv run experiments/exp01_train_yolo/prepare_yolo_dataset.py \
  --split-dir data/pcb_splits \
  --output-dir data/pcb_yolo
```

### 3) 파인튜닝 학습
```sh
uv run experiments/exp01_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### 4) 스크래치 학습
```sh
uv run experiments/exp01_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --from-scratch \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

## 참고

- 기본 저장 경로는 `runs/detect/<project>/<name>/weights/best.pt` 입니다.
- MLflow 실험/런은 `train_yolo.py` 옵션(`--experiment-name`, `--run-name`)으로 설정합니다.
- 기본 `project`가 `runs/exp05`로 되어 있으니, 필요 시 `--project`로 변경하세요.
