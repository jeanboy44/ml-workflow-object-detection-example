# Exp05: PCB Object Detection Training

PCB 결함 데이터셋을 학습/평가 가능한 형태로 분할하고, DETR/YOLO를 각각 파인튜닝 및 처음부터 학습하는 실험을 정리합니다. 모든 학습 단계는 MLflow로 기록합니다.

## 1) 데이터 분할 (train/val/test)
```sh
uv run experiments/exp05/split_pcb_dataset.py \
  --base-dir data/PCB_DATASET \
  --output-dir data/pcb_splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

## 2) YOLO 학습용 데이터셋 변환
```sh
uv run experiments/exp05/prepare_yolo_dataset.py \
  --split-dir data/pcb_splits \
  --output-dir data/pcb_yolo
```

## 3) DETR 파인튜닝
```sh
uv run experiments/exp05/train_detr.py \
  --data-dir data/pcb_splits \
  --output-dir artifacts/exp05/detr_finetune \
  --pretrained \
  --epochs 10 \
  --batch-size 4
```

### DETR Transfer Learning (Backbone Freeze)
```sh
uv run experiments/exp05/train_detr.py \
  --data-dir data/pcb_splits \
  --output-dir artifacts/exp05/detr_transfer \
  --pretrained \
  --freeze-backbone \
  --epochs 10 \
  --batch-size 4
```

## 4) YOLO 파인튜닝
```sh
uv run experiments/exp05/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### YOLO Transfer Learning (Freeze Backbone)
```sh
uv run experiments/exp05/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --freeze-backbone 10 \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

## 5) 처음부터 학습 (from scratch)

### DETR
```sh
uv run experiments/exp05/train_detr.py \
  --data-dir data/pcb_splits \
  --output-dir artifacts/exp05/detr_from_scratch \
  --no-pretrained \
  --epochs 20 \
  --batch-size 4
```

### YOLO
```sh
uv run experiments/exp05/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --from-scratch \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

## 6) MLflow Tracking (Databricks)
```sh
export MLFLOW_TRACKING_URI=databricks
```

Databricks 환경변수(`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)는 `.env`에 설정된 값을 사용합니다.
