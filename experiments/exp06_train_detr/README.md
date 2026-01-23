# Exp06: DETR Training

DETR 학습/평가 실험을 정리합니다. 데이터 분할은 `exp05_train_yolo`의 스크립트를 사용합니다.

## 1) 데이터 분할 (train/val/test)
```sh
uv run experiments/exp05_train_yolo/split_pcb_dataset.py \
  --base-dir data/PCB_DATASET \
  --output-dir data/pcb_splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

## 2) DETR 파인튜닝
```sh
uv run experiments/exp06_train_detr/train_detr.py \
  --data-dir data/pcb_splits \
  --output-dir artifacts/exp06/detr_finetune \
  --pretrained \
  --epochs 10 \
  --batch-size 4
```

### DETR Transfer Learning (Backbone Freeze)
```sh
uv run experiments/exp06_train_detr/train_detr.py \
  --data-dir data/pcb_splits \
  --output-dir artifacts/exp06/detr_transfer \
  --pretrained \
  --freeze-backbone \
  --epochs 10 \
  --batch-size 4
```

## 3) 처음부터 학습 (from scratch)
```sh
uv run experiments/exp06_train_detr/train_detr.py \
  --data-dir data/pcb_splits \
  --output-dir artifacts/exp06/detr_from_scratch \
  --no-pretrained \
  --epochs 20 \
  --batch-size 4
```

## 4) MLflow Tracking (Databricks)
```sh
export MLFLOW_TRACKING_URI=databricks
```

Databricks 환경변수(`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)는 `.env`에 설정된 값을 사용합니다.
