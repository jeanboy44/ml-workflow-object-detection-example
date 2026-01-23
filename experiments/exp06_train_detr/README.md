# Exp06: DETR Training

PCB 결함 데이터셋을 대상으로 DETR 학습 파이프라인을 구성하고, 파인튜닝/스크래치 학습을 실험합니다. 데이터 분할은 `exp05_train_yolo`의 스크립트를 사용합니다.

## 사전 준비

### 1) 데이터 분할 (train/val/test)
```sh
uv run experiments/exp05_train_yolo/split_pcb_dataset.py --base-dir data/PCB_DATASET --output-dir data/pcb_splits
```

## 실험

### 1) DETR 파인튜닝
```sh
uv run experiments/exp06_train_detr/train_detr_hydra.py
```

### 2) Transfer Learning (Backbone Freeze)
```sh
uv run experiments/exp06_train_detr/train_detr_hydra.py train.freeze_backbone=true
```

### 3) From Scratch
```sh
uv run experiments/exp06_train_detr/train_detr_hydra.py train.pretrained=false train.epochs=20
```

### 4) 고해상도 학습 (더 많은 epoch)
```sh
uv run experiments/exp06_train_detr/train_detr_hydra.py train.epochs=20 train.batch_size=2 train.lr=5e-5
```

## MLflow Tracking (Databricks)
```sh
export MLFLOW_TRACKING_URI=databricks
```

Databricks 환경변수(`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)는 `.env`에 설정된 값을 사용합니다.
