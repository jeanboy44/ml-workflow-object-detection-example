# Exp05: YOLO Training

PCB 결함 데이터셋을 대상으로 YOLO 학습 파이프라인을 구성하고, 파인튜닝/스크래치 학습/ONNX 변환/벤치마크까지 전 과정을 실험합니다.

## 사전 준비

### 1) 데이터 분할 (train/val/test)
```sh
uv run experiments/exp05_train_yolo/split_pcb_dataset.py \
  --base-dir data/PCB_DATASET \
  --output-dir data/pcb_splits
```

### 2) YOLO 데이터셋 변환
```sh
uv run experiments/exp05_train_yolo/prepare_yolo_dataset.py \
  --split-dir data/pcb_splits \
  --output-dir data/pcb_yolo
```

## 실험

### 0) Fast run
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 5 \
  --imgsz 320 \
  --batch 2 \
  --fraction 0.01
```

### 1) Transfer Learning (Freeze Backbone)
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --freeze-backbone 10 \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### 2) From Scratch
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --from-scratch \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

### 3) 작은 박스 대응 (고해상도)
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 10 \
  --imgsz 960 \
  --batch 8
```

```sh
uv run experiments/e03_eda_pcb_data.py
```

### 5) Hydra 기반 학습 (Mosaic 증강 조절)
기본 실행 (default_config.yaml 사용):
Fast test:
```sh
uv run experiments/exp05_train_yolo/train_yolo_hydra.py --config-name fasttest_config
```

```sh
uv run experiments/exp05_train_yolo/train_yolo_hydra.py
```

Mosaic 증강:
```sh
uv run experiments/exp05_train_yolo/train_yolo_hydra.py train.imgsz=1280 train.batch=16 train.freeze=10 train.mosaic=1.0 train.epochs=10
```

## 배포 준비

### 1) YOLO ONNX Export
```sh
uv run experiments/exp05_train_yolo/export_yolo_onnx.py \
  --model-path runs/exp05/yolo_finetune/weights/best.pt \
  --output-dir artifacts/exp05/yolo_onnx
```

### 2) PT vs ONNX Benchmark
```sh
uv run experiments/exp05_train_yolo/benchmark_yolo_onnx.py \
  --pt-model runs/exp05/yolo_finetune/weights/best.pt \
  --onnx-model artifacts/exp05/yolo_onnx/best.onnx \
  --image-path data/PCB_DATASET/images/Short/01_short_01.jpg
```

## 참고 문서
- `experiments/exp05_train_yolo/yolo_experiment.md`
