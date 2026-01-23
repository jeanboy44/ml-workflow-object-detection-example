# Exp05: YOLO Training

PCB 결함 데이터셋을 대상으로 YOLO 학습 파이프라인을 구성하고, 파인튜닝/스크래치 학습/ONNX 변환/벤치마크까지 전 과정을 실험합니다. 데이터 분할과 YOLO 포맷 변환부터 시작해, 학습 설정별 성능을 비교하고 ONNX 결과의 품질과 성능을 확인합니다.

## 1) 데이터 분할 (train/val/test)
```sh
uv run experiments/exp05_train_yolo/split_pcb_dataset.py \
  --base-dir data/PCB_DATASET \
  --output-dir data/pcb_splits
```

## 2) YOLO 데이터셋 변환
```sh
uv run experiments/exp05_train_yolo/prepare_yolo_dataset.py \
  --split-dir data/pcb_splits \
  --output-dir data/pcb_yolo
```

## 3) 파인튜닝
사전학습 가중치로 베이스라인을 확보합니다.
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### Transfer Learning (Freeze Backbone)
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --freeze-backbone 10 \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

## 4) From Scratch
```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --from-scratch \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

## 5) 작은 박스 대응 실험
`imgsz`를 올리고 데이터 분포를 점검해 작은 객체 인식 성능을 개선합니다.

```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 10 \
  --imgsz 960 \
  --batch 16
```

```sh
uv run experiments/exp05_train_yolo/train_yolo.py \
  --data-yaml data/pcb_yolo/data.yaml \
  --model artifacts/yolo/yolo26n.pt \
  --epochs 10 \
  --imgsz 1280 \
  --batch 16
```

```sh
uv run experiments/e03_eda_pcb_data.py
```

## 6) YOLO ONNX Export
```sh
uv run experiments/exp05_train_yolo/export_yolo_onnx.py \
  --model-path runs/exp05/yolo_finetune/weights/best.pt \
  --output-dir artifacts/exp05/yolo_onnx
```

## 7) PT vs ONNX Benchmark
```sh
uv run experiments/exp05_train_yolo/benchmark_yolo_onnx.py \
  --pt-model runs/exp05/yolo_finetune/weights/best.pt \
  --onnx-model artifacts/exp05/yolo_onnx/best.onnx \
  --image-path data/PCB_DATASET/images/Short/01_short_01.jpg
```

## 참고 문서
- `experiments/exp05_train_yolo/yolo_experiment.md`
