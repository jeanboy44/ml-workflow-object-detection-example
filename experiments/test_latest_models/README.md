# Test Latest Models

사전학습 모델(GroundingDINO, DETR, YOLO)을 빠르게 테스트하는 스크립트 모음입니다. 샘플 이미지에 대해 추론하고 결과 시각화를 저장합니다.

## 사전 준비

```sh
uv run scripts/download_files.py jgryu \
  --blob-path ml-workflow-object-detection-example/IDEA-Research/grounding-dino-base/ \
  --dst-path artifacts/IDEA-Research/grounding-dino-base/

uv run scripts/download_files.py jgryu \
  --blob-path ml-workflow-object-detection-example/facebook/detr-resnet-50/ \
  --dst-path artifacts/facebook/detr-resnet-50/
```

## 실행

### GroundingDINO (Zero-shot)
```sh
uv run experiments/test_latest_models/grounddino_zero_shot_inference.py \
  experiments/sample_data/cat_03.jpg "cat" \
  --threshold 0.2 \
  --save-path data/output_ground_dino/cat_03.jpg
```

### DETR (Pretrained)
```sh
uv run experiments/test_latest_models/detr_pre_trained_model.py \
  experiments/sample_data \
  --output-dir data/output_detr \
  --score-threshold 0.7
```

### YOLO (Pretrained)
```sh
uv run experiments/test_latest_models/yolo_pre_trained_model.py \
  experiments/sample_data \
  --output-dir data/output_yolo26n \
  --model-path artifacts/yolo/yolo26n.pt
```
