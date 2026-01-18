
### Pre Trained model 1: DETR
```
uv run experiments/e01_detr_pre_trained_model.py data/PCB_DATASET/images/MOUSE_bite --output-dir data/output_detr/PCB_DATASET_MOUSE_bite --score-threshold 0.7
```


### Pre Trained model 2: Yolo
```
uv run experiments/e02_yolo_pre_trained_model.py data/PCB_DATASET/images/MOUSE_bite --output-dir data/output_yolo26n/PCB_DATASET_MOUSE_bite/
```