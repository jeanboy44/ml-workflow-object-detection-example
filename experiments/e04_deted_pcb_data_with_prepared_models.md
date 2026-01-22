
### Pre Trained model 1: DETR
```
uv run experiments/e01_detr_pre_trained_model.py data/PCB_DATASET/images/MOUSE_bite --output-dir data/output_detr/PCB_DATASET_MOUSE_bite --score-threshold 0.7
```


### Pre Trained model 2: Yolo
```
uv run experiments/e02_yolo_pre_trained_model.py data/PCB_DATASET/images/MOUSE_bite --output-dir data/output_yolo26n/PCB_DATASET_MOUSE_bite/
```


### Zero-shot model: Ground-dino
#### Prompt 1. cat
```
uv run experiments/e00_grounddino_zero_shot_inference.py data/PCB_DATASET/images/Short/01_short_01.jpg "cat" --threshold 0.2 --save-path data/output_ground_dino/01_short_01_1.jpg
```

#### Prompt 2. silver circle
```
uv run experiments/e00_grounddino_zero_shot_inference.py data/PCB_DATASET/images/Short/01_short_01.jpg "white circle" --threshold 0.2 --save-path data/output_ground_dino/01_short_01_2.jpg
```

#### Prompt 3. circle
```
uv run experiments/e00_grounddino_zero_shot_inference.py data/PCB_DATASET/images/Spurious_copper/10_spurious_copper_05.jpg "circle" --threshold 0.2 --save-path data/output_ground_dino/10_spurious_copper_05_1.jpg
```

#### Prompt 4. square
```
uv run experiments/e00_grounddino_zero_shot_inference.py data/PCB_DATASET/images/Spurious_copper/10_spurious_copper_05.jpg "square" --threshold 0.2 --save-path data/output_ground_dino/10_spurious_copper_05_2.jpg
```


#### Prompt 5. circle, square
```
uv run experiments/e00_grounddino_zero_shot_inference.py data/PCB_DATASET/images/Spurious_copper/10_spurious_copper_05.jpg "circle, square" --threshold 0.2 --save-path data/output_ground_dino/10_spurious_copper_05_3.jpg
```


