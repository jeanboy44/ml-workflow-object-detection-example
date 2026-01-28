# ml-object-detector
MLflow Registry에서 YOLO 모델을 내려받아 간단히 예측하는 패키지입니다.

## 설치 (uv)
```bash
uv pip install "git+https://github.com/jeanboy44/ml-workflow-object-detection-example.git#subdirectory=packages/ml-object-detector"
```

## 사용 예시
```python
from ml_object_detector import load, load_image, predict

model = load("models:/exp05_yolo/Production", tracking_uri="databricks")
image = load_image("sample.jpg")
result = predict(model, image, threshold=0.25)
print(result.detections)
```

MLflow Databricks 연결 시 `.env`의 `DATABRICKS_HOST`, `DATABRICKS_TOKEN` 환경 변수를 사용합니다.
`ML_OBJECT_DETECTOR_CACHE_DIR`를 설정하면 모델 가중치를 캐시하고, 캐시에 유효한 `.pt`가 있으면 다운로드를 생략합니다.
실제 실행 예시는 `examples/ml_object_detector_demo.py`를 참고하세요.

## 테스트
```bash
uv run pytest packages/ml-object-detector/tests
```
