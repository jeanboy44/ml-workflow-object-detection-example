# ml-object-detector
객체 탐지 패키지입니다.

## 📦 설치 방법
```bash
uv pip install "git+https://github.com/jeanboy44/ml-workflow-object-detection-example.git#subdirectory=packages/ml-object-detector"
```
`#subdirectory=packages/ml-object-detector` 옵션을 반드시 포함하세요.
- Python >= 3.11 필요
- uv 최신 버전 권장(≥0.9.25)

## ⚡️ 주요 기능
- MLflow에 등록된 모델 다운로드 및 예측

## 🧩 사용 예시

### MLflow Registry (YOLO)
```python
from ml_object_detector import load, load_image, predict

model = load("models:/exp05_yolo/Production")
image = load_image("sample.jpg")
result = predict(model, image, threshold=0.25)
print(result.detections)
```

환경 변수 `DATABRICKS_HOST`, `DATABRICKS_TOKEN`은 MLflow에서 사용합니다.

## 🛠️ 개발/테스트 환경
- Python >= 3.11
- uv, poetry, pip 모두 지원
- 프로젝트 루트의 pyproject.toml을 기본으로 사용

### 테스트 실행 예시 (pytest)
```bash
pytest tests/
```

## 폴더 구조
```
ml-object-detector/
├── README.md            # 패키지 상세 안내 파일
├── pyproject.toml       # 패키지 메타데이터/의존성/빌드 설정 파일
├── src                  # 실제 파이썬 코드(모듈) 폴더, 표준 src layout
│   └── ml_object_detector    # "ml_object_detector" 패키지 네임스페이스 모듈 디렉토리
└── tests                # 테스트 코드 저장 디렉토리
```
