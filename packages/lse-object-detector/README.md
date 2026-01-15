# lse-object-detector
객체 탐지 패키지입니다.

## 📦 설치 방법
```bash
uv pip install "git+https://github.com/jeanboy44/ml-workflow-object-detection-example.git#subdirectory=packages/lse-object-detector"
```
`#subdirectory=packages/lse-object-detector` 옵션을 반드시 포함하세요.
- Python >= 3.11 필요
- uv 최신 버전 권장(≥0.9.25)

## ⚡️ 주요 기능
- 객체 탐지 기능

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
lse-object-detector/
├── README.md            # 패키지 상세 안내 파일
├── pyproject.toml       # 패키지 메타데이터/의존성/빌드 설정 파일
├── src                  # 실제 파이썬 코드(모듈) 폴더, 표준 src layout
│   └── lse_object_detector    # "lse_object_detector" 패키지 네임스페이스 모듈 디렉토리
└── tests                # 테스트 코드 저장 디렉토리
```
