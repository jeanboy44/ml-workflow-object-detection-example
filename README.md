# ml-workflow-object-detection-example
머신러닝 워크플로우(Object Detection) 예제 프로젝트입니다. 이 저장소는 기본적인 객체 탐지 파이프라인 예제를 제공합니다.

## 🚀 프로젝트 시작하기
### uv 설치
#### macOS / Linux
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
자세한 설치 방법은 [uv 공식 설치 문서](https://docs.astral.sh/uv/getting-started/installation/)에서 확인하세요.

### 의존성 설치
```sh
uv sync  # 모든 의존성 설치 (pyproject.toml/uv.lock 기반)
```

## 🧑‍💻 개발 환경
- Python >= 3.11
- uv >= 0.9.25
- macOS, Linux, Windows 지원

## 📦 디렉토리 구조
```
📦 프로젝트 폴더 구조

.
├── .vscode/                    # VSCode 에디터 설정
├── data/                       # 데이터 및 샘플 저장소
├── experiments/                # 실험 코드 및 샘플 테스트
├── packages                    # 프로젝트의 파이썬 모듈(패키지) 저장소
│   └── ml-object-detector     # "ml-object-detector" 파이썬 패키지 루트 디렉토리
├── tests/                      # 프로젝트 루트 테스트 코드
├── .gitignore                  # Git 무시 파일 목록
├── .pre-commit-config.yaml     # pre-commit 관리 파일
├── .python-version             # 파이썬 버전 지정 파일
├── pyproject.toml              # 프로젝트 전체 Python 설정/의존성
├── README.md                   # 프로젝트 안내 파일
└── uv.lock                     # Python 패키지 lock 파일
```


## ℹ️ 추가 정보
- uv 공식 문서: https://docs.astral.sh/uv/
