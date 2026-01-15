# 🛠️ uv로 새 파이썬 프로젝트 시작하기

## uv 설치 가이드 (모든 운영체제)
uv 설치 가이드입니다. 더 다양한 설치법과 최신 안내는 공식문서를 참고하세요: https://docs.astral.sh/uv/getting-started/installation/

#### macOS / Linux
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### PyPI (pip, 운영체제 무관)
```sh
pip install uv
```

#### 설치 확인
```sh
uv --version
```


## 프로젝트 시작 가이드
uv 를 활용하여 프로젝트를 시작하는 가이드입니다. 더 자세한 내용은 공식문서를 참고하세요: https://docs.astral.sh/uv/guides/projects/

#### 1. 프로젝트 폴더 생성
```sh
mkdir my-uv-project
cd my-uv-project
```

#### 2. 프로젝트 초기화
```sh
uv init
```
- 안내에 따라 프로젝트명, 설명, (옵션) main 스크립트명 등을 입력하면 `pyproject.toml`과 (기본) main.py 스크립트가 만들어집니다.
- uv는 자동으로 가상환경(.venv)도 생성합니다.
- `uv init --python 3.11` 과같이 python 버전을 입력하면 해당 python 버전의 가상환경을 생성합니다.(3.8버전 이상부터 지원)

#### 3. main.py 스크립트 수정(선택)
생성된 main.py의 코드를 원하는 대로 편집할 수 있습니다.
예)
```python
print("Hello, uv!")
```

#### 4. uv로 main.py 실행
```sh
uv run main.py
```
- (참고) uv는 가상환경 자동 활성화 → 별도 source/activate 명령 없이 실행

#### 5. 실행 결과 예시
```sh
$ uv run main.py
Hello, uv!
```

## [참고] venv(전통적 방식)로 실행하는 예시
아래 명령은 uv 없이 직접 venv를 활성화해서 python을 실행하는 방법입니다.

#### 1. 프로젝트 폴더 생성
```sh
mkdir my-venv-project
cd my-venv-project
```

#### 2. 가상환경 생성 (python은 별도 설치 필요)
```sh
python -m venv .venv
```

#### 3. 가상환경 활성화
- macOS/Linux
  ```sh
  source .venv/bin/activate
  ```
- Windows (CMD)
  ```bat
  .venv\Scripts\activate
  ```
- Windows (PowerShell)
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

#### 4. 프로젝트 초기화 및 main.py 작성
```sh
uv init
```
(main.py가 만들어지며, 원하는 코드를 작성)

#### 5. 스크립트 실행
```sh
python main.py
```

#### 6. 실행 결과 예시
```sh
$ python main.py
Hello, uv!
```
