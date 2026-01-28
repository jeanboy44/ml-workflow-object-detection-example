# scripts

스크립트 실행 예시 모음.

## Databricks MLflow run 모델 다운로드

MLflow run 기준으로 모델 아티팩트를 내려받습니다.

```bash
python scripts/download_model_from_run.py <run_id> <artifact_path>
```

예시:

```bash
python scripts/download_model_from_run.py 1234567890abcdef model_name
```

## Databricks Volume 모델/데이터 다운로드

`scripts/dbx_volumes.py`로 Databricks Volume의 파일을 내려받습니다.

- 모델 파일은 `artifacts` 아래에 다운로드합니다.
- 데이터셋은 `data` 아래에 다운로드합니다.

예시:

```bash
# model_files -> artifacts
python scripts/dbx_volumes.py download /Volumes/study/object_detection/model_files/yolo/ --dst-path ./artifacts
python scripts/dbx_volumes.py download /Volumes/study/object_detection/model_files/IDEA-Research/ --dst-path ./artifacts
python scripts/dbx_volumes.py download /Volumes/study/object_detection/model_files/facebook/ --dst-path ./artifacts

# datasets -> data
python scripts/dbx_volumes.py download /Volumes/study/object_detection/datasets/PCB_DATASET/ --dst-path ./data
```
