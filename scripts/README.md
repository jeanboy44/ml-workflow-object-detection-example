# scripts

스크립트 실행 예시 모음.

## 프로젝트 초기 설정 점검

필수 도구/파일이 준비되었는지 확인합니다.

```bash
python scripts/check_project_setup.py
```

## Databricks MLflow run 모델 다운로드

MLflow run 기준으로 모델 아티팩트를 내려받습니다.

```bash
python scripts/download_model_from_run.py <run_id> <model_path>
```

예시:

```bash
python scripts/download_model_from_run.py 1234567890abcdef model_name
```

## YOLO PT -> ONNX 변환 및 검증

다운로드한 `.pt` 모델을 ONNX로 변환하고, ONNX 유효성 및 예측 결과를 검증합니다.
예측 검증은 `experiments/data_sample` 폴더의 이미지들을 사용합니다.

```bash
python scripts/convert_pt_to_onnx.py artifacts/runs/<run_id>/model/best.pt \
  --output-dir artifacts/onnx \
  --sample-dir experiments/data_sample
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
