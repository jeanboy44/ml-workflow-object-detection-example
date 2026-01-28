# PCB 결함 데이터셋 소개

본 데이터셋은 인쇄회로기판(PCB, Printed Circuit Board)에서 발생하는 결함(Defects) 감지 및 분류를 위한 고해상도 이미지와 라벨 정보를 제공합니다. 머신러닝/딥러닝 기반의 불량 검사 모델 개발 및 평가에 적합합니다.

---

## 📦 데이터 구성

- **train/**: 훈련용 이미지 디렉터리 (jpg/png)
- **test/**: 테스트용 이미지 디렉터리
- **train.csv**: 훈련 데이터의 레이블 및 위치 정보(csv)
    - 주요 컬럼: `filename`, `class`, `xmin`, `ymin`, `xmax`, `ymax`
- **test.csv**: 테스트 데이터 레이블(있는 경우)
- **classlist.txt**: 결함 클래스 목록

---

## 🏷️ 클래스(결함 유형)

다음과 같은 결함 종류가 포함되어 있습니다(예시):
- Short (쇼트)
- Open (단선)
- Mousebite (부식/이물불량)
- Spur (돌기, 바늘)
- Spurious Copper (불필요한 동박)
- 기타

---

## 🖼️ 예제 레이블
| filename     | class      | xmin | ymin | xmax | ymax |
|--------------|------------|------|------|------|------|
| pcb_train1.jpg | Short    | 120  | 30   | 230  | 165  |
| pcb_train2.jpg | Spur     | 400  | 80   | 488  | 165  |
| ... | ... | ... | ... | ... | ... |


---

## ⚠️ 라이선스/기타

- 목적: 비전(Visual Inspection), 불량 검출 알고리즘 실험/연구, 논문, 과제 등
- 라이선스: Kaggle 및 데이터셋 페이지 명시 조건을 따르세요.
- 데이터셋 원 저작자/관리자: Kaggle 유저 [Aigerim Akhatova](https://www.kaggle.com/akhatova)

---

## 🔗 원본 링크

- [Kaggle: PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects)
