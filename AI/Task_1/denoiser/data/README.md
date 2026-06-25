# Dataset Generation

## 개요
본 모듈은 번호판 이미지에 다양한 노이즈를 인위적으로 추가하여 디노이징 모델 학습 및 평가를 위한 데이터셋을 생성한다.

입력은 YOLO 등을 통해 추출된 번호판 이미지이며, 출력은 clean / noisy 쌍으로 구성된 학습용 데이터셋이다.

---

## 입력 데이터 설명
- `generate_dataset.py`: `plates/` 폴더의 번호판 이미지를 Grayscale 데이터셋으로 변환
- `generate_color_dataset.py`: `AI/Task_1/train/`의 원본 번호판 이미지를 RGB 컬러 데이터셋으로 변환
- 데이터 형식: PNG, JPG, JPEG, BMP, WEBP

각 이미지에 대해 동일한 파일명으로 clean / noisy 쌍이 생성된다.

---

## 노이즈 생성 방식

본 데이터셋은 실제 CCTV 환경을 모사하기 위해 다양한 열화 요소를 조합하여 생성된다.

### 적용되는 노이즈 유형
- Gaussian Noise (센서 노이즈)
- Salt & Pepper Noise (강한 픽셀 손상)
- JPEG Compression (압축 손실)
- Motion Blur (이동 블러)
- Defocus Blur (초점 흐림)
- Brightness / Contrast 변화 (조명 변화)
- Downsampling + Upsampling (저해상도 환경)

각 노이즈는 확률적으로 적용되며, 여러 노이즈가 복합적으로 결합될 수 있다.

---

## 데이터 분할 방식

입력 데이터는 다음 비율로 자동 분할된다:

- Train: 80%
- Validation: 10%
- Test: 10%

랜덤 셔플 후 분할되며, 각 실행마다 결과가 달라질 수 있다.

---

## 출력 구조

생성된 데이터셋은 다음과 같은 구조를 가진다. 생성 결과는 용량 문제로 GitHub에 포함하지 않는다.

```
data/dataset/
  ├── train/
  │    ├── clean/
  │    └── noisy/
  ├── val/
  │    ├── clean/
  │    └── noisy/
  └── test/
       ├── clean/
       └── noisy/

data/dataset_color/
  ├── train/
  │    ├── clean/
  │    └── noisy/
  ├── val/
  │    ├── clean/
  │    └── noisy/
  └── test/
       ├── clean/
       └── noisy/
```

- `clean/` : 원본 이미지
- `noisy/` : 노이즈가 추가된 이미지

각 split에서 동일한 파일명을 기준으로 clean / noisy가 1:1 대응된다.

---

## 실행 방법

다음 명령어를 통해 데이터셋을 생성할 수 있다:

```
cd AI/Task_1/denoiser/data
python generate_dataset.py
python generate_color_dataset.py
```

기본 설정:
- Grayscale 입력 폴더: `plates/`
- Grayscale 출력 폴더: `data/dataset/`
- RGB 입력 폴더: `AI/Task_1/train/`
- RGB 출력 폴더: `data/dataset_color/`
- preview 이미지 출력: 5장
- 열화 강도: `strong`

---

## 참고 사항

- 데이터셋은 용량 문제로 GitHub에 포함되지 않는다.
- 동일한 설정으로 재생성 가능하도록 코드가 제공된다.
- 학습/평가 코드에서 동일한 구조를 사용하도록 설계되었다.
