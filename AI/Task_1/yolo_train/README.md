# YOLO Segmentation 번호판 학습 프로젝트

## 디렉토리 구조

```
yolo_train/
├── train_main.py              # 진입점
├── dataset.yaml               # 데이터셋 경로 및 클래스 설정
├── modules/
│   ├── dataset_validator.py   # 데이터셋 구조·라벨 무결성 검증
│   ├── plate_segmenter.py     # 학습된 모델 추론 및 폴리곤/마스크 추출
│   ├── train_config.py        # 전체 하이퍼파라미터 설정 (dataclass)
│   ├── model_trainer.py       # 학습 실행 (YOLO.train 래핑)
│   └── format_converter.py    # COCO/LabelMe → YOLO seg 변환
└── temp/                      # 임시 파일 (자동 생성)
```

---

## 설치

본 프로젝트는 GPU(CUDA)를 활용하여 학습 및 추론을 진행하며, PyTorch 2.5.1 (CUDA 12.1) 환경에서 검증되었다.

```bash
pip install ultralytics>=8.0 torch torchvision pyyaml
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu121"
```

---

## 사용 흐름

### 1. 라벨 변환 (기존 라벨이 COCO/LabelMe인 경우)

```python
from modules.format_converter import FormatConverter

converter = FormatConverter(output_root="./dataset")

# COCO JSON인 경우
converter.from_coco(
    coco_json_path="annotations/train.json",
    images_dir="images/",
    split="train",
    class_names=["license_plate"],
)

# LabelMe JSON 폴더인 경우
result = converter.from_labelme(labelme_dir="labelme_jsons/", split="train")

# dataset.yaml 자동 생성
converter.generate_yaml(class_names=["license_plate"])
```

### 2. 데이터셋 검증 (단독 실행)

```bash
python modules/dataset_validator.py ./dataset
```

### 3. 학습 실행

```bash
# 기본 실행
python train_main.py --data dataset.yaml --weights yolov8s-seg.pt --epochs 100 --batch 16 --device 0

# JSON 설정 파일 사용
python train_main.py --config temp/last_train_config.json

# 중단된 학습 재개
python train_main.py --resume

# 데이터셋 검증 생략
python train_main.py --skip-val
```

---

## 주요 설정 (train_config.py)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `pretrained_weights` | `yolov8n-seg.pt` | n/s/m/l/x 중 선택 |
| `epochs` | 100 | |
| `batch_size` | 16 | OOM 시 8로 낮춤 |
| `imgsz` | 640 | |
| `perspective` | 0.0005 | 원근 왜곡 증강 (번호판 핵심) |
| `fliplr` | 0.0 | 번호판 좌우 반전 OFF |
| `copy_paste` | 0.1 | seg 전용 증강 |
| `patience` | 30 | Early stopping |

---

## 학습 결과 위치

```
runs/train/plate_seg/
├── weights/
│   ├── best.pt    ← 최적 가중치
│   └── last.pt    ← 마지막 체크포인트
├── train_summary.json
└── (plots, confusion_matrix, ...)
```

---

## 번호판 전용 권장 설정

```python
TrainConfig(
    pretrained_weights = "yolov8s-seg.pt",   # s 이상 권장
    epochs             = 100,
    batch_size         = 16,
    perspective        = 0.001,  # 원근 왜곡 강화
    degrees            = 10.0,   # 회전 범위 확대
    fliplr             = 0.0,    # 좌우 반전 OFF (번호 깨짐)
    flipud             = 0.0,    # 상하 반전 OFF
    copy_paste         = 0.3,    # 데이터 적을 때 증가
)
```
