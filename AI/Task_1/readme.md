
인스턴스 세그멘테이션(YOLO-seg)을 활용하여 이미지 내 번호판 영역을 픽셀 단위로 탐지하고, 투시 보정(Perspective Transform)을 거쳐 정면 번호판 이미지를 추출하는 전처리 파이프라인입니다. 

사용자 편의를 위해 `input` 디렉토리 하위의 폴더들을 자동으로 탐색하며, 처리된 결과물은 `output` 디렉토리 아래 동일한 폴더 구조로 저장됩니다.

---

## 1. 입출력 폴더 구조 및 처리 방식

`main.py` 실행 시 `input` 디렉토리 하위의 모든 이미지셋 폴더를 순차적으로 탐색하여 크롭(Crop) 및 보정을 수행합니다.

```text
plate_detection_seg/
├── input/
│   ├── imageset_A/          # 입력 이미지셋 A
│   │   ├── frame_1.jpg
│   │   └── frame_2.jpg
│   └── imageset_B/          # 입력 이미지셋 B
│       └── frame_1.jpg
├── output/
│   ├── imageset_A/          # 동일한 폴더명으로 결과물 저장
│   │   ├── frame_1_det0.png
│   │   ├── frame_2_det0.png
│   │   └── result_log.json  # 해당 폴더의 처리 로그
│   └── imageset_B/
│       ├── frame_1_det0.png
│       └── result_log.json
├── temp/
│   └── progress.json        # 중단 시 재개를 위한 진행 상태 기록
├── configs/
├── modules/
└── main.py
```

---

## 2. 환경 설정 및 실행

### 3.1 요구 사항
* **OS:** Windows, Linux, macOS (Python 및 PyTorch 구동 환경)
* **버전:** Python 3.9 이상
* **라이브러리:** `opencv-python`, `numpy`, `torch`, `ultralytics`, `pyyaml`, `matplotlib`

### 3.2 실행 명령어

**전체 파이프라인 일괄 실행:**
`input/` 하위 폴더를 자동 탐색하여 작업을 수행합니다.
```bash
python main.py --config configs/settings.yaml
```

**시각화 보조 도구 실행 (선택 사항):**
처리 과정을 눈으로 쉽게 확인하기 위해 만들어진 편의성 스크립트입니다. 원본, 분할 마스크, 투시 변환 결과를 비교하는 요약 테이블(`segmentation_summary_table.png`)을 생성합니다.
```bash
python result_visualizer.py
```