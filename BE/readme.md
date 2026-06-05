# BE

KNU 번호판 인식 데모의 FastAPI 백엔드입니다.

## 실행 환경

- Python 3.10 권장
- CUDA가 없으면 일부 모델은 CPU로 자동 전환을 시도합니다.
- SR/OCR까지 실제 모델로 실행하려면 `requirements.txt` 설치와 weight 배치가 모두 필요합니다.

## 의존성 설치

주요 의존성은 `fastapi`, `uvicorn`, `python-multipart`, `opencv-python`, `torch`, `ultralytics`, `omegaconf`, `einops`입니다.

```bash
cd BE
pip install -r requirements.txt
```

## 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Windows에서는 `run_backend.bat`를 사용할 수 있습니다.

## 가중치 파일 위치

대용량 모델 가중치 파일은 Git 관리 대상에서 배제되어 있습니다. 정확한 파일명과 배치 구조는 [WEIGHTS.md](WEIGHTS.md)를 기준으로 확인해 주세요.

설정 파일은 `BE/configs/settings.yaml`입니다. 현재 기본 설정은 `BE/pipeline/**/weights/` 아래의 weight를 참조합니다. `BE/vendors/`는 외부 구현 코드 위치이며 weight를 넣는 폴더가 아닙니다.

## API

### `POST /api/analyze`

`multipart/form-data` 요청을 받습니다.

| 필드 | 설명 |
|---|---|
| `files` | 이미지 모드에서는 이미지 1장 이상, 영상 모드에서는 영상 1개 |
| `input_mode` | `image` 또는 `video` |
| `hr_width` | 사용자가 입력한 기준 HR 너비 |
| `hr_height` | 사용자가 입력한 기준 HR 높이 |
| `video_start` | 영상 분석 시작 초 |
| `video_end` | 영상 분석 종료 초 |
| `sr_mode` | `auto`, `always`, `skip` |
| `denoise_enabled` | 디노이징 사용 여부 |

## 파이프라인 흐름

```text
uploaded images/video
-> video frame extraction
-> YOLO plate crop/highlight
-> original plate bbox 기준 상위 5개 후보 선택
-> denoiser
-> SR 또는 OCR ensemble 분기
-> OCR result
```

영상 모드에서는 선택 구간에서 프레임을 추출한 뒤 이미지 모드와 동일하게 처리합니다. 이미지 입력이 이미 HR 번호판 크기 이하로 보이면 YOLO를 자동으로 생략하고 crop된 후보로 취급합니다.

## 입력 preview와 후보 선택

- `input_preview`에는 전체 입력 프레임 중 앞 5장을 담습니다.
- 입력 프레임이 5장을 넘으면 `input_omitted_count`로 생략된 개수를 전달합니다.
- YOLO는 전체 입력 프레임에 대해 실행합니다.
- 최종 처리 대상 5장은 YOLO 이후에 선택합니다.
- 선택 기준은 최종 crop 이미지 크기가 아니라 원본 프레임에서 탐지된 번호판 bbox 면적입니다.
- 정렬 우선순위는 `detected 여부 -> original plate bbox area -> confidence -> source index`입니다.
- `selected_inputs`는 최종 선택된 5개 crop에 대응되는 원본 프레임입니다.
- `selected_plate_bboxes`는 각 후보의 원본 bbox 너비, 높이, 면적, confidence, source index를 전달합니다.
- 번호판 crop 이미지는 perspective 보정 후 설정된 크기로 정규화되므로, 화면에 보이는 crop 해상도는 후보 정렬 기준으로 사용하지 않습니다.

## 자동 SR 분기 기준

- 기준 면적은 `hr_width * hr_height * 0.8`입니다.
- 선택된 후보 중 탐지된 원본 번호판 bbox의 최소 면적이 기준 면적 이하이면 SR 경로로 갑니다.
- 기준 면적을 넘는 고해상도 후보 수는 `high_resolution_count`로 응답에 포함합니다.
- `sr_mode=always` 또는 `sr_mode=skip`으로 자동 분기를 강제로 바꿀 수 있습니다.
- YOLO 탐지가 실패한 후보는 HR 후보로 계산하지 않습니다.

## 모듈 구조

```text
BE/
├── api/
│   └── routes.py
├── configs/
│   └── settings.yaml
├── pipeline/
│   ├── orchestrator.py
│   ├── processor/
│   │   ├── image_processor.py
│   │   └── video_processor.py
│   ├── plate_cropper/
│   ├── denoiser/
│   ├── superresolution/
│   │   └── sr_model.py
│   └── ocr/
│       ├── ocr_model.py
│       └── ensemble.py
└── vendors/
    ├── rgdiffsr/
    ├── gplpr/
    └── taming/
```

`pipeline/`은 우리 서비스 코드와 adapter 중심으로 유지하고, RGDiffSR/PARSeq/GP-LPR/taming 같은 외부 구현체는 `vendors/` 아래로 분리했습니다.

## 응답에서 사용하는 주요 필드

| 필드 | 설명 |
|---|---|
| `input_preview` | 전체 입력 중 앞 5개 preview |
| `input_omitted_count` | 5개 이후 생략된 입력 개수 |
| `selected_inputs` | 최종 선택된 5개 후보에 대응되는 원본 프레임 |
| `selected_source_indices` | 최종 선택 후보의 원본 입력 index |
| `selected_plate_bboxes` | 원본 프레임 기준 번호판 bbox 너비, 높이, 면적, confidence |
| `yolo_crops` | YOLO crop 및 perspective 보정 결과 |
| `yolo_selected` | 원본 프레임 위에 YOLO 탐지 결과를 표시한 이미지 |
| `denoised` | 디노이징 결과 |
| `sr` | SR 결과 또는 SR 생략 시 입력 통과 결과 |
| `ocr_text` | OCR 결과 문자열 |
| `yolo_ocr_preds` | 후보별 OCR 결과 문자열 |
| `pipeline_route` | `sr_then_ocr` 또는 `ocr_ensemble` |
| `sr_applied` | SR 적용 여부 |
| `high_resolution_count` | HR 기준 면적을 넘는 후보 수 |

## 구현 상태

| 항목 | 상태 | 설명 |
|---|---|---|
| 이미지 업로드 처리 | 구현됨 | Multi-file 업로드 가능 |
| 영상 업로드 처리 | 구현됨 | 영상 구간 프레임 자동 추출 기능 |
| YOLO 번호판 crop | 구현 완료 | 실제 YOLOv8-seg 연동 완료 (`best.pt`) |
| 5장 선정 | 구현 완료 | 번호판 bbox 면적 기준 내림차순 정렬 및 상위 5장 자동 픽 |
| DnCNN denoiser | 구현 완료 | 3채널 컬러 복합 열화 디노이저 연동 완료 (`best_model.pt`) |
| SR (Super Resolution) | 구현 완료 | RGDiffSR 초해상화 모델 adapter 연동 완료 (`SR.ckpt`, `VQGAN.ckpt`) |
| OCR | 구현 완료 | GP-LPR 기반 번호판 텍스트 해독 모델 연동 완료 (`best_model.pth`) |
| OCR ensemble voting | 구현 완료 | 여러 프레임 간의 글자 단위 다수결 보정 로직 연동 완료 |

## 현재 구현상 주의사항

- 각 모델 모듈(YOLO, DnCNN, SR, OCR)은 가중치 파일이 누락되었거나 로드 시 예외가 발생할 경우, 전체 파이프라인의 중단을 막기 위해 fallback 구조가 적용되어 있습니다. 따라서 정상 결과를 시연하려면 **가중치 파일 수동 배치가 필수적**입니다.
- `BE/vendors/rgdiffsr/parseq/strhub/data`는 데이터셋이 아니라 PARSeq 실행에 필요한 소스 코드입니다.
- `BE/yolov8n-seg.pt`를 전달받을 수 있지만, 현재 기본 파이프라인은 [WEIGHTS.md](WEIGHTS.md)의 YOLO crop weight를 사용합니다.
