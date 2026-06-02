# BE

KNU 번호판 인식 데모의 FastAPI 백엔드입니다.

## 실행

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

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
-> processor
-> YOLO plate crop/highlight
-> original plate bbox 기준 상위 5개 후보 선택
-> denoiser
-> SR 또는 OCR ensemble 분기
-> OCR result
```

영상 모드에서는 선택 구간에서 프레임을 추출한 뒤 이미지 모드와 동일하게 처리합니다.

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
- 탐지된 원본 번호판 bbox 면적이 기준 면적을 넘으면 고해상도 후보로 계산합니다.
- 고해상도 후보가 3장 이상이면 SR을 생략하고 OCR ensemble 경로로 갑니다.
- 그 외에는 SR 경로를 거친 뒤 OCR로 갑니다.
- `sr_mode=always` 또는 `sr_mode=skip`으로 자동 분기를 강제로 바꿀 수 있습니다.
- YOLO 탐지가 실패한 후보는 HR 후보로 계산하지 않습니다.

## 모듈 구조

```text
BE/pipeline/
├── orchestrator.py
├── processor/
│   ├── image_processor.py
│   └── video_processor.py
├── plate_cropper/
├── denoiser/
├── superresolution/
│   └── sr_model.py
└── ocr/
    ├── ocr_model.py
    └── ensemble.py
```

## 현재 구현 상태

- YOLO와 DnCNN denoiser는 가중치가 없거나 로드에 실패하면 입력 이미지를 그대로 통과시키는 fallback을 사용합니다.
- SR은 현재 bicubic resize stub입니다. 실제 SR 모델은 `pipeline/superresolution/sr_model.py`의 `run_sr()`에 연결하면 됩니다.
- OCR은 현재 고정 문자열 stub입니다. 실제 OCR 모델은 `pipeline/ocr/ocr_model.py`에 연결하면 됩니다.
- OCR ensemble은 문자 단위 majority voting 구조를 먼저 고정해두었습니다.
