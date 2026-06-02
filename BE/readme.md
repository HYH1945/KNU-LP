# BE

KNU 번호판 인식 데모용 FastAPI 백엔드입니다.

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
| `hr_width` | 사용자 기준 HR 너비 |
| `hr_height` | 사용자 기준 HR 높이 |
| `video_start` | 영상 분석 시작 초 |
| `video_end` | 영상 분석 종료 초 |
| `sr_mode` | `auto`, `always`, `skip` |
| `denoise_enabled` | 디노이징 사용 여부 |

## 파이프라인 흐름

```text
uploaded images/video
-> processor
-> YOLO plate crop/highlight
-> top 5 crops by resolution
-> denoiser
-> SR or OCR ensemble branch
-> OCR result
```

영상 모드에서는 선택 구간에서 프레임을 추출한 뒤 이미지 모드와 동일하게 처리합니다.

## 모듈 구조

```text
BE/pipeline/
├── orchestrator.py
├── processor/
│   ├── image_processor.py
│   └── video_processor.py
├── denoiser/
├── superresolution/
│   └── sr_model.py
└── ocr/
    ├── ocr_model.py
    └── ensemble.py
```

## 현재 구현 상태

- YOLO와 DnCNN denoiser는 weight 또는 의존성이 없으면 입력 이미지를 그대로 통과시키는 fallback을 사용합니다.
- SR은 현재 bicubic resize stub입니다. 실제 SR 모델은 `pipeline/superresolution/sr_model.py`의 `run_sr()` 내부에 연결하면 됩니다.
- OCR은 현재 고정 문자열 stub입니다. 실제 OCR 모델은 `pipeline/ocr/ocr_model.py`에 연결하면 됩니다.
- OCR ensemble은 문자열 리스트를 문자 단위 majority voting 하는 구조를 먼저 고정해두었습니다.

## 분기 기준

- 선택된 번호판 crop의 면적이 `hr_width * hr_height * 0.8`보다 큰 경우 고해상도 후보로 계산합니다.
- 고해상도 후보가 3장 이상이면 SR을 생략하고 OCR ensemble 경로로 갑니다.
- 그 외에는 SR stub을 거친 뒤 단일 OCR 경로로 갑니다.
- `sr_mode=always` 또는 `sr_mode=skip`으로 자동 분기를 강제로 바꿀 수 있습니다.
