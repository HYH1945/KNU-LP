# BE

KNU-LP FastAPI 백엔드입니다. 실행 기준 위치는 루트의 `BE/` 디렉터리입니다.

## 실행

```bash
cd BE
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Windows에서는 `run_backend.bat`를 사용할 수 있습니다.

## 폴더 구조

```text
BE/
  api/                 FastAPI route
  configs/             settings.yaml
  pipeline/            파이프라인 orchestration 및 adapter
  vendors/             외부 연구 코드
    rgdiffsr/          RGDiffSR, LDM, PARSeq, text_super_resolution
    gplpr/             GP-LPR OCR 모델 코드
    taming/            taming-transformers 호환 코드
```

`pipeline/`은 우리 서비스 코드 중심으로 유지하고, 외부 구현체는 `vendors/` 아래로 분리했습니다. SR adapter는 `settings.yaml`의 `superresolution.paths.repo`를 통해 `BE/vendors/rgdiffsr`를 참조합니다.

## Weight

weight 파일은 Git에 포함하지 않습니다. 필요한 위치는 [WEIGHTS.md](WEIGHTS.md)를 참고하세요.

현재 YOLO 설정은 `BE/pipeline/plate_cropper/weights/best.pt`를 사용합니다. `BE/yolov8n-seg.pt`는 기본 설정에서 직접 참조하지 않습니다.

## API

### `POST /api/analyze`

`multipart/form-data` 요청을 받습니다.

| 필드 | 설명 |
|---|---|
| `files` | 이미지 모드: 이미지 여러 장, 영상 모드: 영상 1개 |
| `input_mode` | `image` 또는 `video` |
| `hr_width` | 기준 HR 너비 |
| `hr_height` | 기준 HR 높이 |
| `video_start` | 영상 분석 시작 초 |
| `video_end` | 영상 분석 종료 초 |
| `sr_mode` | `auto`, `always`, `skip` |
| `denoise_enabled` | denoiser 사용 여부 |

## 파이프라인

```text
uploaded images/video
-> video frame extraction
-> YOLO plate crop/highlight
-> original plate bbox 면적 기준 상위 5개 후보 선택
-> DnCNN denoiser
-> SR 또는 OCR ensemble 분기
-> OCR result
```

## 분기 기준

- 자동 SR 기준 면적은 `hr_width * hr_height * 0.8`입니다.
- 선택된 후보 중 가장 작은 original plate bbox 면적이 기준 이하이면 SR 경로로 보냅니다.
- `sr_mode=always` 또는 `sr_mode=skip`으로 자동 분기를 강제할 수 있습니다.

## 주요 응답 필드

| 필드 | 설명 |
|---|---|
| `input_preview` | 전체 입력 중 앞 5개 preview |
| `input_omitted_count` | 5개 이후 생략된 입력 수 |
| `selected_inputs` | 최종 선택 후보의 원본 프레임 |
| `selected_plate_bboxes` | original plate bbox 너비, 높이, 면적, confidence |
| `yolo_crops` | YOLO crop 및 perspective 보정 결과 |
| `yolo_selected` | 원본 프레임의 YOLO highlight 결과 |
| `denoised` | denoising 결과 |
| `sr` | SR 결과 또는 SR 생략 시 입력 이미지 |
| `ocr_text` | 최종 OCR 결과 |
| `yolo_ocr_preds` | 후보별 OCR 결과 |

## 주의사항

- 모델 weight가 없거나 로드 실패가 발생하면 파이프라인 중단을 막기 위해 fallback이 적용됩니다.
- SR/OCR까지 실제로 검증하려면 `requirements.txt` 설치와 weight 배치가 모두 필요합니다.
- `BE/vendors/rgdiffsr/parseq/strhub/data`는 데이터셋이 아니라 PARSeq 실행에 필요한 소스 코드입니다.
