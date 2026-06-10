# FE

KNU 번호판 인식 데모의 React + Vite 프론트엔드입니다.

## 실행 환경

- Node.js 20 이상 권장
- npm 사용

## 실행

```bash
cd FE
npm install
npm run dev
```

Vite proxy 설정 때문에 프론트엔드의 `/api` 요청은 자동으로 `http://localhost:8000` 백엔드로 전달됩니다. 따라서 FE 테스트 전에 BE 서버를 먼저 실행해야 합니다.
BE 서버 실행 방법과 가중치 위치는 [../BE/readme.md](../BE/readme.md), [../BE/WEIGHTS.md](../BE/WEIGHTS.md)를 참고합니다.

## 빌드

```bash
npm run build
npm run preview
```

## 동작 요약

- 이미지 모드에서는 여러 장의 이미지를 업로드할 수 있습니다.
- 영상 모드에서는 영상 1개를 업로드하고 분석 구간을 지정할 수 있습니다.
- HR 기준 해상도(`hr_width`, `hr_height`), SR 모드(`auto`, `always`, `skip`), 디노이징 사용 여부를 입력합니다.
- 분석 요청은 `POST /api/analyze`로 전송합니다.
- 결과 화면은 `INPUT`, `YOLO Crop`, `Denoised`, `SR` 단계와 OCR 결과를 표시합니다.
- OCR 영역에는 최종 OCR 결과와 후보별 OCR 결과를 함께 표시합니다.

## 백엔드 요청 필드

| 필드 | 설명 |
|---|---|
| `files` | 이미지 모드: 이미지 여러 장, 영상 모드: 영상 1개 |
| `input_mode` | `image` 또는 `video` |
| `hr_width` | 기준 HR 너비 |
| `hr_height` | 기준 HR 높이 |
| `video_start` | 영상 분석 시작 초 |
| `video_end` | 영상 분석 종료 초 |
| `sr_mode` | `auto`, `always`, `skip` |
| `denoise_enabled` | 디노이징 사용 여부 |

## 결과 표시 방식

- `INPUT` 행은 `input_preview`에 담긴 앞 5개 원본 입력을 보여줍니다.
- 입력이 5장을 넘으면 마지막 INPUT 칸에 `+N` 형태로 생략된 개수를 표시합니다.
- `YOLO Crop`, `Denoised`, `SR` 행은 YOLO 이후 선택된 최종 5개 후보를 기준으로 표시합니다.
- Detail 모달은 이미지의 실제 표시 데이터 크기와 원본 프레임 기준 번호판 bbox 정보를 함께 보여줍니다.
- 번호판 crop 이미지는 백엔드에서 perspective 보정 후 고정 크기로 정규화되므로, crop 이미지 해상도는 후보 정렬 기준이 아닙니다.
- 후보 정렬과 자동 SR 분기는 백엔드가 전달하는 원본 번호판 bbox 면적 기준으로 이루어집니다.
- 분석 summary 영역은 fallback이 발생한 처리 단계를 함께 표시합니다.

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
| `warnings` | 단계별 fallback 또는 decode 실패 경고 목록 |
| `stage_fallbacks` | yolo, denoise, sr, ocr 단계별 fallback 발생 여부 |

## 구현 상태

| 항목 | 상태 |
|---|---|
| 이미지 업로드 처리 | 구현됨 |
| 영상 업로드 처리 | 구현됨 |
| 영상 프레임 추출 결과 표시 | 구현됨 |
| YOLO 번호판 crop/highlight 결과 표시 | 구현됨 |
| 5장 선정 결과 표시 | YOLO 이후 original plate bbox 면적 기준 |
| INPUT preview | 앞 5장 표시 및 `+N` 생략 표시 구현됨 |
| HR Width/Height 입력 | 구현됨 |
| SR 모드 선택 | `auto`, `always`, `skip` 구현됨 |
| Denoiser 사용 여부 선택 | 구현됨 |
| Detail 모달 | 이미지 크기와 original plate bbox 표시 |
| SR | 백엔드 실제 SR 결과 표시 |
| OCR | 백엔드 실제 OCR 결과 표시 |

## 더미 모드

`FE/src/constants/dummy.js`의 `USE_DUMMY`를 `true`로 바꾸면 백엔드 없이 placeholder 응답으로 UI를 확인할 수 있습니다.
