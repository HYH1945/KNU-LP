# FE

KNU-LP React + Vite 프론트엔드입니다. 실행 기준 위치는 루트의 `FE/` 디렉터리입니다.

## 실행

```bash
cd FE
npm install
npm run dev
```

Vite proxy 설정으로 `/api` 요청은 `http://localhost:8000` 백엔드로 전달됩니다. FE 실행 전에 BE 서버를 먼저 실행하세요.

## 화면 동작

- 이미지 여러 장 또는 영상 1개를 업로드할 수 있습니다.
- 영상 모드에서는 분석 구간을 지정할 수 있습니다.
- HR 기준 해상도(`hr_width`, `hr_height`), SR 모드(`auto`, `always`, `skip`), denoiser 사용 여부를 설정합니다.
- 결과는 `INPUT`, `YOLO Crop`, `Denoised`, `SR` 단계별 5개 후보 기준으로 표시합니다.
- Detail 모달에서는 이미지 표시 크기와 original plate bbox 정보를 확인할 수 있습니다.
- OCR 영역에는 최종 OCR 결과와 후보별 OCR 결과를 함께 표시합니다.

## 주요 응답 필드

| 필드 | 설명 |
|---|---|
| `input_preview` | 전체 입력 중 앞 5개 preview |
| `input_omitted_count` | 5개 이후 생략된 입력 수 |
| `selected_inputs` | 최종 선택 후보의 원본 프레임 |
| `selected_plate_bboxes` | original plate bbox metadata |
| `yolo_crops` | YOLO crop 결과 |
| `yolo_selected` | 원본 프레임의 YOLO highlight 결과 |
| `denoised` | denoising 결과 |
| `sr` | SR 결과 또는 SR 생략 시 입력 이미지 |
| `ocr_text` | 최종 OCR 결과 |
| `yolo_ocr_preds` | 후보별 OCR 결과 |

## 더미 모드

`FE/src/constants/dummy.js`의 `USE_DUMMY`를 `true`로 바꾸면 백엔드 없이 placeholder 응답으로 UI를 확인할 수 있습니다.
