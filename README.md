# KNU-LP

CCTV 같은 저화질 환경에서도 동작하는 번호판 인식 파이프라인입니다.

전체 흐름은 다음과 같습니다.

```text
이미지/영상 입력
-> 영상이면 프레임 추출
-> YOLO 번호판 검출 및 crop
-> original plate bbox 면적 기준 상위 5개 후보 선택
-> DnCNN denoising
-> SR 또는 OCR ensemble 분기
-> OCR 결과 표시
```

## 실행 기준

- 실제 실행 기준 디렉터리는 루트의 `BE/`, `FE/`입니다.
- 전달용 코드 묶음이었던 `knulp/` 폴더는 실행에 사용하지 않습니다.
- 대용량 weight 파일은 Git에 포함하지 않습니다.

## 백엔드 구조

```text
BE/
  api/                 FastAPI route
  configs/             settings.yaml
  pipeline/            우리 프로젝트 파이프라인/adapter 코드
  vendors/             외부 연구 코드 및 모델 구현체
    rgdiffsr/
    gplpr/
    taming/
```

`pipeline/`에는 파이프라인 연결 코드만 두고, RGDiffSR/PARSeq/GP-LPR/taming 같은 외부 코드는 `vendors/` 아래로 분리했습니다.

## Weight 배치

현재 설정이 참조하는 weight 경로는 아래와 같습니다.

```text
BE/pipeline/plate_cropper/weights/best.pt
BE/pipeline/denoiser/weights/best_model.pt
BE/pipeline/superresolution/weights/SR.ckpt
BE/pipeline/superresolution/weights/VQGAN.ckpt
BE/pipeline/superresolution/weights/parseq.pt
BE/pipeline/ocr/weights/best_model.pth
```

별도로 전달받은 `BE/yolov8n-seg.pt`는 현재 기본 설정에서는 직접 사용하지 않습니다. YOLO 번호판 검출은 `BE/pipeline/plate_cropper/weights/best.pt`를 사용합니다.

## BE 실행

```bash
cd BE
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Windows에서는 `BE/run_backend.bat`를 사용할 수도 있습니다.

## FE 실행

```bash
cd FE
npm install
npm run dev
```

FE의 `/api` 요청은 Vite proxy를 통해 `http://localhost:8000`으로 전달됩니다.

## 현재 구현 상태

| 항목 | 상태 |
|---|---|
| 이미지/영상 업로드 | 구현됨 |
| 영상 프레임 추출 | 구현됨 |
| YOLO 번호판 crop/highlight | 실제 모델 연결됨 |
| 후보 5개 선정 | YOLO 이후 original plate bbox 면적 기준 |
| DnCNN denoising | 컬러 DnCNN weight로 동작 확인 |
| SR | RGDiffSR adapter 연결 및 로드 확인 |
| OCR | GP-LPR adapter 연결 및 로드 확인 |
| OCR ensemble | 후보별 OCR 결과 character voting |

## 참고 문서

- [BE/readme.md](BE/readme.md)
- [BE/WEIGHTS.md](BE/WEIGHTS.md)
- [FE/readme.md](FE/readme.md)
- [rule.md](rule.md)
