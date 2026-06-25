# KNU-LP

## 저화질 CCTV 환경을 위한 번호판 인식 파이프라인

프로젝트 목표는 CCTV, 단속 카메라와 같이 해상도가 낮고 노이즈가 포함된 영상 환경에서도 번호판을 안정적으로 검출하고 인식할 수 있는 end-to-end 파이프라인을 구현하는 것입니다.

<img width="1182" height="470" alt="image" src="https://github.com/user-attachments/assets/e84fcf36-bbd3-44b1-b907-c079ca812e31" />


프로젝트는 크게 **전처리**, **초해상화(Super-Resolution)**, **OCR** 세 단계로 구성되며, 최종 시연을 위해 React 기반 Web UI와 FastAPI 기반 Backend를 함께 제공합니다.

## 시스템 개요

```text
이미지 / 영상 입력
-> 영상 구간 선택 및 프레임 추출
-> YOLO 기반 번호판 검출 및 crop
-> 원본 번호판 bbox 면적 기준 상위 5개 후보 선택
-> NAFNet 기반 컬러 디노이징
-> SR 적용 여부 분기
-> OCR 또는 OCR ensemble voting
-> Web UI에서 단계별 결과 및 최종 번호판 문자열 확인
```

본 시스템은 모든 입력을 동일한 방식으로 처리하지 않고, 입력 품질과 사용자 설정에 따라 일부 모듈을 skip하거나 ensemble 경로를 사용하는 정책을 포함합니다.
이를 통해 불필요한 연산을 줄이고, 실제 저화질 영상 환경에서 필요한 처리 흐름을 유연하게 선택할 수 있도록 구성했습니다.

## 주요 기능

| 구분 | 내용 |
|---|---|
| 입력 처리 | 이미지 다중 업로드 및 영상 업로드 지원 |
| 영상 처리 | 사용자가 지정한 구간에서 프레임 추출 |
| 번호판 검출 | YOLOv8-seg 기반 번호판 영역 검출 및 perspective 보정 |
| 후보 선택 | 검출된 번호판 bbox 면적 기준 상위 5개 후보 자동 선택 |
| 전처리 | 3채널 컬러 NAFNet 기반 디노이징 적용 |
| SR 분기 | 사용자 입력 HR 해상도와 번호판 bbox 면적을 비교하여 SR 적용 여부 결정 |
| OCR | GP-LPR 기반 번호판 문자 인식 |
| Ensemble | 고해상도 후보가 3개 이상인 경우 OCR 결과 majority voting 수행 |
| 결과 표시 | INPUT, YOLO Crop, Denoised, SR, OCR 결과를 Web UI에서 단계별 확인 |

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| Frontend | React, JavaScript, Vite, CSS Modules |
| Backend | Python, FastAPI, Uvicorn |
| Image / Video Processing | OpenCV, NumPy, Pillow |
| Deep Learning Framework | PyTorch, torchvision |
| Plate Detection | Ultralytics YOLOv8-seg |
| Denoising | NAFNet 기반 3채널 컬러 디노이징 |
| Super-Resolution | RGDiffSR, VQGAN, PARSeq |
| OCR | GP-LPR 기반 번호판 문자 인식 |
| Collaboration | Git, GitHub branch / PR workflow |

## 저장소 구조

```text
KNU-LP/
├── FE/                  # React + Vite 기반 시연용 Web UI
├── BE/                  # FastAPI 기반 백엔드 및 통합 AI 파이프라인
│   ├── api/             # API route
│   ├── configs/         # 모델 및 파이프라인 설정
│   ├── pipeline/        # YOLO, denoiser, SR, OCR adapter
│   └── vendors/         # 외부 모델 구현 코드
├── AI/                  # 과업별 실험, 학습, 검증 코드
│   ├── Task_1/          # 전처리: 번호판 crop, denoising
│   ├── Task_2/          # Super-Resolution
│   └── Task_3/          # OCR
├── rule.md              # 팀 브랜치/커밋 규칙
└── README.md            # 프로젝트 소개 및 실행 안내
```

실제 시연용 통합 코드는 `FE/`, `BE/`에 위치하며, `AI/Task_*` 디렉토리는 각 팀원이 수행한 모델 실험, 학습, 검증 과정을 보존하기 위한 공간입니다.

## 실행 방법

### 1. 모델 가중치 배치

대용량 모델 가중치는 GitHub 저장소에 포함하지 않습니다.

전체 파이프라인을 실행하려면 필요한 weight 파일을 수동으로 배치해야 하며, 정확한 파일명과 위치는 [BE/WEIGHTS.md](BE/WEIGHTS.md)를 기준으로 확인합니다.

주요 weight 예시는 다음과 같습니다.

```text
BE/pipeline/plate_cropper/weights/best.pt
BE/pipeline/denoiser/weights/best_model.pt
BE/pipeline/superresolution/weights/SR.ckpt
BE/pipeline/superresolution/weights/VQGAN.ckpt
BE/pipeline/superresolution/weights/parseq.pt
BE/pipeline/ocr/weights/best_model.pth
```

### 2. Backend 실행

```bash
cd BE
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Windows 환경에서는 `BE/run_backend.bat`을 사용할 수 있습니다.

### 3. Frontend 실행

```bash
cd FE
npm install
npm run dev
```

Frontend의 `/api` 요청은 Vite proxy 설정을 통해 기본적으로 `http://localhost:8000`의 Backend로 전달됩니다.

## 구현 상태

| 항목 | 상태 |
|---|---|
| 이미지 / 영상 업로드 | 구현 완료 |
| 영상 구간 선택 및 프레임 추출 | 구현 완료 |
| YOLO 번호판 검출 및 crop | 구현 완료 |
| 번호판 후보 상위 5개 선정 | 구현 완료 |
| NAFNet 컬러 디노이징 | 구현 완료 |
| SR 자동 분기 및 RGDiffSR 연동 | 구현 완료 |
| GP-LPR OCR 연동 | 구현 완료 |
| OCR ensemble voting | 구현 완료 |
| Web UI 단계별 결과 시각화 | 구현 완료 |

## 참고 문서

- Backend 실행 및 API 설명: [BE/readme.md](BE/readme.md)
- 모델 가중치 배치 위치: [BE/WEIGHTS.md](BE/WEIGHTS.md)
- Frontend 실행 및 UI 동작: [FE/readme.md](FE/readme.md)
- AI 과업별 실험 코드: [AI/readme.md](AI/readme.md)
- 팀 작업 규칙: [rule.md](rule.md)

## 참고사항

- 본 저장소는 과목 최종 제출을 위한 산출물로, 실제 시연용 통합 파이프라인과 과업별 실험 기록을 함께 포함합니다.
- 모델 weight 파일은 용량 문제로 제외되어 있으므로, 전체 기능 시연 시에는 `BE/WEIGHTS.md`에 맞춰 별도 배치가 필요합니다.
- `BE/vendors/`에는 RGDiffSR, GP-LPR 등 외부 모델 구현 코드가 포함되어 있으며, 서비스 코드와 분리하여 관리했습니다.
- `AI/Task_*` 디렉토리는 팀원별 실험 및 검증 과정을 보여주기 위한 기록 성격이 강하며, 최종 Web UI 시연은 `FE/`와 `BE/`를 기준으로 실행합니다.
