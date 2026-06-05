# KNU-LP

저화질 CCTV 환경에서도 강건하게 동작하는 번호판 인식 파이프라인 데모입니다.

전체 흐름은 다음과 같습니다.

```text
이미지/영상 입력
-> 영상이면 프레임 추출
-> YOLO 번호판 검출 및 crop
-> 원본 번호판 bbox 면적 기준 상위 5개 후보 선택
-> DnCNN denoising
-> SR 또는 OCR ensemble 분기
-> OCR 결과 표시
```

## 빠른 실행 순서

1. **BE 가중치 파일 다운로드 및 수동 배치**
   * 대용량 가중치 파일들은 `.gitignore` 처리되어 레포지토리에 포함되어 있지 않습니다.
   * 정확한 파일명과 배치 구조는 [BE/WEIGHTS.md](BE/WEIGHTS.md)를 기준으로 확인해 주세요.
   * `BE/vendors/`는 외부 모델 구현 코드 위치이며, weight 파일 위치가 아닙니다.
2. **BE 서버 실행**
   * `BE` 디렉터리로 이동 후 패키지 의존성 설치: `pip install -r requirements.txt`
   * 서버 실행: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` (또는 `run_backend.bat` 실행)
   * 자세한 설명: [BE/readme.md](BE/readme.md)
3. **FE 개발 서버 실행**
   * `FE` 디렉터리로 이동 후 패키지 의존성 설치: `npm install`
   * 서버 실행: `npm run dev`
   * 자세한 설명: [FE/readme.md](FE/readme.md)

## 구현 상태

| 항목 | 상태 | 설명 |
|---|---|---|
| 이미지/영상 업로드 및 프레임 추출 | 구현 완료 | 영상 구간 추출 프레임 YOLO 파이프라인 연동 |
| YOLO 번호판 검출 및 크롭 | 구현 완료 | 실제 YOLOv8-seg 기반 크롭 모델 탑재 (`best.pt`) |
| 후보군 5장 자동 선정 | 구현 완료 | YOLO 바운딩 박스 면적 기준 정렬 필터링 |
| 3채널 컬러 DnCNN Denoising | 구현 완료 | 3채널 컬러 이미지용 노이즈 보정 모델 탑재 (`best_model.pt`) |
| Super-Resolution (SR) | 구현 완료 | RGDiffSR 모델 adapter 및 가중치 로드 확인 (`SR.ckpt`, `VQGAN.ckpt`) |
| OCR (Text Recognition) | 구현 완료 | GP-LPR 기반 번호판 텍스트 해독 모델 연동 (`best_model.pth`) |
| OCR Ensemble Voting | 구현 완료 | 다중 프레임 인식 결과의 캐릭터 단위 Majority Voting 처리 |

## 참고 문서

- 백엔드 실행 및 가중치 위치: [BE/readme.md](BE/readme.md), [BE/WEIGHTS.md](BE/WEIGHTS.md)
- 프론트엔드 실행 및 UI 동작: [FE/readme.md](FE/readme.md)
- 팀 공동 작업 규칙: [rule.md](rule.md)
