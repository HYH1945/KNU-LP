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

1. BE 가중치 파일을 지정된 위치에 넣습니다.
   - `BE/pipeline/plate_cropper/weights/best.pt`
   - `BE/pipeline/denoiser/weights/best_model.pt`
2. BE 서버를 실행합니다.
   - 자세한 내용: `BE/readme.md`
3. FE 개발 서버를 실행합니다.
   - 자세한 내용: `FE/readme.md`

## 구현 상태

| 항목 | 상태 |
|---|---|
| 이미지 업로드 처리 | 구현됨 |
| 영상 업로드 처리 | 구현됨 |
| 영상 프레임 추출 | 구현됨 |
| YOLO 번호판 crop/highlight | 실제 모델 연결됨 |
| 5장 선정 | YOLO 이후 original plate bbox 면적 기준으로 구현됨 |
| INPUT preview | 앞 5장 표시 및 `+N` 생략 표시 구현됨 |
| HR Width/Height 기반 자동 분기 | original bbox area 기준으로 구현됨 |
| DnCNN denoiser | 실제 가중치로 동작 확인됨 |
| SR | 현재 bicubic resize stub |
| OCR | 현재 stub |
| OCR ensemble | 구조 구현됨. 실제 OCR 연결 후 의미 있는 voting 가능 |

## 참고 문서

- 백엔드 실행 및 가중치 위치: `BE/readme.md`
- 프론트엔드 실행 및 UI 동작: `FE/readme.md`
- 팀 규칙: `rule.md`
