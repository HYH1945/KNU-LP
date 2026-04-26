# FE

KNU 번호판 인식 데모용 React + Vite 프런트엔드입니다.

## 실행

```bash
npm install
npm run dev
```

## 빌드

```bash
npm run build
npm run preview
```

## 동작 요약

- 업로드 슬롯은 5개 고정입니다.
- 배치 업로드는 빈 슬롯부터 순서대로 채웁니다.
- `USE_DUMMY = true`이면 `placeholder1.png`~`placeholder5.png` 풀에서 랜덤 선택한 로컬 더미 응답으로 전체 UI를 검증할 수 있습니다.
- `USE_DUMMY = false`이면 `POST /api/analyze`로 5개 파일을 전송합니다.
