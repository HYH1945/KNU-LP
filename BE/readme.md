# BE

KNU 번호판 인식 데모용 FastAPI 백엔드 스켈레톤입니다.

## 실행

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

`FE/src/constants/dummy.js` 의 `USE_DUMMY` 를 `false` 로 바꾸면 프런트엔드가 `/api/analyze` 를 호출합니다.

## 동작 요약

- `POST /api/analyze`
- `multipart/form-data` 의 `files` 필드로 이미지 5장을 받습니다.
- 5장이 아니거나 비이미지 파일이 포함되면 `HTTP 400` 을 반환합니다.
- 정상 요청이면 각 파이프라인 단계가 업로드 이미지를 data URL 로 그대로 되돌리고, OCR 은 `"00가 0000"` 을 반환합니다.
