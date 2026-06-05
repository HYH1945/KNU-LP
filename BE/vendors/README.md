# BE Vendors

이 폴더는 외부 연구 코드와 모델 구현체를 모아두는 위치입니다.

```text
rgdiffsr/   RGDiffSR, LDM, PARSeq, text_super_resolution
gplpr/      GP-LPR OCR 모델 코드
taming/     taming-transformers 호환 코드
```

서비스 파이프라인 코드는 `BE/pipeline/`에 두고, 외부 코드에는 adapter를 통해 접근합니다. weight 파일은 이 폴더가 아니라 `BE/pipeline/**/weights/` 위치에 둡니다.
