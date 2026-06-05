# BE Weight 배치

모델 weight 파일은 Git에 올리지 않고 각자 로컬에 배치합니다.

```text
BE/pipeline/plate_cropper/weights/best.pt
BE/pipeline/denoiser/weights/best_model.pt
BE/pipeline/superresolution/weights/SR.ckpt
BE/pipeline/superresolution/weights/VQGAN.ckpt
BE/pipeline/superresolution/weights/parseq.pt
BE/pipeline/ocr/weights/best_model.pth
```

- `best.pt`: YOLO 번호판 검출/crop 모델
- `best_model.pt`: 컬러 DnCNN denoiser
- `SR.ckpt`: RGDiffSR 메인 checkpoint
- `VQGAN.ckpt`: RGDiffSR first-stage VQGAN checkpoint
- `parseq.pt`: SR conditioning용 PARSeq checkpoint
- `best_model.pth`: GP-LPR OCR checkpoint

현재 기본 설정은 `BE/configs/settings.yaml`에 있습니다.

## Vendor 코드 위치

외부 모델 구현체는 아래 위치에 둡니다.

```text
BE/vendors/rgdiffsr
BE/vendors/gplpr
BE/vendors/taming
```

weight 파일 위치와 vendor 코드 위치는 다릅니다. weight는 위의 `weights/` 폴더에 두고, 외부 코드 구현체는 `BE/vendors/`에 둡니다.
