# BE Weight 배치

모델 weight 파일은 Git에 올리지 않고 각자 로컬에 배치합니다. 아래 구조와 파일명을 그대로 맞춰 주세요.

```text
BE/
├── yolov8n-seg.pt (선택: 전달받은 YOLOv8 기본 가중치, 현재 기본 설정에서는 직접 참조하지 않음)
├── pipeline/
│   ├── plate_cropper/
│   │   └── weights/
│   │       └── best.pt (YOLO 번호판 검출/crop 모델)
│   ├── denoiser/
│   │   └── weights/
│   │       └── best_model.pt (3채널 컬러 DnCNN denoiser)
│   ├── ocr/
│   │   └── weights/
│   │       └── best_model.pth (GP-LPR OCR checkpoint)
│   └── superresolution/
│       └── weights/
│           ├── SR.ckpt (RGDiffSR 메인 checkpoint)
│           ├── VQGAN.ckpt (RGDiffSR first-stage VQGAN checkpoint)
│           └── parseq.pt (SR conditioning용 PARSeq checkpoint)
└── vendors/
    ├── rgdiffsr/ (RGDiffSR, LDM, PARSeq, text_super_resolution 외부 코드)
    ├── gplpr/    (GP-LPR OCR 외부 코드)
    └── taming/   (taming-transformers 호환 코드)
```

현재 기본 설정은 `BE/configs/settings.yaml`에 있습니다.

## 현재 설정이 직접 참조하는 weight

| 모듈 | 설정 키 | 파일 위치 |
|---|---|---|
| YOLO crop | `yolo.paths.weights` | `BE/pipeline/plate_cropper/weights/best.pt` |
| DnCNN denoiser | `denoiser.paths.weights` | `BE/pipeline/denoiser/weights/best_model.pt` |
| SR main | `superresolution.paths.checkpoint` | `BE/pipeline/superresolution/weights/SR.ckpt` |
| SR VQGAN | `superresolution.paths.vqgan` | `BE/pipeline/superresolution/weights/VQGAN.ckpt` |
| SR PARSeq | `superresolution.paths.parseq` | `BE/pipeline/superresolution/weights/parseq.pt` |
| OCR | `ocr.paths.checkpoint` | `BE/pipeline/ocr/weights/best_model.pth` |

## Vendor 코드 위치

외부 모델 구현체는 아래 위치에 둡니다.

```text
BE/vendors/rgdiffsr
BE/vendors/gplpr
BE/vendors/taming
```

weight 파일 위치와 vendor 코드 위치는 다릅니다. weight는 `BE/pipeline/**/weights/` 아래에 두고, 외부 코드 구현체는 `BE/vendors/`에 둡니다.
